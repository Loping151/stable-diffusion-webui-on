#!/usr/bin/env python
"""
xwsdwebui: standalone diffusers/torch inference for the Anima architecture
(CircleStone Labs / Comfy Org "Anima", a NVIDIA Cosmos-Predict2-2B finetune).

Forge's backend cannot load Anima (it is a Cosmos DiT with a Qwen3-0.6B text
encoder + an LLM adapter + the Qwen-Image/Wan2.1 VAE). This script assembles the
pipeline by hand: transformers Qwen3 + a from-scratch port of ComfyUI's
`comfy/ldm/anima/model.py` and `comfy/ldm/cosmos/predict2.py` + diffusers'
AutoencoderKLWan. Spec source: ComfyUI master.

Usage:
  ./venv/bin/python anima_infer.py --check          # just verify weights load
  CUDA_VISIBLE_DEVICES=0 ./venv/bin/python anima_infer.py \
      --prompt "1girl, silver hair, kimono, maple" --steps 30 --cfg 5 --size 1024
"""
import argparse, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

ANIMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "anima")
DIT_PATH = os.path.join(ANIMA, "diffusion_models", "anima-base-v1.0.safetensors")
TE_PATH  = os.path.join(ANIMA, "text_encoders", "qwen_3_06b_base.safetensors")
VAE_PATH = os.path.join(ANIMA, "vae", "qwen_image_vae.safetensors")
QWEN_TOK = os.path.join(ANIMA, "tokenizer")
T5_TOK   = os.path.join(ANIMA, "t5_tokenizer")

LAT_MEAN = [-0.7571,-0.7089,-0.9113,0.1075,-0.1745,0.9653,-0.1517,1.5508,
            0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,0.2503,-0.2921]
LAT_STD  = [2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,2.6052,2.0743,
            3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.9160]


# ----------------------------- primitives -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        d = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(d)


import os as _os
_ROPE_MODE = _os.environ.get("ANIMA_ROPE", "splithalf")

def rope_rotate_half(x, cos, sin):
    # x: [..., D]; cos/sin: [..., D/2]
    if _ROPE_MODE == "interleave":
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        out = torch.stack([o1, o2], dim=-1).flatten(-2)
        return out
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def timestep_sinusoid(t, dim=2048, max_period=10000.0):
    # ComfyUI Timesteps: cos first then sin
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


# ----------------------------- DiT (MiniTrainDIT) -----------------------------
class DiTAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, heads, head_dim, use_rope):
        super().__init__()
        self.h, self.hd, self.use_rope = heads, head_dim, use_rope
        self.q_proj = nn.Linear(q_dim, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, heads * head_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, heads * head_dim, bias=False)
        self.output_proj = nn.Linear(heads * head_dim, q_dim, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, ctx=None, rope=None):
        B, Lq, _ = x.shape
        kv = x if ctx is None else ctx
        Lk = kv.shape[1]
        q = self.q_norm(self.q_proj(x).view(B, Lq, self.h, self.hd))
        k = self.k_norm(self.k_proj(kv).view(B, Lk, self.h, self.hd))
        v = self.v_proj(kv).view(B, Lk, self.h, self.hd)
        if self.use_rope and rope is not None:
            cos, sin = rope
            q = rope_rotate_half(q, cos, sin)
            k = rope_rotate_half(k, cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # B,h,L,hd
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, Lq, self.h * self.hd)
        return self.output_proj(o)


class GPT2FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden, bias=False)
        self.layer2 = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.layer2(F.gelu(self.layer1(x)))


def adaln_seq(dim, lora):
    return nn.Sequential(nn.SiLU(),
                         nn.Linear(dim, lora, bias=False),
                         nn.Linear(lora, 3 * dim, bias=False))


class DiTBlock(nn.Module):
    def __init__(self, dim=2048, ctx=1024, heads=16, mlp_ratio=4.0, lora=256):
        super().__init__()
        hd = dim // heads
        self.layer_norm_self_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = DiTAttention(dim, dim, heads, hd, use_rope=True)
        self.layer_norm_cross_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = DiTAttention(dim, ctx, heads, hd, use_rope=False)
        self.layer_norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(dim, int(dim * mlp_ratio))
        self.adaln_modulation_self_attn = adaln_seq(dim, lora)
        self.adaln_modulation_cross_attn = adaln_seq(dim, lora)
        self.adaln_modulation_mlp = adaln_seq(dim, lora)

    def forward(self, x, emb, ctx, adaln_lora, rope):
        def mod(layer):
            return (layer(emb) + adaln_lora).chunk(3, dim=-1)
        s_sa, sc_sa, g_sa = mod(self.adaln_modulation_self_attn)
        s_ca, sc_ca, g_ca = mod(self.adaln_modulation_cross_attn)
        s_ml, sc_ml, g_ml = mod(self.adaln_modulation_mlp)
        x = x + g_sa * self.self_attn(self.layer_norm_self_attn(x) * (1 + sc_sa) + s_sa, rope=rope)
        x = x + g_ca * self.cross_attn(self.layer_norm_cross_attn(x) * (1 + sc_ca) + s_ca, ctx=ctx)
        x = x + g_ml * self.mlp(self.layer_norm_mlp(x) * (1 + sc_ml) + s_ml)
        return x


class TimestepMLP(nn.Module):
    def __init__(self, dim=2048):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim, bias=False)
        self.linear_2 = nn.Linear(dim, 3 * dim, bias=False)
    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


class XEmbedder(nn.Module):
    def __init__(self, in_dim=68, dim=2048):
        super().__init__()
        self.proj = nn.Sequential(nn.Identity(), nn.Linear(in_dim, dim, bias=False))
    def forward(self, x):
        return self.proj(x)


class FinalLayer(nn.Module):
    def __init__(self, dim=2048, out=64, lora=256):
        super().__init__()
        self.adaln_modulation = nn.Sequential(nn.SiLU(),
                                              nn.Linear(dim, lora, bias=False),
                                              nn.Linear(lora, 2 * dim, bias=False))
        self.linear = nn.Linear(dim, out, bias=False)
    def forward(self, x, emb, adaln_lora_2):
        shift, scale = (self.adaln_modulation(emb) + adaln_lora_2).chunk(2, dim=-1)
        x = F.layer_norm(x, (x.shape[-1],), eps=1e-6) * (1 + scale) + shift
        return self.linear(x)


class LLMAdapterBlock(nn.Module):
    def __init__(self, dim=1024, heads=16, head_dim=64, mlp_ratio=4.0):
        super().__init__()
        self.norm_self_attn = RMSNorm(dim)
        self.self_attn = AdapterAttention(dim, heads, head_dim)
        self.norm_cross_attn = RMSNorm(dim)
        self.cross_attn = AdapterAttention(dim, heads, head_dim)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio), bias=True),
                                 nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim, bias=True))
    def forward(self, x, ctx, rope_q, rope_kv):
        x = x + self.self_attn(self.norm_self_attn(x), rope_q=rope_q, rope_kv=rope_q)
        x = x + self.cross_attn(self.norm_cross_attn(x), ctx=ctx, rope_q=rope_q, rope_kv=rope_kv)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class AdapterAttention(nn.Module):
    def __init__(self, dim, heads, head_dim):
        super().__init__()
        self.h, self.hd = heads, head_dim
        self.q_proj = nn.Linear(dim, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, dim, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
    def forward(self, x, ctx=None, rope_q=None, rope_kv=None):
        B, Lq, _ = x.shape
        kv = x if ctx is None else ctx
        Lk = kv.shape[1]
        q = self.q_norm(self.q_proj(x).view(B, Lq, self.h, self.hd))
        k = self.k_norm(self.k_proj(kv).view(B, Lk, self.h, self.hd))
        v = self.v_proj(kv).view(B, Lk, self.h, self.hd)
        if rope_q is not None:
            q = rope_rotate_half(q, rope_q[0], rope_q[1])
            k = rope_rotate_half(k, rope_kv[0], rope_kv[1])
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(o.transpose(1, 2).reshape(B, Lq, self.h * self.hd))


class LLMAdapter(nn.Module):
    def __init__(self, dim=1024, layers=6, heads=16, head_dim=64, vocab=32128):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([LLMAdapterBlock(dim, heads, head_dim) for _ in range(layers)])
        self.norm = RMSNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.head_dim = head_dim
    def forward(self, t5_ids, qwen_hidden):
        x = self.embed(t5_ids)
        rope_q = make_llama_rope(x.shape[1], self.head_dim, x.device, x.dtype)
        rope_kv = make_llama_rope(qwen_hidden.shape[1], self.head_dim, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, qwen_hidden, rope_q, rope_kv)
        return self.out_proj(self.norm(x))


class MiniTrainDIT(nn.Module):
    def __init__(self, dim=2048, blocks=28, heads=16, ctx=1024, lora=256):
        super().__init__()
        self.x_embedder = XEmbedder(68, dim)
        self.t_embedder = nn.Sequential(nn.Identity(), TimestepMLP(dim))
        self.t_embedding_norm = RMSNorm(dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, ctx, heads, 4.0, lora) for _ in range(blocks)])
        self.final_layer = FinalLayer(dim, 64, lora)
        self.llm_adapter = LLMAdapter(ctx)
        self.dim, self.heads = dim, heads

    def forward(self, lat, sigma, ctx, rope):
        # lat: [B,16,1,H,W]
        B, C, T, H, W = lat.shape
        pad = torch.zeros(B, 1, T, H, W, device=lat.device, dtype=lat.dtype)
        x = torch.cat([lat, pad], dim=1)  # B,17,T,H,W
        # patchify 2x2: "b c (h m) (w n) -> b (h w) (c m n)" for T=1
        x = x.squeeze(2)  # B,17,H,W
        Hp, Wp = H // 2, W // 2
        x = x.view(B, 17, Hp, 2, Wp, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, Hp * Wp, 17 * 4)
        x = self.x_embedder(x)  # B,L,dim
        import os as _o
        _ts = float(_o.environ.get("ANIMA_TSCALE", "1"))
        sinus = timestep_sinusoid(sigma * _ts, self.dim).to(x.dtype)  # B,dim
        adaln_lora = self.t_embedder[1](sinus)            # B,3*dim
        emb = self.t_embedding_norm(sinus)                # B,dim
        adaln_lora = adaln_lora[:, None, :].expand(B, x.shape[1], -1) if False else adaln_lora[:, None, :]
        emb = emb[:, None, :]
        for blk in self.blocks:
            x = blk(x, emb, ctx, adaln_lora, rope)
        x = self.final_layer(x, emb, adaln_lora[..., : 2 * self.dim])  # B,L,64
        # unpatchify: 64 = (p1 p2 t C)=(2 2 1 16)
        x = x.view(B, Hp, Wp, 2, 2, 16).permute(0, 5, 1, 3, 2, 4).reshape(B, 16, Hp * 2, Wp * 2)
        return x.unsqueeze(2)  # B,16,1,H,W


# ----------------------------- rope builders -----------------------------
def make_llama_rope(L, head_dim, device, dtype, theta=10000.0):
    pairs = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(L, device=device).float()
    a = torch.outer(pos, freqs)  # L,pairs
    cos = a.cos()[None, :, None, :].to(dtype)
    sin = a.sin()[None, :, None, :].to(dtype)
    return cos, sin


def make_rope3d(H, W, device, dtype, head_dim=128):
    # pairs: t=22, h=21, w=21 (T=1 so temporal angle = 0)
    dim_t, dim_h, dim_w = 44, 42, 42
    theta_t = 10000.0
    theta_h = 10000.0 * (4.0 ** (dim_h / (dim_h - 2)))   # ~42869
    theta_w = theta_h
    ft = 1.0 / (theta_t ** (torch.arange(0, dim_t, 2, device=device).float() / dim_t))  # 22
    fh = 1.0 / (theta_h ** (torch.arange(0, dim_h, 2, device=device).float() / dim_h))  # 21
    fw = 1.0 / (theta_w ** (torch.arange(0, dim_w, 2, device=device).float() / dim_w))  # 21
    h_idx = torch.arange(H, device=device).float()
    w_idx = torch.arange(W, device=device).float()
    ang_t = torch.zeros(H * W, ft.shape[0], device=device)                  # T=1 -> 0
    ang_h = torch.outer(h_idx, fh)[:, None, :].expand(H, W, fh.shape[0]).reshape(H * W, -1)
    ang_w = torch.outer(w_idx, fw)[None, :, :].expand(H, W, fw.shape[0]).reshape(H * W, -1)
    ang = torch.cat([ang_t, ang_h, ang_w], dim=-1)        # L,64
    cos = ang.cos()[None, :, None, :].to(dtype)
    sin = ang.sin()[None, :, None, :].to(dtype)
    return cos, sin


# ----------------------------- loading -----------------------------
def load_dit(device, dtype):
    sd = load_file(DIT_PATH)
    sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}
    m = MiniTrainDIT()
    miss, unexp = m.load_state_dict(sd, strict=False)
    return m.to(device, dtype).eval(), miss, unexp


def _resolve(root, path):
    m = root
    for p in path.split('.'):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m


def merge_lora(dit, path, scale=1.0):
    """Merge an Anima-format LoRA (diffusion_model.<module>.lora_A/lora_B, rank r) into the DiT
    weights in-place: W += scale * (B @ A). Supports the main blocks and the llm_adapter."""
    sd = load_file(path)
    names = sorted({k.rsplit(".lora_", 1)[0] for k in sd if ".lora_" in k})
    merged = 0
    for name in names:
        A = sd.get(name + ".lora_A.weight")
        B = sd.get(name + ".lora_B.weight")
        if A is None or B is None:
            continue
        mod_path = name[len("diffusion_model."):] if name.startswith("diffusion_model.") else name
        try:
            mod = _resolve(dit, mod_path)
        except (AttributeError, IndexError, ValueError):
            continue
        delta = (B.float() @ A.float()) * scale
        with torch.no_grad():
            mod.weight.add_(delta.to(mod.weight.device, mod.weight.dtype))
        merged += 1
    return merged, len(names)


def load_qwen3(device, dtype):
    from transformers import Qwen3Config, Qwen3Model
    sd = load_file(TE_PATH)
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    cfg = Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                      num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                      head_dim=128, max_position_embeddings=40960, rms_norm_eps=1e-6,
                      rope_theta=1e6, tie_word_embeddings=True)
    m = Qwen3Model(cfg)
    m.load_state_dict(sd, strict=True)
    return m.to(device, dtype).eval()


def load_vae(device):
    # kept on CPU by default: the Wan 3D VAE decode is very memory-hungry and would OOM
    # alongside the 2B DiT on a 24GB card.
    from diffusers import AutoencoderKLWan
    return AutoencoderKLWan.from_single_file(VAE_PATH, torch_dtype=torch.float32).eval()


# ----------------------------- text encoding -----------------------------
def encode_text(prompt, qwen_tok, t5_tok, qwen, device, dtype):
    qids = qwen_tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        qh = qwen(input_ids=qids).last_hidden_state.to(dtype)   # B,Lq,1024 (final-norm applied)
    t5 = t5_tok(prompt, return_tensors="pt").input_ids.to(device)   # appends </s>
    return qh, t5


# ----------------------------- sampling -----------------------------
@torch.no_grad()
def sample(dit, ctx_pos, ctx_neg, H, W, steps, cfg, seed, device, dtype, shift=3.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, 16, 1, H, W, generator=g).to(device, dtype)
    rope = make_rope3d(H // 2, W // 2, device, dtype)
    # rectified-flow sigmas with shift
    t = torch.linspace(1, 0, steps + 1)
    sig = (shift * t) / (1 + (shift - 1) * t)
    sig = sig.to(device)
    def velocity(x, s):
        st = s.expand(1).to(dtype)
        vp = dit(x, st, ctx_pos, rope)
        vn = dit(x, st, ctx_neg, rope)
        return vn + cfg * (vp - vn)
    mode = os.environ.get("ANIMA_SAMPLER", "euler")
    gn = torch.Generator(device="cpu").manual_seed(seed + 1)
    for i in range(steps):
        s, sn = sig[i], sig[i + 1]
        v = velocity(x, s)
        denoised = x - s * v
        if mode == "sde" and sn > 0:
            # fully-stochastic flow step: jump to x0 estimate, renoise to sigma_next
            noise = torch.randn(x.shape, generator=gn).to(x.device, x.dtype)
            x = (1 - sn) * denoised + sn * noise
        elif mode == "heun" and sn > 0:
            x_e = x + (sn - s) * v
            v2 = velocity(x_e, sn)
            x = x + (sn - s) * 0.5 * (v + v2)
        else:
            x = x + (sn - s) * v
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="masterpiece, best quality, score_7, safe, 1girl, silver hair, blue eyes, kimono, autumn maple leaves, detailed")
    ap.add_argument("--neg", default="worst quality, low quality, score_1, score_2, score_3, artist name")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="anima_out.png")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--lora", default=None, help="path to an Anima-format LoRA .safetensors")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    args = ap.parse_args()

    device, dtype = "cuda", (torch.float32 if args.fp32 else torch.bfloat16)
    dit, miss, unexp = load_dit(device, dtype)
    print(f"DiT load: missing={len(miss)} unexpected={len(unexp)}")
    if args.lora:
        m, n = merge_lora(dit, args.lora, args.lora_scale)
        print(f"LoRA merged: {m}/{n} modules from {os.path.basename(args.lora)} (scale {args.lora_scale})")
    if miss: print("  missing[:8]:", miss[:8])
    if unexp: print("  unexpected[:8]:", unexp[:8])
    if args.check:
        return

    from transformers import AutoTokenizer
    qwen_tok = AutoTokenizer.from_pretrained(QWEN_TOK)
    t5_tok = AutoTokenizer.from_pretrained(T5_TOK)
    qwen = load_qwen3(device, dtype)
    vae = load_vae(device)

    nopad = os.environ.get("ANIMA_NOPAD", "0") == "1"
    def ctx(p):
        qh, t5 = encode_text(p, qwen_tok, t5_tok, qwen, device, dtype)
        c = dit.llm_adapter(t5, qh)
        if not nopad and c.shape[1] < 512:
            c = F.pad(c, (0, 0, 0, 512 - c.shape[1]))
        return c
    ctx_pos, ctx_neg = ctx(args.prompt), ctx(args.neg)

    H = W = args.size // 8
    print(f"ctx_pos {tuple(ctx_pos.shape)} mean {ctx_pos.float().mean():.3f} std {ctx_pos.float().std():.3f}")
    z = sample(dit, ctx_pos, ctx_neg, H, W, args.steps, args.cfg, args.seed, device, dtype)
    print(f"final z: mean {z.float().mean():.3f} std {z.float().std():.3f} min {z.float().min():.2f} max {z.float().max():.2f}")
    import gc
    del dit, qwen
    gc.collect()
    torch.cuda.empty_cache()
    mean = torch.tensor(LAT_MEAN).view(1, 16, 1, 1, 1)
    std = torch.tensor(LAT_STD).view(1, 16, 1, 1, 1)
    lat = (z.float().cpu() * std + mean)
    with torch.no_grad():
        img = vae.decode(lat).sample  # B,3,1,H*8,W*8 (on CPU)
    img = ((img[0, :, 0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    from PIL import Image
    Image.fromarray(img).save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
