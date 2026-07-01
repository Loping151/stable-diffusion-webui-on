"""
Anima (CircleStone Labs / Comfy Org) — NVIDIA Cosmos-Predict2-2B DiT with a Qwen3-0.6B
text encoder + LLM adapter, integrated into Forge's backend.

This is a forge-native port of the reference implementation validated in `anima_infer.py`
(bit-exact vs ComfyUI master). It uses plain torch nn.* so `using_forge_operations`
swaps in Forge's quant/memory-managed operations at load time. The forward signature
matches Forge's k_model contract: forward(x, timestep, context, **extra_conds).

The Qwen3 text encoder and the Wan/Qwen-Image VAE are separate components wired by the
Anima diffusion engine; the LLM adapter (`net.llm_adapter.*`) lives inside this model and
runs in `preprocess_text_embeds`, exactly like ComfyUI's `comfy/ldm/anima/model.py`.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def rope_rotate_half(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def timestep_sinusoid(t, dim=2048, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


def make_rope3d(H, W, device, dtype, head_dim=128):
    dim_t, dim_h, dim_w = 44, 42, 42
    theta_t = 10000.0
    theta_h = 10000.0 * (4.0 ** (dim_h / (dim_h - 2)))
    theta_w = theta_h
    ft = 1.0 / (theta_t ** (torch.arange(0, dim_t, 2, device=device).float() / dim_t))
    fh = 1.0 / (theta_h ** (torch.arange(0, dim_h, 2, device=device).float() / dim_h))
    fw = 1.0 / (theta_w ** (torch.arange(0, dim_w, 2, device=device).float() / dim_w))
    h_idx = torch.arange(H, device=device).float()
    w_idx = torch.arange(W, device=device).float()
    ang_t = torch.zeros(H * W, ft.shape[0], device=device)
    ang_h = torch.outer(h_idx, fh)[:, None, :].expand(H, W, fh.shape[0]).reshape(H * W, -1)
    ang_w = torch.outer(w_idx, fw)[None, :, :].expand(H, W, fw.shape[0]).reshape(H * W, -1)
    ang = torch.cat([ang_t, ang_h, ang_w], dim=-1)
    return ang.cos()[None, :, None, :].to(dtype), ang.sin()[None, :, None, :].to(dtype)


def make_llama_rope(L, head_dim, device, dtype, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    a = torch.outer(torch.arange(L, device=device).float(), freqs)
    return a.cos()[None, :, None, :].to(dtype), a.sin()[None, :, None, :].to(dtype)


# ---------------- DiT ----------------
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
            q = rope_rotate_half(q, rope[0], rope[1])
            k = rope_rotate_half(k, rope[0], rope[1])
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v)
        return self.output_proj(o.transpose(1, 2).reshape(B, Lq, self.h * self.hd))


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
        s, sc, g = mod(self.adaln_modulation_self_attn)
        x = x + g * self.self_attn(self.layer_norm_self_attn(x) * (1 + sc) + s, rope=rope)
        s, sc, g = mod(self.adaln_modulation_cross_attn)
        x = x + g * self.cross_attn(self.layer_norm_cross_attn(x) * (1 + sc) + s, ctx=ctx)
        s, sc, g = mod(self.adaln_modulation_mlp)
        x = x + g * self.mlp(self.layer_norm_mlp(x) * (1 + sc) + s)
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


# ---------------- LLM adapter ----------------
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
        return self.norm(self.out_proj(x))   # order matters: out_proj THEN norm


class AnimaNet(nn.Module):
    """The `net.*` submodule of the Anima checkpoint (Cosmos-Predict2 DiT + LLM adapter)."""

    def __init__(self, in_channels=16, out_channels=16, model_channels=2048, num_blocks=28,
                 num_heads=16, mlp_ratio=4.0, crossattn_emb_channels=1024, patch_spatial=2,
                 patch_temporal=1, adaln_lora_dim=256, concat_padding_mask=True, **kwargs):
        super().__init__()
        self.dim = model_channels
        self.heads = num_heads
        self.patch = patch_spatial
        self.out_channels = out_channels
        self.concat_padding_mask = concat_padding_mask
        in_ch = in_channels + (1 if concat_padding_mask else 0)
        in_dim = in_ch * patch_spatial * patch_spatial * patch_temporal
        self.x_embedder = XEmbedder(in_dim, model_channels)
        self.t_embedder = nn.Sequential(nn.Identity(), TimestepMLP(model_channels))
        self.t_embedding_norm = RMSNorm(model_channels)
        self.blocks = nn.ModuleList([
            DiTBlock(model_channels, crossattn_emb_channels, num_heads, mlp_ratio, adaln_lora_dim)
            for _ in range(num_blocks)])
        self.final_layer = FinalLayer(model_channels, patch_spatial * patch_spatial * patch_temporal * out_channels, adaln_lora_dim)
        self.llm_adapter = LLMAdapter(crossattn_emb_channels)

    def preprocess_text_embeds(self, qwen_hidden, t5xxl_ids, t5xxl_weights=None):
        out = self.llm_adapter(t5xxl_ids, qwen_hidden)
        if t5xxl_weights is not None:
            out = out * t5xxl_weights
        if out.shape[1] < 512:
            out = F.pad(out, (0, 0, 0, 512 - out.shape[1]))
        return out

    def forward(self, x, timestep, context, t5xxl_ids=None, t5xxl_weights=None,
                control=None, transformer_options={}, **kwargs):
        if t5xxl_ids is not None:
            context = self.preprocess_text_embeds(context, t5xxl_ids, t5xxl_weights)

        squeeze_out = False
        if x.dim() == 4:                       # [B,C,H,W] -> [B,C,1,H,W]
            x = x.unsqueeze(2)
            squeeze_out = True
        B, C, T, H, W = x.shape
        p = self.patch
        if self.concat_padding_mask:
            pad = torch.zeros(B, 1, T, H, W, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        xin = x.squeeze(2)                      # T=1
        Hp, Wp = H // p, W // p
        ch = xin.shape[1]
        xin = xin.view(B, ch, Hp, p, Wp, p).permute(0, 2, 4, 1, 3, 5).reshape(B, Hp * Wp, ch * p * p)
        h = self.x_embedder(xin)

        sinus = timestep_sinusoid(timestep, self.dim).to(h.dtype)
        adaln_lora = self.t_embedder[1](sinus)[:, None, :]
        emb = self.t_embedding_norm(sinus)[:, None, :]
        rope = make_rope3d(Hp, Wp, h.device, h.dtype)

        for blk in self.blocks:
            h = blk(h, emb, context, adaln_lora, rope)
        h = self.final_layer(h, emb, adaln_lora[..., : 2 * self.dim])
        oc = self.out_channels
        h = h.view(B, Hp, Wp, p, p, oc).permute(0, 5, 1, 3, 2, 4).reshape(B, oc, Hp * p, Wp * p)
        out = h.unsqueeze(2)                   # [B,oc,1,H,W]
        if squeeze_out:
            out = out.squeeze(2)
        return out


class IntegratedAnima(nn.Module):
    """Forge diffusion_model wrapper. The checkpoint stores the DiT under `net.*`, so the
    real model lives in self.net. forward(x, timestep, context, t5xxl_ids=..., ...)."""

    def __init__(self, **config):
        super().__init__()
        self.net = AnimaNet(**config)

    def forward(self, x, timestep, context, **kwargs):
        return self.net(x, timestep, context, **kwargs)
