import os
import glob
import torch

# Inject the Anima architecture into the git-ignored huggingface_guess vendor copy before we
# reference model_list.Anima below. Idempotent: a no-op if the vendor copy is already patched.
from backend.nn import anima_hf_register as _anima_hf_register
_anima_hf_register.register()

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.unet import UnetPatcher
from backend.modules.k_prediction import PredictionAnima
from backend import memory_management
from backend.nn.anima import LLMAdapter


# Components that are not part of the single-file DiT checkpoint (Qwen3 text encoder, its
# tokenizer, the T5 tokenizer used by the LLM adapter, and the Wan/Qwen-Image VAE) are loaded
# from this directory. Point ANIMA_ASSETS at another dir to override.
#
# The layout is filename-agnostic: the text encoder and VAE are located by common names and
# glob patterns, so files packaged differently (e.g. Civitai's `anima_baseV10_txt.safetensors`
# and `anima_baseV10.safetensors`) work without renaming. Individual paths can also be pinned
# via ANIMA_TEXT_ENCODER / ANIMA_VAE / ANIMA_QWEN_TOKENIZER / ANIMA_T5_TOKENIZER.
ANIMA_ASSETS = os.environ.get(
    "ANIMA_ASSETS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "anima"),
)

# Tokenizers are small and bundled in the repo so a bare Civitai download (DiT + text encoder +
# VAE, no tokenizer) still works out of the box.
_BUNDLED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "huggingface", "Anima")


def _resolve_file(env_var, dirs, exact_names, globs, what):
    """Locate a component file: explicit env override, then known names, then glob heuristics."""
    v = os.environ.get(env_var)
    if v:
        if os.path.isfile(v):
            return v
        raise FileNotFoundError(f"{env_var}={v!r} is set but not a file")
    for d in dirs:
        for name in exact_names:
            fp = os.path.join(d, name)
            if os.path.isfile(fp):
                return fp
    for d in dirs:
        for pat in globs:
            hits = sorted(glob.glob(os.path.join(d, pat)))
            if hits:
                return hits[0]
    raise FileNotFoundError(
        f"Anima {what} not found. Set {env_var} to its path, or place it under {ANIMA_ASSETS} "
        f"(looked for {exact_names} or {globs})."
    )


def _resolve_dir(env_var, candidates):
    v = os.environ.get(env_var)
    if v:
        if os.path.isdir(v):
            return v
        raise FileNotFoundError(f"{env_var}={v!r} is set but not a directory")
    for d in candidates:
        if os.path.isdir(d) and os.listdir(d):
            return d
    return candidates[-1]  # bundled fallback


class _WanVAEWrapper:
    """Minimal VAE holder so ForgeObjects/save paths don't crash; real work is in the engine."""
    latent_channels = 16

    def __init__(self, model):
        self.first_stage_model = model

    def shallow_copy(self):
        return _WanVAEWrapper(self.first_stage_model)


class Anima(ForgeDiffusionEngine):
    matched_guesses = [model_list.Anima]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False
        self.is_anima = True  # arch marker (base engine leaves it unset); used by e.g. Anima Turbo

        from transformers import AutoTokenizer, Qwen3Config, Qwen3Model
        from diffusers import AutoencoderKLWan
        from safetensors.torch import load_file

        self.device = memory_management.get_torch_device()
        self.te_dtype = memory_management.text_encoder_dtype()

        # --- text encoders / tokenizers ---
        self.qwen_tok = AutoTokenizer.from_pretrained(_resolve_dir(
            "ANIMA_QWEN_TOKENIZER",
            [os.path.join(ANIMA_ASSETS, "tokenizer"), os.path.join(_BUNDLED_DIR, "tokenizer")]))
        self.t5_tok = AutoTokenizer.from_pretrained(_resolve_dir(
            "ANIMA_T5_TOKENIZER",
            [os.path.join(ANIMA_ASSETS, "t5_tokenizer"), os.path.join(_BUNDLED_DIR, "t5_tokenizer")]))

        te_path = _resolve_file(
            "ANIMA_TEXT_ENCODER",
            dirs=[os.path.join(ANIMA_ASSETS, "text_encoders"), ANIMA_ASSETS],
            exact_names=["qwen_3_06b_base.safetensors", "anima_baseV10_txt.safetensors"],
            globs=["*txt*.safetensors", "*qwen*3*.safetensors", "qwen_3*.safetensors"],
            what="text encoder")
        qsd = load_file(te_path)
        stripped = {k[len("model."):]: v for k, v in qsd.items() if k.startswith("model.")}
        qsd = stripped if stripped else qsd
        qcfg = Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                           num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                           head_dim=128, max_position_embeddings=40960, rms_norm_eps=1e-6,
                           rope_theta=1e6, tie_word_embeddings=True)
        self.qwen = Qwen3Model(qcfg)
        self.qwen.load_state_dict(qsd, strict=True)
        self.qwen = self.qwen.to(memory_management.text_encoder_offload_device(), self.te_dtype).eval()
        self.qwen.requires_grad_(False)

        # --- Wan / Qwen-Image VAE ---
        vae_path = _resolve_file(
            "ANIMA_VAE",
            dirs=[os.path.join(ANIMA_ASSETS, "vae"), ANIMA_ASSETS],
            exact_names=["qwen_image_vae.safetensors"],
            globs=["*vae*.safetensors"],
            what="VAE")
        # Load with a repo-bundled config so from_single_file never reaches out to HuggingFace
        # for the AutoencoderKLWan config (installs behind no proxy / offline would otherwise
        # hang retrying Wan-AI/Wan2.1-T2V-14B-Diffusers).
        vae_model = AutoencoderKLWan.from_single_file(
            vae_path, config=os.path.join(_BUNDLED_DIR, "vae"),
            local_files_only=True, torch_dtype=torch.float32
        ).eval()
        vae_model.requires_grad_(False)
        self.wan_vae = vae_model
        self.latent_format = estimated_config.latent_format

        # --- DiT (from the loaded checkpoint) wrapped for Forge sampling ---
        transformer = huggingface_components['transformer']
        unet = UnetPatcher.from_model(
            model=transformer, diffusers_scheduler=None,
            k_predictor=PredictionAnima(shift=3.0), config=estimated_config,
        )

        # --- engine-side copy of the LLM adapter (runs during conditioning) ---
        adapter = LLMAdapter(1024)
        asd = {k[len("net.llm_adapter."):]: v for k, v in transformer.state_dict().items()
               if k.startswith("net.llm_adapter.")}
        adapter.load_state_dict(asd, strict=True)
        self.adapter = adapter.to(memory_management.text_encoder_offload_device(), self.te_dtype).eval()
        self.adapter.requires_grad_(False)

        vae = _WanVAEWrapper(vae_model)
        self.forge_objects = ForgeObjects(unet=unet, clip=None, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    # ---- text conditioning: Qwen3 hidden + T5 ids -> LLM adapter -> 512-token context ----
    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        dev, dt = self.device, self.te_dtype
        self.qwen.to(dev)
        self.adapter.to(dev)
        outs = []
        for p in prompt:
            qids = self.qwen_tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            if qids.shape[1] == 0:
                # empty prompt (e.g. no negative prompt) tokenizes to nothing; feed a single
                # token so Qwen3's attention has non-zero length instead of crashing on a
                # 0-length reshape.
                fallback = self.qwen_tok.eos_token_id
                if fallback is None:
                    fallback = self.qwen_tok.pad_token_id or 0
                qids = torch.tensor([[fallback]], device=dev, dtype=torch.long)
            qh = self.qwen(input_ids=qids).last_hidden_state.to(dt)
            t5 = self.t5_tok(p, return_tensors="pt").input_ids.to(dev)
            c = self.adapter(t5, qh)
            if c.shape[1] < 512:
                c = torch.nn.functional.pad(c, (0, 0, 0, 512 - c.shape[1]))
            else:
                c = c[:, :512]
            outs.append(c)
        self.qwen.to(memory_management.text_encoder_offload_device())
        self.adapter.to(memory_management.text_encoder_offload_device())
        cond = torch.cat(outs, dim=0)
        return cond   # tensor -> compile_conditions uses the crossattn-only path (no pooled 'vector')

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        n = len(self.t5_tok(prompt).input_ids)
        return n, max(255, n)

    def _lat_stats(self, dev, dt):
        m = self.latent_format.latents_mean.to(dev, dt)
        s = self.latent_format.latents_std.to(dev, dt)
        return m, s

    @torch.inference_mode()
    def encode_first_stage(self, x):
        # x: [B,3,H,W] in [-1,1]
        self.wan_vae.to(self.device)
        v = x.to(self.device, torch.float32).unsqueeze(2)          # [B,3,1,H,W]
        lat = self.wan_vae.encode(v).latent_dist.mode()            # [B,16,1,H/8,W/8]
        m, s = self._lat_stats(lat.device, lat.dtype)
        z = (lat - m) / s
        return z.squeeze(2).to(x)                                  # [B,16,H/8,W/8]

    @torch.inference_mode()
    def decode_first_stage(self, x):
        self.wan_vae.to(self.device)
        z = x.to(self.device, torch.float32).unsqueeze(2)          # [B,16,1,H,W]
        m, s = self._lat_stats(z.device, z.dtype)
        lat = z * s + m
        img = self.wan_vae.decode(lat).sample                      # [B,3,1,H*8,W*8]
        return img.squeeze(2).to(x)
