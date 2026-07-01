import os
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
ANIMA_ASSETS = os.environ.get(
    "ANIMA_ASSETS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "anima"),
)


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

        from transformers import AutoTokenizer, Qwen3Config, Qwen3Model
        from diffusers import AutoencoderKLWan
        from safetensors.torch import load_file

        self.device = memory_management.get_torch_device()
        self.te_dtype = memory_management.text_encoder_dtype()

        # --- text encoders / tokenizers ---
        self.qwen_tok = AutoTokenizer.from_pretrained(os.path.join(ANIMA_ASSETS, "tokenizer"))
        self.t5_tok = AutoTokenizer.from_pretrained(os.path.join(ANIMA_ASSETS, "t5_tokenizer"))

        qsd = load_file(os.path.join(ANIMA_ASSETS, "text_encoders", "qwen_3_06b_base.safetensors"))
        qsd = {k[len("model."):]: v for k, v in qsd.items() if k.startswith("model.")}
        qcfg = Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                           num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                           head_dim=128, max_position_embeddings=40960, rms_norm_eps=1e-6,
                           rope_theta=1e6, tie_word_embeddings=True)
        self.qwen = Qwen3Model(qcfg)
        self.qwen.load_state_dict(qsd, strict=True)
        self.qwen = self.qwen.to(memory_management.text_encoder_offload_device(), self.te_dtype).eval()
        self.qwen.requires_grad_(False)

        # --- Wan / Qwen-Image VAE ---
        vae_model = AutoencoderKLWan.from_single_file(
            os.path.join(ANIMA_ASSETS, "vae", "qwen_image_vae.safetensors"), torch_dtype=torch.float32
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
