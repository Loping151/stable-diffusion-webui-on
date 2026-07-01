import os
import torch

# Inject the Z-Image architecture into the git-ignored huggingface_guess vendor copy before we
# reference model_list.ZImage below. Idempotent.
from backend.nn import zimage_hf_register as _zimage_hf_register
_zimage_hf_register.register()

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.unet import UnetPatcher
from backend.modules.k_prediction import PredictionZImage
from backend import memory_management


# Components not present in the single-file DiT checkpoint (Qwen3 text encoder, its tokenizer,
# and the AutoencoderKL) are loaded from this directory. Point ZIMAGE_ASSETS elsewhere to override.
ZIMAGE_ASSETS = os.environ.get(
    "ZIMAGE_ASSETS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                 "models", "z-image", "Z-Image-Turbo"),
)

# Fixed length we zero-pad the (variable-length, masked) Qwen3 text embeddings to, so Forge can
# carry them as a single tensor and batch cond/uncond together. The DiT wrapper trims the
# trailing zeros back to each item's true length.
ZIMAGE_MAX_TOKENS = 512


class _AutoKLWrapper:
    """Minimal VAE holder so ForgeObjects/save paths don't crash; real work is in the engine."""
    latent_channels = 16

    def __init__(self, model):
        self.first_stage_model = model

    def shallow_copy(self):
        return _AutoKLWrapper(self.first_stage_model)


class ZImage(ForgeDiffusionEngine):
    matched_guesses = [model_list.ZImage]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        from transformers import AutoTokenizer, AutoModel
        from diffusers import AutoencoderKL

        self.device = memory_management.get_torch_device()
        self.te_dtype = memory_management.text_encoder_dtype()

        # --- Qwen3 text encoder + tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(ZIMAGE_ASSETS, "tokenizer"))
        self.text_encoder = AutoModel.from_pretrained(
            os.path.join(ZIMAGE_ASSETS, "text_encoder"), dtype=self.te_dtype
        ).eval()
        self.text_encoder.requires_grad_(False)
        self.text_encoder = self.text_encoder.to(memory_management.text_encoder_offload_device())

        # --- VAE (standard AutoencoderKL, 16ch, SD3-style scale/shift) ---
        vae_model = AutoencoderKL.from_pretrained(
            os.path.join(ZIMAGE_ASSETS, "vae"), torch_dtype=torch.float32
        ).eval()
        vae_model.requires_grad_(False)
        self.vae_model = vae_model
        self.vae_scale = float(vae_model.config.scaling_factor)
        self.vae_shift = float(getattr(vae_model.config, "shift_factor", 0.0) or 0.0)

        # --- DiT (from the loaded checkpoint) wrapped for Forge sampling ---
        transformer = huggingface_components['transformer']
        unet = UnetPatcher.from_model(
            model=transformer, diffusers_scheduler=None,
            k_predictor=PredictionZImage(shift=3.0), config=estimated_config,
        )

        vae = _AutoKLWrapper(vae_model)
        self.forge_objects = ForgeObjects(unet=unet, clip=None, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    # ---- text conditioning: Qwen3 chat-template -> hidden_states[-2], zero-padded to fixed len ----
    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        dev, dt = self.device, self.te_dtype
        self.text_encoder.to(dev)
        outs = []
        for p in prompt:
            s = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            ti = self.tokenizer(
                [s], padding="max_length", max_length=ZIMAGE_MAX_TOKENS,
                truncation=True, return_tensors="pt",
            )
            ids = ti.input_ids.to(dev)
            m = ti.attention_mask.to(dev).bool()
            h = self.text_encoder(input_ids=ids, attention_mask=m, output_hidden_states=True).hidden_states[-2]
            h = h[0].to(dt)
            # zero out padding positions so the DiT wrapper can recover the true length by trimming
            h = h * m[0].unsqueeze(-1).to(dt)
            outs.append(h.unsqueeze(0))
        self.text_encoder.to(memory_management.text_encoder_offload_device())
        return torch.cat(outs, dim=0)   # [B, ZIMAGE_MAX_TOKENS, 2560]

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        n = len(self.tokenizer(prompt).input_ids)
        return n, max(255, n)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        self.vae_model.to(self.device)
        v = x.to(self.device, torch.float32)
        lat = self.vae_model.encode(v).latent_dist.sample()
        z = (lat - self.vae_shift) * self.vae_scale
        return z.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        self.vae_model.to(self.device)
        lat = x.to(self.device, torch.float32) / self.vae_scale + self.vae_shift
        img = self.vae_model.decode(lat).sample
        return img.to(x)
