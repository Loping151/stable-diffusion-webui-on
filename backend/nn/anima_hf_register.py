"""Durable runtime registration of the Anima architecture into ``huggingface_guess``.

``repositories/huggingface_guess`` is a vendored, git-ignored dependency, so any edits made
directly inside it are NOT tracked by this repo and are lost on a fresh clone / re-pull. This
module reconstructs the three additions Anima needs -- the latent format, the model config, and
the checkpoint detection branch -- and injects them at import time. It is fully idempotent: if
the vendored copy already contains the additions (as on a development machine that was patched
by hand), each step is skipped, so importing this never produces duplicates.

Import it (and call :func:`register`) before anything references ``model_list.Anima`` -- the
Anima diffusion engine does exactly that at the top of its module.
"""

import torch

from huggingface_guess import detection, latent, model_list
from huggingface_guess.model_list import BASE, ModelType


# Wan2.1 / Qwen-Image VAE latent statistics (16-channel 3D latents).
_WAN21_LATENTS_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
_WAN21_LATENTS_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]


def _register_latent_format():
    if hasattr(latent, "Wan21"):
        return latent.Wan21

    class Wan21(latent.LatentFormat):
        latent_channels = 16

        def __init__(self):
            self.scale_factor = 1.0
            self.latents_mean = torch.tensor(_WAN21_LATENTS_MEAN).view(1, 16, 1, 1, 1)
            self.latents_std = torch.tensor(_WAN21_LATENTS_STD).view(1, 16, 1, 1, 1)

        def process_in(self, lat):
            m = self.latents_mean.to(lat.device, lat.dtype)
            s = self.latents_std.to(lat.device, lat.dtype)
            return (lat - m) * self.scale_factor / s

        def process_out(self, lat):
            m = self.latents_mean.to(lat.device, lat.dtype)
            s = self.latents_std.to(lat.device, lat.dtype)
            return lat * s / self.scale_factor + m

    latent.Wan21 = Wan21
    return Wan21


def _register_model_config(wan21_cls):
    if getattr(model_list, "Anima", None) is not None:
        return model_list.Anima

    class Anima(BASE):
        huggingface_repo = "Anima"

        unet_config = {
            "image_model": "anima",
        }

        sampling_settings = {
            "multiplier": 1.0,
            "shift": 3.0,
        }

        unet_extra_config = {}
        latent_format = wan21_cls

        memory_usage_factor = 1.5

        # Cosmos-Predict2 residual stream has large values; fp16 overflows to NaN, so bf16/fp32 only.
        supported_inference_dtypes = [torch.bfloat16, torch.float32]

        vae_key_prefix = ["vae."]
        text_encoder_key_prefix = ["text_encoders."]

        unet_target = "transformer"

        def model_type(self, state_dict, prefix=""):
            return ModelType.FLOW

        def clip_target(self, state_dict={}):
            result = {}
            pref = self.text_encoder_key_prefix[0]
            if "{}qwen3.transformer.model.embed_tokens.weight".format(pref) in state_dict:
                result["qwen3"] = "text_encoder"
            return result

    model_list.Anima = Anima
    if Anima not in model_list.models:
        model_list.models.append(Anima)
    return Anima


def _anima_dit_config(state_dict, key_prefix):
    """Return the Anima dit_config if the checkpoint is an Anima DiT, else None."""
    q_key = "{}net.llm_adapter.blocks.0.self_attn.q_proj.weight".format(key_prefix)
    if q_key not in state_dict:
        return None
    xw = state_dict["{}net.x_embedder.proj.1.weight".format(key_prefix)]
    head_dim = state_dict["{}net.blocks.0.self_attn.q_norm.weight".format(key_prefix)].shape[0]
    return {
        "image_model": "anima",
        "model_channels": xw.shape[0],
        "in_channels": 16,
        "patch_spatial": 2,
        "num_blocks": detection.count_blocks(state_dict, "{}net.blocks.".format(key_prefix) + "{}."),
        "crossattn_emb_channels": state_dict["{}net.blocks.0.cross_attn.k_proj.weight".format(key_prefix)].shape[1],
        "num_heads": xw.shape[0] // head_dim,
        "out_channels": state_dict["{}net.final_layer.linear.weight".format(key_prefix)].shape[0] // 4,
    }


def _register_detection():
    if getattr(detection, "_anima_detection_patched", False):
        return

    _orig_detect_unet_config = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        cfg = _anima_dit_config(state_dict, key_prefix)
        if cfg is not None:
            return cfg
        return _orig_detect_unet_config(state_dict, key_prefix, *args, **kwargs)

    detection.detect_unet_config = detect_unet_config
    detection._anima_detection_patched = True


def register():
    """Inject Anima into huggingface_guess. Idempotent; safe to call repeatedly."""
    wan21_cls = _register_latent_format()
    _register_model_config(wan21_cls)
    _register_detection()
