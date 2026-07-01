"""Durable runtime registration of the Z-Image architecture into ``huggingface_guess``.

Same rationale as ``anima_hf_register``: ``repositories/huggingface_guess`` is a git-ignored
vendor copy, so we inject Z-Image's latent format, model config and checkpoint-detection branch
at import time from this version-controlled, idempotent module.
"""

import torch

from huggingface_guess import detection, latent, model_list
from huggingface_guess.model_list import BASE, ModelType


# Z-Image VAE (standard AutoencoderKL, 16ch) uses SD3-style shift/scale (vae/config.json).
_ZIMAGE_SCALE = 0.3611
_ZIMAGE_SHIFT = 0.1159


def _register_latent_format():
    if hasattr(latent, "ZImageLatent"):
        return latent.ZImageLatent

    class ZImageLatent(latent.LatentFormat):
        latent_channels = 16
        scale_factor = _ZIMAGE_SCALE

        def __init__(self):
            self.shift_factor = _ZIMAGE_SHIFT

        def process_in(self, lat):
            return (lat - self.shift_factor) * self.scale_factor

        def process_out(self, lat):
            return lat / self.scale_factor + self.shift_factor

    latent.ZImageLatent = ZImageLatent
    return ZImageLatent


def _register_model_config(latent_cls):
    if getattr(model_list, "ZImage", None) is not None:
        return model_list.ZImage

    class ZImage(BASE):
        huggingface_repo = "ZImage"

        unet_config = {
            "image_model": "zimage",
        }

        sampling_settings = {
            "multiplier": 1.0,
            "shift": 3.0,
        }

        unet_extra_config = {}
        latent_format = latent_cls

        memory_usage_factor = 1.6

        supported_inference_dtypes = [torch.bfloat16, torch.float32]

        vae_key_prefix = ["vae."]
        text_encoder_key_prefix = ["text_encoders."]

        unet_target = "transformer"

        def model_type(self, state_dict, prefix=""):
            return ModelType.FLOW

        def clip_target(self, state_dict={}):
            # Text encoder ships separately (loaded by the engine), never inside the DiT file.
            return {}

    model_list.ZImage = ZImage
    if ZImage not in model_list.models:
        model_list.models.append(ZImage)
    return ZImage


def _zimage_dit_config(state_dict, key_prefix):
    """Return the Z-Image dit_config if the checkpoint is a Z-Image DiT, else None."""
    sig = "{}all_x_embedder.2-1.weight".format(key_prefix)
    refiner = "{}noise_refiner.0.attention.qkv.weight".format(key_prefix)
    if sig not in state_dict:
        return None
    # A second signature key guards against false positives; fall back to just `sig` if the
    # refiner naming differs across variants.
    if refiner not in state_dict and not any(
        k.startswith("{}noise_refiner.".format(key_prefix)) for k in state_dict
    ):
        return None
    # in_channels is consumed by BASE.inpaint_model() and the loader; 16 for the Z-Image VAE.
    return {"image_model": "zimage", "in_channels": 16}


def _register_detection():
    if getattr(detection, "_zimage_detection_patched", False):
        return

    _orig = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        cfg = _zimage_dit_config(state_dict, key_prefix)
        if cfg is not None:
            return cfg
        return _orig(state_dict, key_prefix, *args, **kwargs)

    detection.detect_unet_config = detect_unet_config
    detection._zimage_detection_patched = True


def register():
    """Inject Z-Image into huggingface_guess. Idempotent; safe to call repeatedly."""
    latent_cls = _register_latent_format()
    _register_model_config(latent_cls)
    _register_detection()
