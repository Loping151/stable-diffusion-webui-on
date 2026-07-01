"""Durable runtime registration of the Qwen-Image architecture into ``huggingface_guess``.

Same rationale as ``anima_hf_register`` / ``zimage_hf_register``: inject Qwen-Image's latent
format, model config and checkpoint detection at import time from a version-controlled,
idempotent module (``repositories/huggingface_guess`` is a git-ignored vendor copy).

The Qwen-Image VAE is the 16-channel Wan2.1 / Qwen-Image VAE, so we reuse the same latent
format Anima registered (``latent.Wan21``).
"""

import torch

from huggingface_guess import detection, latent, model_list
from huggingface_guess.model_list import BASE, ModelType


def _ensure_wan21_latent():
    if hasattr(latent, "Wan21"):
        return latent.Wan21
    # Reuse Anima's registration (defines latent.Wan21) rather than duplicating the constants.
    from backend.nn import anima_hf_register
    return anima_hf_register._register_latent_format()


def _register_model_config(latent_cls):
    if getattr(model_list, "QwenImage", None) is not None:
        return model_list.QwenImage

    class QwenImage(BASE):
        huggingface_repo = "QwenImage"

        unet_config = {
            "image_model": "qwenimage",
        }

        # FlowMatch with dynamic (resolution-dependent) shift; the predictor computes the sigma
        # schedule from image sequence length (see PredictionQwenImage), so this is informational.
        sampling_settings = {
            "multiplier": 1.0,
            "shift": 3.0,
        }

        unet_extra_config = {}
        latent_format = latent_cls

        memory_usage_factor = 2.0

        supported_inference_dtypes = [torch.bfloat16, torch.float32]

        vae_key_prefix = ["vae."]
        text_encoder_key_prefix = ["text_encoders."]

        unet_target = "transformer"

        def model_type(self, state_dict, prefix=""):
            return ModelType.FLOW

        def clip_target(self, state_dict={}):
            # Qwen2.5-VL text encoder ships separately (loaded by the engine).
            return {}

    model_list.QwenImage = QwenImage
    if QwenImage not in model_list.models:
        model_list.models.append(QwenImage)
    return QwenImage


def _qwenimage_dit_config(state_dict, key_prefix):
    """Return the Qwen-Image dit_config if the checkpoint is a Qwen-Image MMDiT, else None."""
    sig_a = "{}transformer_blocks.0.img_mod.1.weight".format(key_prefix)
    sig_b = "{}txt_norm.weight".format(key_prefix)
    if sig_a not in state_dict or sig_b not in state_dict:
        return None
    # in_channels (packed) is consumed by BASE.inpaint_model()/loader; 64 = 16 * patch(2)^2.
    return {"image_model": "qwenimage", "in_channels": 64}


def _register_detection():
    if getattr(detection, "_qwenimage_detection_patched", False):
        return

    _orig = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        cfg = _qwenimage_dit_config(state_dict, key_prefix)
        if cfg is not None:
            return cfg
        return _orig(state_dict, key_prefix, *args, **kwargs)

    detection.detect_unet_config = detect_unet_config
    detection._qwenimage_detection_patched = True


def register():
    """Inject Qwen-Image into huggingface_guess. Idempotent; safe to call repeatedly."""
    latent_cls = _ensure_wan21_latent()
    _register_model_config(latent_cls)
    _register_detection()
