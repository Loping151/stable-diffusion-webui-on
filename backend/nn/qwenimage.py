"""Forge-native wrapper around the diffusers Qwen-Image (MMDiT) transformer.

The diffusers ``QwenImageTransformer2DModel`` implements the full 20B MMDiT (joint text+image
attention, 3D RoPE). We only adapt Forge's batched ``(x, timestep, context)`` convention to its
API, which works on *packed* latents ([B, seq, in_channels] where in_channels = 16 * patch^2)
plus an ``encoder_hidden_states_mask`` and per-image ``img_shapes``.

We subclass the diffusers model so its parameter names load with no prefix, and — crucially —
so that when Forge's loader builds it inside ``using_forge_operations(...)`` the internal
``torch.nn.Linear`` layers become Forge's quantization-aware ops. That is what lets a quantized
(fp8 / nf4 / gguf) Qwen-Image checkpoint run within a 24 GB GPU: Forge stores the weights
quantized and up-casts per-layer at compute time; the bf16 (unquantized) weights then also work
unchanged on a larger card.

Conditioning arrives zero-padded to a fixed length ([B, L, 3584]); the attention mask is
recovered from the all-zero padding rows (Qwen-Image's DiT takes an explicit mask, so no
trimming is needed). timestep is the raw sigma (0..1) and the model output is used verbatim
(the diffusers pipeline applies no sign flip before the scheduler step)."""

import torch

from diffusers import QwenImageTransformer2DModel


# Fixed architecture of the released Qwen-Image transformer (transformer/config.json). Detection
# may override any of these; unknown keys (image_model, in_channels used by inpaint_model, ...)
# are ignored so the huggingface_guess unet_config can double as the model_loader argument.
QWEN_IMAGE_CONFIG = dict(
    patch_size=2,
    in_channels=64,
    out_channels=16,
    num_layers=60,
    attention_head_dim=128,
    num_attention_heads=24,
    joint_attention_dim=3584,
    guidance_embeds=False,
    axes_dims_rope=(16, 56, 56),
)


def _pack_latents(latents):
    # [B, C, H, W] -> [B, (H/2)*(W/2), C*4]
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def _unpack_latents(latents, height, width, out_channels):
    # [B, (H/2)*(W/2), out_channels*4] -> [B, out_channels, H, W]
    b = latents.shape[0]
    hh, ww = height // 2, width // 2
    latents = latents.view(b, hh, ww, out_channels, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(b, out_channels, height, width)


class IntegratedQwenImage(QwenImageTransformer2DModel):
    def __init__(self, **config):
        cfg = dict(QWEN_IMAGE_CONFIG)
        for k, v in config.items():
            if k in cfg:
                cfg[k] = v
        super().__init__(**cfg)

    def forward(self, x, timestep, context, **kwargs):
        # x: [B, 16, H, W]; timestep: [B] (= sigma, 0..1); context: [B, L, 3584] zero-padded.
        B, C, H, W = x.shape
        packed = _pack_latents(x)                                   # [B, seq, 64]
        img_shapes = [(1, H // 2, W // 2)] * B

        ctx = context
        mask = (ctx.abs().sum(dim=-1) > 0).long()                  # [B, L] from zero-padding

        t = timestep.flatten().to(packed.dtype)

        out = super().forward(
            hidden_states=packed,
            timestep=t,
            encoder_hidden_states=ctx.to(packed.dtype),
            encoder_hidden_states_mask=mask,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]

        return _unpack_latents(out, H, W, C).to(x.dtype)
