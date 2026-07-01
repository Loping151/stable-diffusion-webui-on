"""Forge-native wrapper around the diffusers Z-Image (Tongyi NextDiT) transformer.

The diffusers ``ZImageTransformer2DModel`` already implements the full model (patchify,
variable-resolution sequence packing, 3D RoPE, refiners, main layers, unpatchify). We only
adapt Forge's batched ``(x, timestep, context)`` calling convention to its list-based
``forward(x_list, t, cap_feats)`` API. We subclass the diffusers model directly so the
checkpoint keys (``all_x_embedder.*``, ``layers.*``, ...) load with no prefix.

Conditioning arrives as a zero-padded ``[B, L, 2560]`` tensor (Forge can only carry
fixed-length cond tensors). Z-Image was trained with variable-length, masked text sequences
and — unlike Anima — is sensitive to attending over padding (padding to 512 without a mask
diverges ~30% rel-L2 from the true-length output), so we recover each item's true length by
trimming trailing all-zero rows before calling the transformer.
"""

import torch

from diffusers import ZImageTransformer2DModel


# Fixed architecture of the released Z-Image / Z-Image-Turbo transformer (transformer/config.json).
# Detection may override any of these keys from the checkpoint; unknown keys (e.g. image_model)
# are ignored so the huggingface_guess unet_config can double as the model_loader argument.
Z_IMAGE_CONFIG = dict(
    all_patch_size=[2],
    all_f_patch_size=[1],
    in_channels=16,
    dim=3840,
    n_layers=30,
    n_refiner_layers=2,
    n_heads=30,
    n_kv_heads=30,
    norm_eps=1e-5,
    qk_norm=True,
    cap_feat_dim=2560,
    rope_theta=256.0,
    t_scale=1000.0,
    axes_dims=[32, 48, 48],
    axes_lens=[1536, 512, 512],
)


class IntegratedZImage(ZImageTransformer2DModel):
    def __init__(self, **config):
        cfg = dict(Z_IMAGE_CONFIG)
        for k, v in config.items():
            if k in cfg:
                cfg[k] = v
        super().__init__(**cfg)

    def forward(self, x, timestep, context, **kwargs):
        # x: [B, 16, H, W]; timestep: [B] (= 1 - sigma); context: [B, L, 2560] zero-padded.
        dt = next(self.parameters()).dtype
        B = x.shape[0]

        x_list = [x[i].unsqueeze(1).to(dt) for i in range(B)]   # each [16, 1, H, W]

        ctx = context.to(dt)
        cap_feats = []
        for i in range(B):
            rows = ctx[i]
            nz = rows.abs().sum(dim=-1) > 0
            n = int(nz.nonzero().max().item()) + 1 if nz.any() else rows.shape[0]
            cap_feats.append(rows[:n])

        t = timestep.to(dt).flatten()

        out_list = super().forward(x_list, t, cap_feats, return_dict=False)[0]

        outs = []
        for o in out_list:
            if o.dim() == 4:          # [16, 1, H, W] -> [16, H, W]
                o = o.squeeze(1)
            outs.append(o)
        # The diffusers ZImagePipeline negates the transformer output before the scheduler step
        # (`noise_pred = -noise_pred`). Forge's CONST predictor uses the model output directly as
        # the flow velocity, so we fold that negation in here: the DiT predicts (data - noise) but
        # Forge's Euler expects d(x)/d(sigma) = (noise - data). Without this the trajectory runs
        # away from the data and decodes to pure noise.
        return -torch.stack(outs, dim=0).to(x.dtype)
