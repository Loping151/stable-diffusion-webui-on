# FLUX.1 Krea [dev] support

[FLUX.1 Krea [dev]](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) is Black Forest
Labs' opinionated, aesthetics-tuned FLUX.1 model. It uses the **same architecture as FLUX.1
[dev]**, so Forge's existing Flux engine runs it with **no new code** — you only need the
checkpoint plus the shared Flux text encoders and VAE.

## Files (all from non-gated mirrors)
| File | Repo | Put in |
|------|------|--------|
| `flux1-krea-dev_fp8_scaled.safetensors` (~12 GB, fits 24 GB) | `Comfy-Org/FLUX.1-Krea-dev_ComfyUI` | `models/Stable-diffusion/` |
| `t5xxl_fp8_e4m3fn.safetensors` | `comfyanonymous/flux_text_encoders` | `models/text_encoder/` |
| `clip_l.safetensors` | `comfyanonymous/flux_text_encoders` | `models/text_encoder/` |
| `ae.safetensors` (Flux VAE, save as `flux_ae.safetensors`) | `Comfy-Org/Lumina_Image_2.0_Repackaged` (`split_files/vae/`) | `models/VAE/` |

The official `black-forest-labs/FLUX.1-Krea-dev` repo is gated (needs a HF token + license
acceptance); the Comfy mirror of the fp8 DiT is not. The bf16 full weights (~24 GB) also work
if you have access.

## Usage
Select `flux1-krea-dev_fp8_scaled.safetensors` as the checkpoint and pick the three modules
above in the WebUI's **VAE / Text Encoder** selector (or pass `forge_additional_modules` via the
API). Flux-style settings: **CFG 1**, **distilled CFG ~4.5**, Euler + Simple scheduler, ~20 steps.

API example:
```json
{
  "prompt": "...", "steps": 20, "width": 1024, "height": 1024,
  "cfg_scale": 1, "distilled_cfg_scale": 4.5, "sampler_name": "Euler", "scheduler": "Simple",
  "override_settings": {
    "sd_model_checkpoint": "flux1-krea-dev_fp8_scaled.safetensors",
    "forge_additional_modules": ["t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors", "flux_ae.safetensors"]
  }
}
```

## Any Civitai Flux checkpoint (quantized)
Krea is just one Flux model; the same setup runs **any** Flux checkpoint from Civitai. Most
Civitai Flux files are **unet-only** (the DiT alone, often fp8 / gguf / nf4 — despite the
`size=full` label, which means "full-precision fp8", not "all-in-one"). For those you supply
the shared **T5 + CLIP + VAE once** (the three files above) and reuse them across every Flux
checkpoint — pick them in the WebUI's *VAE / Text Encoder* selector, or pass
`forge_additional_modules` via the API. A true all-in-one Flux single file (with the text
encoders and VAE bundled) loads on its own, no modules needed.

Quantized DiTs are first-class here (Forge is a forge-native engine for Flux): **fp8 /
fp8_scaled / nf4 / fp4 / gguf** are auto-detected from the checkpoint and loaded through
Forge's quantized-compute path, so a 12B Flux fits a 24 GB card. Verified with a real Civitai
fp8 unet-only checkpoint + the shared fp8 T5 / clip_l / VAE.

## Status — WORKING
Verified in Forge at 1024×1024 (fp8 DiT + fp8 T5 + clip_l + Flux VAE on a 24 GB card),
including a real Civitai fp8 Flux unet-only checkpoint reusing the shared encoders/VAE.

### Compatibility fix
Loading any Flux-family model on this fork's modernized dependencies (diffusers ≥ 0.38) first
required fixing `PredictionFlux.apply_mu_transform` in `backend/modules/k_prediction.py`: it
called `FlowMatchEulerDiscreteScheduler.time_shift(None, ...)`, which newer diffusers rejects
(`time_shift` now reads `self.config`). The exponential time-shift is now inlined, so all Flux
models — Krea included — load correctly.
