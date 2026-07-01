# Z-Image support (experimental)

[Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) is a ~6B text-to-image model from
Tongyi-MAI (Alibaba). It is a **NextDiT**-style single-stream DiT with a **Qwen3** text encoder
and a standard **AutoencoderKL** (16-channel, SD3-style shift/scale). Forge's UNet backend can't
load it, so it is integrated natively here on top of the diffusers `ZImageTransformer2DModel`.

Two variants share one architecture: **Z-Image-Turbo** (8-step distilled, CFG off) and
**Z-Image** (the base model, more steps + CFG). The integration works for both.

## Components (a diffusers-format repo, e.g. `models/z-image/Z-Image-Turbo/`)
- `transformer/` — the NextDiT (sharded). Merge it into a single-file checkpoint for Forge (below).
- `text_encoder/` — Qwen3 (hidden 2560); `tokenizer/` — its Qwen2 tokenizer.
- `vae/` — AutoencoderKL (16ch, scaling_factor 0.3611, shift_factor 0.1159).
- `scheduler/` — FlowMatchEulerDiscreteScheduler (shift 3.0).

Download the repo (public, no token):
```bash
./venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Tongyi-MAI/Z-Image-Turbo', local_dir='models/z-image/Z-Image-Turbo')"
```

Make a single-file DiT checkpoint Forge can select (bf16, ~12 GB):
```bash
./venv/bin/python - <<'PY'
import glob, torch; from safetensors import safe_open; from safetensors.torch import save_file
out={}
for sh in sorted(glob.glob("models/z-image/Z-Image-Turbo/transformer/diffusion_pytorch_model-*.safetensors")):
    with safe_open(sh, framework="pt") as f:
        for k in f.keys(): out[k]=f.get_tensor(k).to(torch.bfloat16).contiguous()
save_file(out, "models/Stable-diffusion/z-image-turbo.safetensors")
PY
```

## Native Forge usage
Select `z-image-turbo.safetensors` as the checkpoint and generate. The text encoder, tokenizer
and VAE load from `ZIMAGE_ASSETS` (default `models/z-image/Z-Image-Turbo`; override with the
env var). Recommended: Euler sampler.
- **Turbo**: 8 steps, **CFG 1** (guidance is baked in / off), 1024x1024.
- **Base**: ~30-50 steps, CFG 4-6.

## Status — WORKING
txt2img in Forge (Turbo, 8 steps, 1024x1024) matches the diffusers `ZImagePipeline` oracle for
the same prompt (per-step latent std tracks the reference; final image quality equivalent).

Pieces (all version-controlled):
- `backend/nn/zimage.py` — `IntegratedZImage` (subclass of diffusers `ZImageTransformer2DModel`);
  batched->list adaptation, zero-pad trimming, and the velocity-sign negation.
- `backend/diffusion_engine/zimage.py` — the engine (Qwen3 + tokenizer + VAE, conditioning, VAE io).
- `backend/nn/zimage_hf_register.py` — durable huggingface_guess registration.
- `backend/modules/k_prediction.py::PredictionZImage` — rectified-flow predictor, `timestep=1-sigma`.
- `backend/loader.py` + `backend/huggingface/ZImage/model_index.json` — routing + engine registration.

### Bring-up notes (see `.claude/skills/add-model-architecture`)
- The diffusers pipeline negates the DiT output (`noise_pred = -noise_pred`) before the scheduler
  step; folded into the wrapper. Without it the sampler diverges (latent std grows) -> pure noise.
- Timestep is `1 - sigma` (t=0 is noise), the opposite of Anima's Cosmos DiT (`t = sigma`).
- Conditioning is variable-length + masked; padding without a mask diverges ~30% rel-L2, so the
  wrapper trims zero-padding back to each prompt's true token length.
