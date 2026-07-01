# Qwen-Image support (experimental)

[Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) is Alibaba's 20B **MMDiT** text-to-image
model (joint text+image attention), paired with a **Qwen2.5-VL 7B** text encoder and the
16-channel **Qwen-Image / Wan2.1 VAE**. It is integrated natively here on top of the diffusers
`QwenImageTransformer2DModel`.

At 20B the bf16 weights (~40 GB) don't fit a 24 GB GPU, so the default here is the **fp8**
checkpoint (~20 GB on disk, quantized-compute in Forge). The exact same code path loads bf16
unchanged on a larger card — Forge auto-detects the checkpoint dtype.

## Components (all from non-gated mirrors)
| File | Repo | Put in |
|------|------|--------|
| `qwen_image_fp8_e4m3fn.safetensors` (~20 GB) | `Comfy-Org/Qwen-Image_ComfyUI` (`split_files/diffusion_models/`) | `models/Stable-diffusion/` (as `qwen-image-fp8.safetensors`) |
| `text_encoder/`, `tokenizer/`, `vae/`, `scheduler/`, `transformer/config.json` | `Qwen/Qwen-Image` | `QWENIMAGE_ASSETS` (default `models/qwen-image/Qwen-Image/`) |

The Comfy fp8 single-file already uses diffusers key naming (diffusers' single-file mapping is
the identity), so no conversion is needed — drop it in and select it. `bf16` and other quant
formats (nf4, gguf) also load via the same path.

```bash
# fp8 DiT
./venv/bin/python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('Comfy-Org/Qwen-Image_ComfyUI','split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors', local_dir='models/qwen-image/_dit')"
# components (Qwen2.5-VL + VAE + tokenizer + configs)
./venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen-Image', local_dir='models/qwen-image/Qwen-Image', \
  allow_patterns=['text_encoder/*','tokenizer/*','vae/*','scheduler/*','transformer/config.json','model_index.json'])"
```

## Native Forge usage
Select `qwen-image-fp8.safetensors`. The Qwen2.5-VL text encoder, tokenizer and VAE load from
`QWENIMAGE_ASSETS`. Qwen-Image uses **real CFG** (not distilled guidance): CFG ~4, a negative
prompt, Euler sampler, ~20 steps, 1024×1024.

## Status — WORKING
txt2img verified in Forge with the fp8 20B DiT at 1024×1024 on a single 24 GB card.

Pieces (all version-controlled): `backend/nn/qwenimage.py` (wrapper: pack/unpack,
mask-from-padding), `backend/diffusion_engine/qwenimage.py` (engine + the fp8/VAE fixes),
`backend/nn/qwenimage_hf_register.py` (huggingface_guess registration),
`backend/modules/k_prediction.py::PredictionFlux` (Qwen-Image's dynamic shift == Flux's),
`backend/loader.py` + `backend/huggingface/QwenImage/model_index.json`.

### Bring-up notes (see `.claude/skills/add-model-architecture`)
- **Quantization is Forge's job, not the model's.** Forge's loader detects the checkpoint dtype
  (fp8/nf4/gguf) and builds the model inside `using_forge_operations(...)`, which monkeypatches
  `torch.nn.Linear`/`Conv` to quantization-aware ops. Because `IntegratedQwenImage` subclasses
  the diffusers model, its internal Linear layers get this treatment automatically — the 20B fp8
  DiT fits 24 GB with no model-specific quant code.
- **Non-Linear fp8 params.** This checkpoint quantizes *everything* to fp8, but RMSNorm /
  LayerNorm / Embedding do raw fp8 multiplies/lookups that PyTorch refuses ("Promotion for
  Float8 Types is not supported"). The engine casts just those params back up to the compute
  dtype after load; the Linear weights stay fp8.
- **3D VAE decode.** The Qwen-Image VAE decode of 1024² OOMs next to the 20B DiT; the engine
  enables VAE tiling + slicing and frees the DiT from VRAM before decoding.
- **Timestep = sigma, no sign flip, dynamic shift = Flux.** Qwen-Image's scheduler uses the same
  `calculate_shift` mu as Flux, so `PredictionFlux` is reused directly.
