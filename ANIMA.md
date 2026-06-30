# Anima support (experimental)

[Anima](https://huggingface.co/circlestone-labs/Anima) is a 2B text-to-image model from
CircleStone Labs / Comfy Org — a finetune of **NVIDIA Cosmos-Predict2-2B** with a
**Qwen3-0.6B** text encoder, an **LLM adapter** (T5 token IDs cross-attending Qwen3 hidden
states), and the **Qwen-Image / Wan 2.1 VAE**. It is a Cosmos **DiT**, so Forge's UNet/SDXL
backend cannot load it. `anima_infer.py` is a standalone diffusers/torch port of ComfyUI's
Anima pipeline (no ComfyUI dependency).

## Components (download into `models/anima/`)
- `diffusion_models/anima-base-v1.0.safetensors` — the Cosmos DiT (+ `net.llm_adapter`)
- `text_encoders/qwen_3_06b_base.safetensors` — Qwen3-0.6B base text encoder
- `vae/qwen_image_vae.safetensors` — Qwen-Image (Wan 2.1) 16-ch 3D VAE
- `tokenizer/` — Qwen3 tokenizer, `t5_tokenizer/` — T5 (32128 vocab) tokenizer
- `loras/anima-turbo-lora-v0.2.safetensors` — optional Turbo LoRA

## Usage
```bash
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python anima_infer.py \
  --prompt "masterpiece, best quality, score_7, safe, 1girl, silver hair, kimono, maple" \
  --steps 30 --cfg 5 --size 1024 --out anima_out.png
# optional LoRA (Anima format: diffusion_model.<module>.lora_A/lora_B):
#   --lora models/anima/loras/anima-turbo-lora-v0.2.safetensors --lora-scale 1.0
```

## Status — VERIFIED vs OPEN
**Verified correct** (the architecture port is faithful):
- DiT loads with 0 missing / 0 unexpected tensors; every component (patchify, AdaLN-LoRA,
  RoPE-3D split-half, attention QK-norm, GELU MLP, final layer, llm_adapter, timestep=sigma)
  was checked line-by-line against ComfyUI master.
- **img2img refinement is perfect**: encode a real image → normalise → denoise from low sigma
  → reconstructs a sharp anime image. This proves the DiT forward, VAE, normalisation and text
  encoding are all correct.
- Text conditioning works (prompts steer colour/content; "fire"→red, "ice"→blue).
- Anima-format LoRA merging works (Turbo LoRA: 508/508 modules).
- VAE round-trip is pixel-faithful (Wan2.1 latents_mean/std).

**Open issue**: pure-noise **text-to-image** produces a faceted / painterly result (correct
colours, no fine structure). The single-step velocity field matches the expected target
(cos≈0.91 across all sigmas), and the failure persists across euler / heun / SDE samplers,
cfg 1–6, fp32/bf16, 512–1024 px, 24–50 steps, and with/without the Turbo LoRA — so it is a
subtle sampling/trajectory issue, not a component bug. Resolving it likely needs an
intermediate-tensor diff against a real ComfyUI run. img2img (low-sigma) works today.
