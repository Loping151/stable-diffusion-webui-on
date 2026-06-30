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

## Status — WORKING
Text-to-image produces clean, high-quality anime images on CUDA0, matching ComfyUI's output
for the same prompt/seed/settings. Verified by diffing every stage against ComfyUI master:
- DiT loads 0 missing / 0 unexpected; RoPE-3D and the full DiT forward are **bit-exact** vs
  ComfyUI's compiled kernel (`apply_rope_split_half`) and `MiniTrainDIT`.
- Tokenization (Qwen3 + T5) and Qwen3 hidden states are bit-exact vs ComfyUI.
- LLM adapter is bit-exact vs ComfyUI.
- Anima-format LoRA merging works (Turbo LoRA: 508/508 modules).

The final bug fixed during bring-up: the LLM adapter applied `out_proj` and the final RMSNorm
in the wrong order. ComfyUI does `norm(out_proj(x))`; an earlier draft did `out_proj(norm(x))`,
which corrupted the cross-attention context and turned text-to-image into faceted mush while
leaving img2img (which leans on existing structure) looking fine. Order matters because RMSNorm
and a Linear don't commute.

Recommended settings (per the model card): 30–50 steps, CFG 4–6, 512²–1536². Samplers euler /
heun (`ANIMA_SAMPLER=heun`) work; the model card also likes er_sde. The Wan 3D VAE decode is
memory-heavy, so the VAE runs on CPU by default (change in `load_vae`).
