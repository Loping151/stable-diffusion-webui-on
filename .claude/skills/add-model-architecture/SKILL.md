---
name: add-model-architecture
description: Port a new text-to-image model architecture (a new DiT/UNet base model that Forge/A1111 or diffusers cannot load natively) into this project as a standalone inference pipeline. Use when asked to "support / 兼容 a new model architecture", run a model whose baseModel is unfamiliar (e.g. a Cosmos/Lumina/Qwen-Image/Anima-style DiT), or reproduce a ComfyUI-only model in plain diffusers/torch.
---

# Adding a new model architecture

Forge/A1111's backend only understands SD1.x / SD2 / SDXL / Flux / SD3-style UNet
checkpoints. Brand-new base models (DiT transformers with their own text encoder +
VAE) won't load. This skill is the method for bringing one in as a standalone
diffusers/torch pipeline (see `anima_infer.py` + `ANIMA.md` for a complete worked
example: the **Anima** model = NVIDIA Cosmos-Predict2-2B DiT + Qwen3 text encoder +
LLM adapter + Wan2.1 VAE).

## The one rule that matters
**Verify every stage bit-exact against a reference implementation before trusting it.**
Most of the work is not writing the forward pass — it's finding the *one* subtle place
where your port silently diverges. Bugs hide in the stages you assume are correct.
(In the Anima port the only bug was the LLM adapter doing `out_proj(norm(x))` instead
of `norm(out_proj(x))` — everything else was already bit-exact. It cost a day because
it was never directly diffed.)

## Step 1 — Identify the architecture
- HuggingFace model card: the `base_model:` / `base_model:finetune:` tags name the real
  architecture (e.g. `nvidia/Cosmos-Predict2-2B-Text2Image`). The `library_name`
  (`diffusion-single-file`) and a `comfyui` tag mean "ComfyUI-native, not diffusers".
- Civitai: the `baseModel` field (e.g. `Anima`, `Illustrious`, `NoobAI`, `Pony`) is the
  architecture family. Use the Civitai API to read it (see the `lora` workflow / civitai
  token convention; civitai needs the proxy).
- Note the **components**: a DiT usually ships a diffusion model + a *separate* text
  encoder + a *separate* VAE. Read the model card "installing" section — it lists which
  file goes in `diffusion_models/`, `text_encoders/`, `vae/`.

## Step 2 — Inspect the checkpoint structure
Read the safetensors header (8-byte length prefix + JSON) without loading tensors:
list keys, shapes, dtype; find the top-level prefix (`net.`), count transformer blocks,
infer hidden size / heads / head_dim (from `q_norm` length), patch size (from the
input/output projection: `in_ch * patch^2 * t_patch`), latent channels, context dim
(from cross-attn `k_proj` in-features). This tells you the exact config.

## Step 3 — Reuse what already exists
- **VAE**: try `diffusers.AutoencoderKL*.from_single_file(path, ...)`. Qwen-Image VAE =
  Wan 2.1 VAE → `AutoencoderKLWan.from_single_file` loads it (auto key conversion).
- **Text encoder**: if it's a standard LLM (Qwen3, T5, Llama), load it with `transformers`
  by stripping the key prefix (`model.`) into the matching `*Model` + config. Match
  `rope_theta`, `rms_norm_eps`, head/kv counts exactly.
- **Tokenizer**: download the matching tokenizer; verify token IDs equal the reference's.
- **DiT transformer**: usually NOT in diffusers as-is (custom). Port it (Step 4).

## Step 4 — Port the custom transformer
- Get the reference implementation. For ComfyUI-native models it's
  `comfy/ldm/<family>/...` + `comfy/text_encoders/<family>.py` + `comfy/supported_models.py`
  (config) + `comfy/model_base.py` (forward wiring) + `comfy/latent_formats.py` (latent
  mean/std). Read them; don't trust a summary for the forward math.
- Build `nn.Module`s whose submodule names **exactly** match the checkpoint keys, then
  `load_state_dict(strict=True)` → **must be 0 missing / 0 unexpected**. That validates
  the structure.
- Replicate the forward precisely: patchify order, AdaLN modulation (and any low-rank
  `adaln_lora` added before chunking), block order (self-attn → cross-attn → MLP),
  QK-norm placement (before RoPE), RoPE axes/theta/NTK, final layer + unpatchify.

## Step 5 — Sampler, schedule, normalization, timestep
- **Latent normalization** (`latent_format`): the DiT works in `(latent - mean)/std`
  space; decode applies `z*std + mean`. Use the reference's exact mean/std.
- **Timestep convention**: flow models often feed `timestep = sigma ∈ (0,1]` (multiplier
  1.0), NOT `sigma*1000`. Check `model_sampling.timestep()` + `process_timestep()`.
- **Sampler**: rectified flow (CONST): `denoised = x - v*sigma`, Euler step
  `x += (σ_next−σ)·v`. Respect the model's `shift`.

## Step 6 — Verify bit-exact, stage by stage (do NOT skip)
Clone the reference runtime into a temp dir (e.g. ComfyUI), make a throwaway venv,
symlink the model files into its model dirs, and diff **each** stage on identical inputs:
1. tokenizer IDs (both encoders)
2. text-encoder hidden states
3. **any adapter** between encoder and DiT  ← easy to get wrong, easy to forget to test
4. RoPE tables / the rope-apply kernel
5. the full DiT forward on a fixed `(x, sigma, context)`
6. the final latent + the decoded image

cos should be ~1.0 and max-abs-diff ~0 at every stage. The first stage that isn't is the
bug. Then **delete the temp runtime** (don't leave multi-GB junk).

## Pitfalls seen in practice
- `out_proj(norm(x))` vs `norm(out_proj(x))` — order matters (RMSNorm ⊄ Linear).
- RoPE "split-half" (`[:d/2],[d/2:]`) vs interleaved; HF `rotate_half` with duplicated
  cos/sin is equivalent to split-half — verify against the actual kernel.
- Context zero-padded to a fixed length (e.g. 512) with no attention mask: this is often
  how the model was trained — keep it; removing it makes things worse.
- 3D/causal VAE decode is memory-heavy; run it on CPU or tile it to avoid OOM next to a
  multi-B DiT on a 24 GB card.
- A symptom where **img2img refines fine but text-to-image is mush** points at the
  conditioning/context path (the DiT and VAE are probably fine) — diff the text→context
  stages.
