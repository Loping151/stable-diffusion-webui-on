---
name: add-model-architecture
description: Port a new text-to-image model architecture (a new DiT/UNet base model that Forge/A1111 or diffusers cannot load natively) into this project — first as a standalone inference pipeline to verify correctness, then as a native Forge diffusion engine so the checkpoint loads and generates in the WebUI/API. Use when asked to "support / 兼容 / 整合 a new model architecture", run a model whose baseModel is unfamiliar (e.g. a Cosmos/Lumina/Qwen-Image/Z-Image/Anima-style DiT), or reproduce a ComfyUI-only model in plain diffusers/torch.
---

# Adding a new model architecture

Forge/A1111's backend only understands SD1.x / SD2 / SDXL / Flux / SD3-style UNet
checkpoints. Brand-new base models (DiT transformers with their own text encoder +
VAE) won't load. This skill is the method for bringing one in — first as a standalone
diffusers/torch pipeline (fast to iterate + easy to verify), then wired **natively** into
the Forge backend so the checkpoint loads and generates from the WebUI/API like any other
model. See `anima_infer.py` (standalone) and `backend/diffusion_engine/anima.py` +
`backend/nn/anima.py` + `backend/nn/anima_hf_register.py` (native) + `ANIMA.md` for a
complete worked example: the **Anima** model = NVIDIA Cosmos-Predict2-2B DiT + Qwen3 text
encoder + LLM adapter + Wan2.1 VAE.

Do the standalone port first: it isolates the model math from Forge's loader/patcher/sampler
plumbing, so when the native path misbehaves you already have a trusted reference to diff against.

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

## Step 7 — Wire it natively into Forge
Once the standalone pipeline is bit-exact, integrate so the checkpoint loads in the UI/API.
Five touch points (Anima example in parentheses):

1. **Arch detection + config** — teach `huggingface_guess` to recognise the checkpoint and
   emit its config. `repositories/huggingface_guess` is a **git-ignored vendor copy**, so do
   NOT rely on editing it in place (a fresh clone loses the edit). Instead write a
   version-controlled, **idempotent** runtime-registration module that injects:
   - a `detection.detect_unet_config` wrapper that returns the DiT config when it sees a
     signature key (`net.llm_adapter.blocks.0.self_attn.q_proj.weight`), else defers;
   - a `latent.<Format>` class (the mean/std latent format);
   - a `model_list.<Arch>(BASE)` config (`unet_config`, `sampling_settings` shift/multiplier,
     `latent_format`, `supported_inference_dtypes`, `model_type()` → `ModelType.FLOW`,
     `clip_target()`), appended to `model_list.models`.
   (`backend/nn/anima_hf_register.py`; import + call `register()` from the engine module
   *before* it references `model_list.<Arch>`.) Make every step guard on existence so it is a
   no-op when the vendor copy is already patched.
2. **DiT loader routing** — in `backend/loader.py` add the diffusers `cls_name`
   (`AnimaTransformer2DModel`) to the transformer branch and map it to your
   `Integrated<Arch>` `nn.Module`. Add a `backend/huggingface/<Arch>/model_index.json` stub
   naming that `cls_name` so the guess resolves to it.
3. **Diffusion engine** — `backend/diffusion_engine/<arch>.py`, a `ForgeDiffusionEngine`
   subclass with `matched_guesses = [model_list.<Arch>]`. It loads the extra components the
   single-file DiT doesn't contain (text encoder, tokenizers, adapter, VAE) from an assets
   dir; implements `get_learned_conditioning` (return the crossattn **tensor** if there is no
   pooled vector — `compile_conditions` then takes the crossattn-only path, no `'vector'`
   KeyError), `encode_first_stage` / `decode_first_stage` (VAE + latent_format), and sets the
   predictor on the UnetPatcher.
4. **Predictor** — a `Prediction*` in `backend/modules/k_prediction.py` for the sampler.
5. **Register the engine** — import it in `backend/loader.py` and add to `possible_models`.

## Step 8 — Debug native generation with stage prints, then remove them
The standalone port is your oracle. If the native image is black/NaN/mush, gate temporary
prints behind an env var and compare the native `(x_in std, timestep, context std, latent
std, decoded min/max)` against the standalone at the same step. Black image = NaN somewhere;
uniform-nonzero = a scaling/normalization mismatch. **Remove the prints once it works.**

### The NaN-black-image trap (rectified flow + k-diffusion)
`ForgeScheduleLinker.get_sigmas(n)` builds the sampling schedule as
`append_zero(predictor.sigma(linspace(len(sigmas)-1, 0, n)))` — i.e. it calls your
predictor's `sigma(t)` on buffer **indices** `t ∈ [0, len-1]`, then appends a terminal 0.
The k-diffusion Euler loop runs `for i in range(len(sigmas)-1)`, so it evaluates the model at
every entry **except** the appended terminal. If your `sigma(index)` returns exactly 0 at
`index == 0` (e.g. `time_snr_shift(shift, t/1000)` with `t=0`), the last *sampled* sigma is 0,
and `to_d = (x - denoised)/sigma = 0/0 = NaN` → black image. Fix: make `sigma(index)` map
index 0 to `sigma_min` (>0), e.g. `time_snr_shift(shift, (t+1)/timesteps)`, so only the
never-evaluated terminal is 0. Symptoms to recognise: DiT forward is clean at every step
(`nan=False`) but the final latent going into decode is `nan=True`; and the printed timestep
of the last step is exactly `0.0`. (Do NOT "fix" this by making `timestep()`/`sigma()` mutual
identities on the raw sigma — that makes the scheduler read indices as sigmas, giving
`sigma_max≈999` and blowing up `x_in`.)

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
- Editing `repositories/huggingface_guess` (or anything under `repositories/`) directly:
  it's git-ignored, so the change vanishes on a fresh clone. Register the arch at runtime
  from a tracked module instead (Step 7.1).
- `fp16` on a DiT with a large residual stream (Cosmos-Predict2) overflows to NaN — declare
  `supported_inference_dtypes = [torch.bfloat16, torch.float32]` so Forge never picks fp16.
- Velocity SIGN. When wrapping a diffusers transformer, read the pipeline's denoising loop, not
  just the transformer: Z-Image's pipeline does `noise_pred = -noise_pred` right before
  `scheduler.step`. Forge's CONST predictor uses the model output directly as the flow
  derivative, so any such negation must be folded into the wrapper. Symptom: no NaN, but the
  latent std *grows* every step (diverges) and decodes to pure colour noise; the fix flips it
  so std tracks the diffusers per-step trajectory. Diagnose by printing per-step latent std for
  both your engine and the reference pipeline (`callback_on_step_end`) — they must match.
- Timestep convention can be inverted between arch families even at the same `shift`: Anima's
  Cosmos DiT wants `t = sigma` (t=1 is noise); Z-Image's NextDiT wants `t = 1 - sigma` (t=0 is
  noise, matching diffusers' `(1000 - sigma*1000)/1000`). Encode this in the predictor's
  `timestep(sigma)`, and check it against the reference by printing the timestep the reference
  feeds its transformer at the first and last step.
- List/variable-length conditioning vs Forge's fixed-size cond tensor: models like Z-Image
  (NextDiT) / Qwen-Image take a *list* of variable-length, masked text embeddings. Forge can
  only carry a single padded tensor per cond, so zero-pad to a fixed length in
  get_learned_conditioning and recover each item's true length in the DiT wrapper by trimming
  trailing all-zero rows. Do NOT just pad and attend over the padding without a mask unless the
  model was trained that way (Anima was; Z-Image was not — padding-without-mask diverges ~30%).
