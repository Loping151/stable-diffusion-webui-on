---
name: write-extension
description: Write or adapt a Forge / A1111 (Stable-Diffusion-WebUI) extension — a script under extensions/ that adds UI and hooks into the generation pipeline (callbacks, attention/UNet monkeypatching). Use when asked to build a new extension, modify one like the regional prompter, or make an attention-based extension work across model architectures (UNet vs DiT).
---

# Writing a Forge / A1111 extension

An extension is a folder under `extensions/` (or `extensions-builtin/`) whose
`scripts/*.py` are auto-loaded. It plugs into the WebUI via `modules` APIs and, for
the powerful ones, by monkeypatching the model's attention. Worked reference in this
repo: `extensions/sd-webui-regional-prompter` (region-masked prompting).

## Anatomy
```
my-extension/
  scripts/
    my_ext.py          # the entry: defines a scripts.Script subclass + registers callbacks
    attention.py       # (optional) the attention hook, if you alter cross-attention
  install.py           # (optional) pip-installs extra deps on first launch
  javascript/*.js      # (optional) front-end
```

### The Script class
```python
from modules import scripts
class MyScript(scripts.Script):
    def title(self): return "My Extension"
    def show(self, is_img2img): return scripts.AlwaysVisible   # or True
    def ui(self, is_img2img):
        import gradio as gr
        enabled = gr.Checkbox(label="Enable", value=False)
        return [enabled]
    def process(self, p, enabled):       # before sampling: read p.prompt, p.width, install hooks
        ...
    def postprocess(self, p, processed, enabled):   # after: clean up / restore patches
        ...
```
`p` is the `StableDiffusionProcessing` (`modules.processing`) — prompts, size, seeds,
sampler, the loaded model (`p.sd_model`). Use `process` to set up and `postprocess` to
tear down (always restore monkeypatches you applied).

### Hooking the pipeline (light touch)
`from modules import script_callbacks` then register:
`on_cfg_denoiser`, `on_cfg_denoised`, `on_model_loaded`, `on_before_image_saved`, etc.
Prefer callbacks over monkeypatching when they're enough.

### Hooking attention (heavy, what regional-prompter does)
Region/coupling extensions replace the UNet cross-attention `forward`. The pattern
(`attention.py:hook_forward`): keep a reference to the original `CrossAttention.forward`,
install a replacement that computes `to_q/to_k/to_v`, splits the prompt conditioning per
region, runs attention per region with spatial masks, and recombines. Restore the
original in `postprocess`. This depends on the SD/SDXL UNet exposing `to_q/to_k/to_v`
and `CrossAttention` modules.

## Making an attention extension architecture-aware
The existing region/attention extensions assume a **UNet** with `to_q/to_k/to_v`. A DiT
model (Cosmos/Anima/Lumina/Qwen-Image — see `add-model-architecture`) has none of that:
its attention is `blocks.N.cross_attn.{q,k,v}_proj`, it isn't loaded by Forge at all, and
the technique (region-masked cross-attention) must be re-targeted, not reused. To support
multiple architectures:
1. **Detect the architecture** at `process` time (inspect `p.sd_model` / the diffusion
   model class, or whether it's a Forge-loadable UNet vs an external DiT pipeline).
2. **Abstract the hook target**: find the cross-attention modules generically (walk the
   model for modules whose name matches a per-arch pattern: `to_k` for UNet,
   `cross_attn.k_proj` for the Cosmos DiT) instead of hard-coding `to_k`.
3. **Re-implement the masking** against that arch's attention call signature (UNet passes
   `context`; the DiT cross-attends to a fixed-length adapter context — region masks must
   map image tokens, not text tokens).
4. Fall back gracefully (log "unsupported architecture, skipping") when the model isn't
   one you handle, rather than crashing.

A clean design is a small registry: `{arch_name: {attn_module_pattern, patch_fn}}`, chosen
by the detected architecture. New architectures = one new registry entry.

## Practicalities
- `install.py` runs on launch; pin deps with ranges (`>=`) and prefer wheels (see the
  project dependency policy in `requirements_versions.txt`).
- Hot-reloadable data (JSON configs) is re-read per request; Python code changes need a
  restart.
- Always undo monkeypatches in `postprocess` (and on exceptions) so a disabled extension
  leaves the pipeline untouched.
- Test with the extension both enabled and disabled; a passive extension must be a no-op.
