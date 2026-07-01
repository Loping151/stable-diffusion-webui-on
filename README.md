# Stable Diffusion WebUI On

> 简介：`stable-diffusion-webui-on` 是个人持续维护的 Stable Diffusion WebUI 分支，基于 Forge，沿用其代码并继续修 bug、更新依赖、做个人定制。

`stable-diffusion-webui-on` is a personally maintained continuation of
[Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge),
which is itself a platform built on top of
[AUTOMATIC1111's Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
(based on [Gradio](https://www.gradio.app/)).

This repository continues the Forge codebase and keeps it maintained going
forward — bug fixes, dependency updates, and personal customizations.

## Lineage & Credits

This project stands entirely on the work it is built on:

- **[lllyasviel/stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)**
  — *Forge*, the direct upstream of this repository. Forge adds optimized GPU
  resource management, faster inference, and experimental features on top of
  SD WebUI. (The name "Forge" was inspired by "Minecraft Forge".)
- **[AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)**
  — the *original* Stable Diffusion WebUI that Forge is based on. Forge is based
  on SD-WebUI 1.10.1 at
  [this commit](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/82a973c04367123ae98bd9abdf80d9eda9b910e2).

For Forge's original feature tutorials, news, and discussions, see the
[Forge repository](https://github.com/lllyasviel/stable-diffusion-webui-forge)
and its [NEWS section](https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/main/NEWS.md).

## Installation

Install Git and Python, then clone and run — same method as SD WebUI / Forge:

```bash
git clone <your-repo-url> stable-diffusion-webui-on
cd stable-diffusion-webui-on

# Linux / macOS
./webui.sh

# Windows
webui-user.bat
```

The launcher creates a virtual environment and installs dependencies on first
run. If you want an all-in-one one-click Windows package instead, use Forge's
[release packages](https://github.com/lllyasviel/stable-diffusion-webui-forge/releases).

## New model architectures

Beyond the SD1/SD2/SDXL/SD3/Flux families Forge already supports, this fork adds
native, in-WebUI support for several newer text-to-image architectures — select the
checkpoint and generate like any other model:

| Model | Type | Notes | Docs |
|-------|------|-------|------|
| **Anima** | Cosmos-Predict2 DiT + Qwen3 + LLM adapter | 2B; single-file DiT + assets | [ANIMA.md](ANIMA.md) |
| **Z-Image / Z-Image-Turbo** | Tongyi NextDiT + Qwen3 + AutoencoderKL | ~6B; Turbo runs in 8 steps @ 1024² | [ZIMAGE.md](ZIMAGE.md) |
| **FLUX.1 Krea [dev]** | Flux architecture | runs on the existing Flux engine (fp8 fits a 24 GB card) | [KREA.md](KREA.md) |

Adding a further architecture is documented as a reusable method in
[`.claude/skills/add-model-architecture`](.claude/skills/add-model-architecture/SKILL.md)
(standalone bring-up → bit-exact verification → native Forge integration, with the
pitfalls that actually cost time: NaN schedules, velocity sign, timestep convention,
masked variable-length conditioning).

> **Qwen-Image** (20B MMDiT + Qwen2.5-VL): the diffusers-backed integration pattern used
> above applies, but the 40 GB bf16 weights exceed a single 24 GB GPU and the fp8 path
> needs custom quantized-compute support, so it is not enabled by default on this hardware.

## License

See [LICENSE.txt](LICENSE.txt). This project inherits the licensing of
Stable Diffusion WebUI and Forge.
