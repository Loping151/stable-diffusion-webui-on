<div align="center">

# Stable Diffusion WebUI On

A fork of Stable Diffusion WebUI Forge: adapting newer AIGC model architectures within its framework, refreshing the runtime dependencies, and keeping the existing LoRA and extension ecosystem working where possible.

<sub>[中文](README.md) · **English**</sub>

[![Base](https://img.shields.io/badge/based%20on-SD%20WebUI%20Forge-8A2BE2)](https://github.com/lllyasviel/stable-diffusion-webui-forge)
[![Python](https://img.shields.io/badge/Python-3.10~3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-yellow.svg)](LICENSE.txt)

</div>

Forge is already very capable. I started this fork because I needed to run a few architectures it did not yet support (Anima, Z-Image, Qwen-Image), so I added native loading for them locally, and refreshed the tech stack along the way. It is experimental, with no stability guarantees.

The `-on` suffix stands for keeping it "on" — following newer models and their ecosystem over time so they stay usable within the same WebUI.

## Relation to Forge

This fork is incremental on top of Forge; it does not change Forge's scope or duplicate what it already does. Upstream and original authors:

- [lllyasviel/stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge): the direct upstream, which rewrote memory management, sped up inference, and introduced many experimental features.
- [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui): the original Stable Diffusion WebUI; Forge is based on its 1.10.1.

Forge's own feature docs, tutorials and [NEWS](https://github.com/lllyasviel/stable-diffusion-webui-forge) still apply here.

## Added model architectures

The architectures below are used like any regular model: place the checkpoint in `models/Stable-diffusion/` and select it in the UI or API.

| Model | Architecture | Notes | Docs |
|---|---|---|---|
| Anima | Cosmos-Predict2 DiT + Qwen3 + LLM adapter | 2B; single-file DiT with external text encoder + VAE | [ANIMA.md](ANIMA.md) |
| Z-Image / Z-Image-Turbo | Tongyi NextDiT + Qwen3 + AutoencoderKL | ~6B; Turbo generates 1024² in 8 steps | [ZIMAGE.md](ZIMAGE.md) |
| FLUX.1 Krea [dev] | Flux architecture | reuses Forge's existing Flux engine; fp8 runs on a single 24 GB GPU | [KREA.md](KREA.md) |
| Qwen-Image | 20B MMDiT + Qwen2.5-VL | fp8 puts the 20B on a single 24 GB GPU; bf16 when memory allows | [QWENIMAGE.md](QWENIMAGE.md) |

For models that exceed available VRAM (e.g. the 20B Qwen-Image), the integration uses Forge's built-in quantized-loading path: provide an fp8 or gguf single-file, and at load time Forge replaces `Linear` layers with the corresponding quantized ops. So if the quantized build runs, the bf16 build runs too on a larger GPU — the same path, distinguished only by weight dtype.

## Dependencies and other changes

- The default stack is updated to Python 3.12 + torch 2.7.1 / CUDA 12.8 + xformers. `requirements` prefer version ranges over exact pins; where pinning is unavoidable, the behavior of the functions actually used was checked for changes. Older environments (RTX 20xx / CUDA 11) remain supported and install with a single command (below).
- Fixed Flux-family loading on newer diffusers (≥ 0.38): the signature of `FlowMatchEulerDiscreteScheduler.time_shift` changed and broke Forge's original call; it is now inlined, restoring Flux and Krea.
- Forge's LoRA loading and extension hooks are kept working where feasible, and common extensions (e.g. regional control) are bundled.

## Install

Same as SD WebUI / Forge: install Git and Python, clone, and run.

```bash
git clone <this-repo-url> stable-diffusion-webui-on
cd stable-diffusion-webui-on
./webui.sh          # Linux / macOS
# webui-user.bat    # Windows
```

For a cleaner isolated environment, use the bundled `uv` one-click script (modern stack by default):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
./install.sh
./run.sh
```

The same script installs the older stack via environment variables, e.g. RTX 20xx / CUDA 11:

```bash
PYTHON_VERSION=3.10 TORCH_VERSION=2.3.1 TORCHVISION_VERSION=0.18.1 \
XFORMERS_VERSION=0.0.27 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./install.sh
```

Each new architecture also needs its own components (text encoder / VAE); see the respective document for what to download and where. Component locations can be set via `ANIMA_ASSETS` / `ZIMAGE_ASSETS` / `QWENIMAGE_ASSETS`.

## Adding a new architecture

The integration process is documented as a skill: [`skills/add-model-architecture`](skills/add-model-architecture/SKILL.md).

The general approach: first reproduce the model as a standalone diffusers/torch pipeline to serve as a bit-exact reference, then wrap it as a Forge diffusion engine for native integration. The error-prone parts — NaNs in the sampling schedule, the sign of the velocity term, whether timestep is sigma or 1−sigma, whether variable-length text conditioning needs a mask, fp8 weight handling, and 3D-VAE memory — are all recorded there.

## License

Inherits Forge / SD WebUI's license; see [LICENSE.txt](LICENSE.txt) (AGPL-3.0).
