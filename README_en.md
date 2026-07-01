<div align="center">

# Stable Diffusion WebUI On

**Keep supporting as many new AIGC model architectures as possible — and carry the LoRA / extension ecosystem along.**

`more architectures · fresh deps · LoRA/extensions kept working · personally maintained`

<sub>[中文](README.md) · **English**</sub>

[![Base](https://img.shields.io/badge/based%20on-SD%20WebUI%20Forge-8A2BE2)](https://github.com/lllyasviel/stable-diffusion-webui-forge)
[![Python](https://img.shields.io/badge/Python-3.10~3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Torch](https://img.shields.io/badge/torch-2.3~2.7%20/%20cu121~cu128-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-yellow.svg)](LICENSE.txt)

</div>

---

Forge *forged* optimizations and new models into SD WebUI. `-on` keeps the switch **on** — new model architectures, new LoRAs, new extensions, lit up one at a time.

**In plain terms:** this is a personally maintained fork of [SD WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge). When Forge slowed down I picked it up to keep fixing bugs and refreshing dependencies — the main point being to **let architectures Forge can't load (Anima, Z-Image, Qwen-Image, …) load and generate right inside the same WebUI**, while trying not to break the existing LoRA and extension ecosystem.

> Experimental, mostly for my own use. Try it and file issues, but don't expect official-grade stability guarantees.

## Lineage & credits

This project stands entirely on the work it's built on:

- **[lllyasviel/stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)** — *Forge*, the direct upstream. It rewrote memory management, sped up inference, and added many experimental features on top of SD WebUI. (The name "Forge" nods to Minecraft Forge.)
- **[AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)** — where it all started; Forge is based on its 1.10.1.

Forge's original tutorials / NEWS still apply — see the [Forge repo](https://github.com/lllyasviel/stable-diffusion-webui-forge). This fork only adds on top.

## New native model architectures

Forge natively understands only the SD1/SD2/SDXL/SD3/Flux families. The architectures below are **native** in this fork — drop the checkpoint into `models/Stable-diffusion/`, select it in the UI/API, and generate like any other model:

| Model | Type | Notes | Docs |
|---|---|---|---|
| **Anima** | Cosmos-Predict2 DiT + Qwen3 + LLM adapter | 2B; single-file DiT + assets | [ANIMA.md](ANIMA.md) |
| **Z-Image / Z-Image-Turbo** | Tongyi NextDiT + Qwen3 + AutoencoderKL | ~6B; Turbo does 1024² in 8 steps | [ZIMAGE.md](ZIMAGE.md) |
| **FLUX.1 Krea [dev]** | Flux architecture | runs on the existing Flux engine; fp8 fits a 24 GB card | [KREA.md](KREA.md) |
| **Qwen-Image** | 20B MMDiT + Qwen2.5-VL | fp8 puts the 20B on a single 24 GB card; bf16 works on a larger one | [QWENIMAGE.md](QWENIMAGE.md) |

On **quantization**: models too big to fit (e.g. the 20B Qwen-Image) go through Forge's built-in quant path — drop in an fp8 / gguf single-file and Forge swaps `Linear` layers for quantized ops at load time. **If the quantized one runs, the bf16 one runs too** on a bigger card — same path, dtype auto-detected.

## What else changed

- **Modernized deps, without dropping old GPUs.** Default stack moves to Python 3.12 + torch 2.7.1 / CUDA 12.8 + xformers; `requirements` prefer version ranges over hard pins, and for the deps that do need pinning I checked whether the used functions actually changed behavior. The old stack (RTX 20xx / CUDA 11) installs with a single command (below).
- **Fixed the whole Flux family on newer diffusers.** diffusers ≥ 0.38 changed `FlowMatchEulerDiscreteScheduler.time_shift`'s signature and Forge's old call crashed; it's inlined now, so Flux / Krea load again.
- **LoRA and extensions carried along.** Keeps Forge's LoRA loading and extension hooks working where possible; common extensions (e.g. regional control) are bundled.
- **Adding an architecture is a documented method.** How to bring in the next one is written up as a reusable skill (below), with the pitfalls that actually cost time recorded (NaN schedules, velocity sign, timestep convention, masked variable-length conditioning, fp8 norms, 3D-VAE OOM).

## Install

Same as SD WebUI / Forge — install Git and Python, clone, run:

```bash
git clone <this-repo-url> stable-diffusion-webui-on
cd stable-diffusion-webui-on
./webui.sh          # Linux / macOS
# webui-user.bat    # Windows
```

For a faster, cleaner isolated env, use the bundled `uv` one-click script (default modern stack: Python 3.12 + torch 2.7.1 / cu128):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
./install.sh
./run.sh
```

The same script installs the **old stack** via env vars (e.g. RTX 20xx / CUDA 11):

```bash
PYTHON_VERSION=3.10 TORCH_VERSION=2.3.1 TORCHVISION_VERSION=0.18.1 \
XFORMERS_VERSION=0.0.27 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./install.sh
```

Each new architecture also needs extra components (text encoder / VAE); see its doc for what to download and where (locations are overridable via `ANIMA_ASSETS` / `ZIMAGE_ASSETS` / `QWENIMAGE_ASSETS`).

## Adding another architecture

The process is written up as a reusable skill: [`.claude/skills/add-model-architecture`](.claude/skills/add-model-architecture/SKILL.md).

In one line: **first get the model running as a standalone diffusers/torch pipeline (your bit-exact reference), then wrap it as a Forge diffusion engine for native integration.** The parts that actually eat time — sampler NaNs, velocity sign, timestep = sigma vs 1-sigma, whether the conditioning needs a mask, fp8 weights, 3D-VAE memory — are all in there.

## License

Inherits Forge / SD WebUI's license — see [LICENSE.txt](LICENSE.txt) (AGPL-3.0).
