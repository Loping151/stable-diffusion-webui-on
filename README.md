<div align="center">

# Stable Diffusion WebUI On

基于 Stable Diffusion WebUI Forge 的分支：在其框架内适配较新的 AIGC 模型架构，同步更新运行依赖，并尽量保持既有的 LoRA 与插件兼容。

<sub>**中文** · [English](README_en.md)</sub>

[![Base](https://img.shields.io/badge/based%20on-SD%20WebUI%20Forge-8A2BE2)](https://github.com/lllyasviel/stable-diffusion-webui-forge)
[![Python](https://img.shields.io/badge/Python-3.10~3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-yellow.svg)](LICENSE.txt)

</div>

Forge 已经相当完善。建立这个分支，是因为我需要运行几个 Forge 尚未支持的新架构模型（Anima、Z-Image、Qwen-Image 等），于是在本地为它们补上了原生加载支持，并在此过程中一并更新了技术栈。它偏实验性质，不提供稳定性保证。

后缀 `-on` 取“持续开着”之意，尽量跟上后续的新模型与生态，让它们在同一套 WebUI 里可用。

## 与 Forge 的关系

本分支在 Forge 之上做增量，不改变其定位，也不重复其已有能力。上游与原作者：

- [lllyasviel/stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)：直接上游，重写了显存管理、优化了推理，并引入大量实验特性。
- [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)：最初的 Stable Diffusion WebUI，Forge 基于其 1.10.1。

Forge 原有的功能说明、教程与 [NEWS](https://github.com/lllyasviel/stable-diffusion-webui-forge) 在此依然适用。

## 新增的模型架构

以下架构用法与常规模型一致：将 checkpoint 放入 `models/Stable-diffusion/`，在界面或 API 中选择后即可推理。

| 模型 | 架构 | 说明 | 文档 |
|---|---|---|---|
| Anima | Cosmos-Predict2 DiT + Qwen3 + LLM adapter | 2B；单文件 DiT，外挂文本编码器与 VAE | [ANIMA.md](ANIMA.md) |
| Z-Image / Z-Image-Turbo | Tongyi NextDiT + Qwen3 + AutoencoderKL | 约 6B；Turbo 8 步生成 1024² | [ZIMAGE.md](ZIMAGE.md) |
| FLUX.1 Krea [dev] | Flux 架构 | 复用 Forge 现有 Flux 引擎，fp8 可在单张 24G 显卡运行 | [KREA.md](KREA.md) |
| Qwen-Image | 20B MMDiT + Qwen2.5-VL | fp8 量化后 20B 可在单张 24G 运行；显存充足时可直接使用 bf16 | [QWENIMAGE.md](QWENIMAGE.md) |

对于超出显存的模型（如 20B 的 Qwen-Image），沿用 Forge 内置的量化加载路径：提供 fp8 或 gguf 单文件即可，加载时 Forge 会将 `Linear` 替换为对应的量化算子。因此量化版本能够运行，即意味着在更大显存上以 bf16 加载同样可行——两者走的是同一条路径，仅由权重 dtype 自动区分。

## 依赖与其他改动

- 默认技术栈更新至 Python 3.12 + torch 2.7.1 / CUDA 12.8 + xformers。`requirements` 尽量采用版本范围而非精确锁定；对确需锁定的依赖，逐一核对了所用函数的行为是否发生变化。旧环境（RTX 20xx / CUDA 11）仍受支持，可通过一条命令安装（见下）。
- 修复了 Flux 系模型在较新 diffusers（≥ 0.38）上的加载失败：`FlowMatchEulerDiscreteScheduler.time_shift` 的签名已变更，导致原调用崩溃；现已内联修正，Flux 与 Krea 恢复正常。
- 在可行范围内保持 Forge 的 LoRA 加载与扩展接口兼容，并附带了常用扩展（如区域控制）。

## 安装

与 SD WebUI / Forge 相同：安装 Git 与 Python，克隆后运行。

```bash
git clone <this-repo-url> stable-diffusion-webui-on
cd stable-diffusion-webui-on
./webui.sh          # Linux / macOS
# webui-user.bat    # Windows
```

如需更干净的隔离环境，可使用附带的 `uv` 一键脚本（默认现代栈）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # 若尚未安装 uv
./install.sh
./run.sh
```

同一脚本可通过环境变量安装旧栈，例如 RTX 20xx / CUDA 11：

```bash
PYTHON_VERSION=3.10 TORCH_VERSION=2.3.1 TORCHVISION_VERSION=0.18.1 \
XFORMERS_VERSION=0.0.27 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./install.sh
```

各新架构另需对应的文本编码器 / VAE 等组件，下载与放置方式见各自文档；组件位置可通过 `ANIMA_ASSETS` / `ZIMAGE_ASSETS` / `QWENIMAGE_ASSETS` 环境变量指定。

## 适配新的架构

接入流程整理为一份 skill：[`skills/add-model-architecture`](skills/add-model-architecture/SKILL.md)。

总体思路：先以独立的 diffusers/torch 管线复现模型、作为逐位对照的参考实现，再将其封装为 Forge 的 diffusion engine 完成原生集成。其中较易出错的环节——采样调度中的 NaN、速度分量的符号、timestep 取 sigma 还是 1−sigma、变长文本条件是否需要 mask、fp8 权重处理、3D VAE 的显存占用——均已在文档中记录。

## License

沿用 Forge / SD WebUI 的许可协议，见 [LICENSE.txt](LICENSE.txt)（AGPL-3.0）。
