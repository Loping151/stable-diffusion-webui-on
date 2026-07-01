<div align="center">

# Stable Diffusion WebUI On

**尽可能多地支持新的 AIGC 模型架构,并把 LoRA / 插件生态延续下去。**

`更多架构 · 更新依赖 · LoRA/插件延续 · 个人长期维护`

<sub>**中文** · [English](README_en.md)</sub>

[![Base](https://img.shields.io/badge/based%20on-SD%20WebUI%20Forge-8A2BE2)](https://github.com/lllyasviel/stable-diffusion-webui-forge)
[![Python](https://img.shields.io/badge/Python-3.10~3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Torch](https://img.shields.io/badge/torch-2.3~2.7%20/%20cu121~cu128-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-yellow.svg)](LICENSE.txt)

</div>

---

Forge 是把优化和新模型「锻造」进 SD WebUI;`-on` 想做的,是把这个开关一直**开着** —— 新出的模型架构、新的 LoRA、新的插件,来一个、接一个、点亮一个。

**人话:** 这是 [SD WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) 的一个个人维护分支。Forge 更新变慢之后,我把它捡起来继续修 bug、更新依赖,重点是**让那些 Forge 原本加载不了的新架构模型(Anima、Z-Image、Qwen-Image……)能在同一个 WebUI 里直接选中就出图**,顺便尽量不破坏原有的 LoRA 和插件生态。

> 实验性质,个人自用为主。欢迎试用和提 issue,但别指望它有官方的稳定性承诺。

## 血缘与致谢

这个项目完全站在前人的肩膀上:

- **[lllyasviel/stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)** —— *Forge*,本仓库的直接上游。它在 SD WebUI 之上重写了显存管理、加速推理、加了很多实验特性。("Forge" 这名字致敬 Minecraft Forge。)
- **[AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)** —— 一切的起点,Forge 基于它的 1.10.1。

Forge 原本的功能教程 / NEWS 仍然适用,见 [Forge 仓库](https://github.com/lllyasviel/stable-diffusion-webui-forge)。本仓库只在它之上做增量。

## 新增的原生模型架构

Forge 原生只认 SD1/SD2/SDXL/SD3/Flux 这几家。下面这些新架构,在本分支里是**原生支持**的 —— 把 checkpoint 丢进 `models/Stable-diffusion/`,在界面/API 里选中,和普通模型一样出图:

| 模型 | 架构 | 说明 | 文档 |
|---|---|---|---|
| **Anima** | Cosmos-Predict2 DiT + Qwen3 + LLM adapter | 2B,单文件 DiT + 附属组件 | [ANIMA.md](ANIMA.md) |
| **Z-Image / Z-Image-Turbo** | Tongyi NextDiT + Qwen3 + AutoencoderKL | ~6B,Turbo 8 步出 1024² | [ZIMAGE.md](ZIMAGE.md) |
| **FLUX.1 Krea [dev]** | Flux 架构 | 直接复用 Flux 引擎,fp8 单卡 24G 可跑 | [KREA.md](KREA.md) |
| **Qwen-Image** | 20B MMDiT + Qwen2.5-VL | fp8 量化后 20B 塞进单卡 24G;换大卡直接上 bf16 | [QWENIMAGE.md](QWENIMAGE.md) |

关于 **量化**:大到装不下的模型(比如 20B 的 Qwen-Image)走 Forge 自带的量化路径 —— 丢 fp8 / gguf 的单文件进去,Forge 在加载时会自动把 `Linear` 换成量化算子。**量化能跑通,等于换大显存后 bf16 也能跑通**(同一条路,dtype 自动识别)。

## 除了新模型,还改了什么

- **依赖现代化,同时不抛弃老卡。** 默认栈升到 Python 3.12 + torch 2.7.1 / CUDA 12.8 + xformers;`requirements` 尽量用版本范围而不是死锁,对确实要锁的依赖逐个确认过行为是否变化。老栈(RTX 20xx / CUDA 11)一条命令就能装(见下)。
- **修好了 Flux 全家在新版 diffusers 上的加载。** diffusers ≥ 0.38 改了 `FlowMatchEulerDiscreteScheduler.time_shift` 的签名,Forge 原来的调用会直接崩;已内联修复,Flux / Krea 等重新可用。
- **LoRA 与插件生态延续。** 尽量保持 Forge 的 LoRA 加载和扩展接口可用;区域控制等常用插件已一并带过来。
- **加架构有章可循。** 怎么再接一个新架构,写成了可复用的方法论 skill(见下),把踩过的坑(NaN 调度、速度符号、timestep 约定、变长带 mask 的文本条件、fp8 norm、3D VAE 爆显存)都记下来了。

## 安装

和 SD WebUI / Forge 一样,装好 Git 和 Python,克隆后运行:

```bash
git clone <this-repo-url> stable-diffusion-webui-on
cd stable-diffusion-webui-on
./webui.sh          # Linux / macOS
# webui-user.bat    # Windows
```

想要更快、更干净的隔离环境,用自带的 `uv` 一键脚本(默认现代栈:Python 3.12 + torch 2.7.1 / cu128):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # 没有 uv 的话先装
./install.sh
./run.sh
```

同一个脚本靠环境变量就能装**老栈**(例如 RTX 20xx / CUDA 11):

```bash
PYTHON_VERSION=3.10 TORCH_VERSION=2.3.1 TORCHVISION_VERSION=0.18.1 \
XFORMERS_VERSION=0.0.27 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./install.sh
```

各新架构还需要额外的文本编码器 / VAE 等组件,下载与放置见对应文档(默认可用 `ANIMA_ASSETS` / `ZIMAGE_ASSETS` / `QWENIMAGE_ASSETS` 环境变量指定位置)。

## 想再接一个新架构?

流程写成了一个可复用的 skill:[`.claude/skills/add-model-architecture`](.claude/skills/add-model-architecture/SKILL.md)。

一句话概括:**先写个独立的 diffusers/torch 管线把模型跑通(当作 bit-exact 的对照参考)→ 再包成 Forge 的 diffusion engine 原生集成**。中间真正费时间的地方(采样器 NaN、速度正负号、timestep 是 sigma 还是 1-sigma、变长条件要不要 mask、fp8 权重、3D VAE 显存)都在里面。

## License

沿用 Forge / SD WebUI 的许可,见 [LICENSE.txt](LICENSE.txt)(AGPL-3.0)。
