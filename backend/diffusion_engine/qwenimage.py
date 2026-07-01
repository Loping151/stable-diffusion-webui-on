import os
import torch

# Inject the Qwen-Image architecture into the git-ignored huggingface_guess vendor copy before we
# reference model_list.QwenImage below. Idempotent.
from backend.nn import qwenimage_hf_register as _qwenimage_hf_register
_qwenimage_hf_register.register()

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.unet import UnetPatcher
from backend.modules.k_prediction import PredictionFlux
from backend import memory_management


# Components not present in the single-file DiT checkpoint (Qwen2.5-VL text encoder, its
# tokenizer, and the AutoencoderKLQwenImage VAE) load from this directory.
QWENIMAGE_ASSETS = os.environ.get(
    "QWENIMAGE_ASSETS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                 "models", "qwen-image", "Qwen-Image"),
)

# Qwen-Image prompt template: the DiT is trained on Qwen2.5-VL hidden states of this wrapped
# prompt, with the leading template tokens dropped (see the diffusers QwenImagePipeline).
QWEN_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
QWEN_TEMPLATE_DROP_IDX = 34
QWEN_MAX_TOKENS = 512     # pipeline caps prompt_embeds at max_sequence_length=512


class _QwenVAEWrapper:
    latent_channels = 16

    def __init__(self, model):
        self.first_stage_model = model

    def shallow_copy(self):
        return _QwenVAEWrapper(self.first_stage_model)


class QwenImage(ForgeDiffusionEngine):
    matched_guesses = [model_list.QwenImage]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False
        # Qwen-Image uses real CFG (true_cfg_scale + negative prompt), not Flux-style distilled
        # guidance (guidance_embeds=False), so leave use_distilled_cfg_scale unset/False.

        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
        from diffusers import AutoencoderKLQwenImage

        self.device = memory_management.get_torch_device()
        self.te_dtype = memory_management.text_encoder_dtype()

        # --- Qwen2.5-VL text encoder + tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(QWENIMAGE_ASSETS, "tokenizer"))
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            os.path.join(QWENIMAGE_ASSETS, "text_encoder"), dtype=self.te_dtype
        ).eval()
        self.text_encoder.requires_grad_(False)
        self.text_encoder = self.text_encoder.to(memory_management.text_encoder_offload_device())

        # --- VAE (AutoencoderKLQwenImage, 16ch, Wan-style latents_mean/std) ---
        vae_model = AutoencoderKLQwenImage.from_pretrained(
            os.path.join(QWENIMAGE_ASSETS, "vae"), torch_dtype=torch.float32
        ).eval()
        vae_model.requires_grad_(False)
        # The Qwen-Image 3D VAE decode of a 1024^2 image is very memory-heavy and OOMs a 24 GB
        # card next to the 20B DiT; tile + slice it to bound peak decode memory.
        vae_model.enable_tiling()
        vae_model.enable_slicing()
        self.vae_model = vae_model
        zdim = vae_model.config.z_dim
        self._lat_mean = torch.tensor(vae_model.config.latents_mean).view(1, zdim, 1, 1, 1)
        self._lat_std = torch.tensor(vae_model.config.latents_std).view(1, zdim, 1, 1, 1)

        # --- DiT (from the loaded checkpoint) wrapped for Forge sampling ---
        transformer = huggingface_components['transformer']

        # fp8 checkpoints: Forge stores every weight as fp8 and dequantizes Linear/Conv weights
        # in-forward via its patched ops. But this DiT's RMSNorm/LayerNorm/Embedding do raw
        # weight multiplies/lookups that PyTorch refuses on fp8 ("Promotion for Float8 Types is
        # not supported"). Cast just those non-Linear/Conv params up to the compute dtype; the
        # bulk (Linear weights) stays quantized so the 20B model still fits a 24 GB card.
        comp = getattr(transformer, "computation_dtype", torch.bfloat16)
        _fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
        for module in transformer.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                continue
            for pname, p in list(module.named_parameters(recurse=False)):
                if p is not None and p.dtype in _fp8:
                    module._parameters[pname] = torch.nn.Parameter(p.data.to(comp), requires_grad=False)
        unet = UnetPatcher.from_model(
            model=transformer, diffusers_scheduler=None,
            k_predictor=PredictionFlux(seq_len=4096, base_seq_len=256, max_seq_len=4096,
                                       base_shift=0.5, max_shift=1.15),
            config=estimated_config,
        )

        vae = _QwenVAEWrapper(vae_model)
        self.forge_objects = ForgeObjects(unet=unet, clip=None, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        dev, dt = self.device, self.te_dtype
        self.text_encoder.to(dev)
        outs = []
        for p in prompt:
            txt = QWEN_PROMPT_TEMPLATE.format(p)
            ti = self.tokenizer(
                [txt], max_length=QWEN_MAX_TOKENS + QWEN_TEMPLATE_DROP_IDX,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            ids = ti.input_ids.to(dev)
            m = ti.attention_mask.to(dev).bool()
            h = self.text_encoder(input_ids=ids, attention_mask=m, output_hidden_states=True).hidden_states[-1]
            # keep only real (unmasked) tokens, drop the template prefix, cap at QWEN_MAX_TOKENS
            real = h[0][m[0]][QWEN_TEMPLATE_DROP_IDX:][:QWEN_MAX_TOKENS].to(dt)
            padded = h.new_zeros(QWEN_MAX_TOKENS, real.shape[-1]).to(dt)
            padded[: real.shape[0]] = real
            outs.append(padded.unsqueeze(0))
        self.text_encoder.to(memory_management.text_encoder_offload_device())
        return torch.cat(outs, dim=0)   # [B, QWEN_MAX_TOKENS, 3584], zero-padded

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        n = len(self.tokenizer(prompt).input_ids)
        return n, max(255, n)

    def _stats(self, dev, dt):
        return self._lat_mean.to(dev, dt), self._lat_std.to(dev, dt)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        self.vae_model.to(self.device)
        v = x.to(self.device, torch.float32).unsqueeze(2)          # [B,3,1,H,W]
        lat = self.vae_model.encode(v).latent_dist.sample()        # [B,16,1,H/8,W/8]
        m, s = self._stats(lat.device, lat.dtype)
        z = (lat - m) / s
        return z.squeeze(2).to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        # Offload the DiT (and any other loaded models) so the memory-heavy 3D VAE decode fits.
        memory_management.free_memory(1e30, self.device, free_all=True)
        self.vae_model.to(self.device)
        z = x.to(self.device, torch.float32).unsqueeze(2)          # [B,16,1,H,W]
        m, s = self._stats(z.device, z.dtype)
        lat = z * s + m
        img = self.vae_model.decode(lat).sample                    # [B,3,1,H,W]
        return img.squeeze(2).to(x)
