import torch
from torch.autograd import grad
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image

# ------------------------------------------------------------
# DiffusionModel
# ------------------------------------------------------------
class DiffusionModel:
    """
    Stable Diffusion Class
    -> Generate Synthetic Image
    """
    # ------------------------------------------------------------
    # Basic
    # ------------------------------------------------------------
    def __init__(self, task, tsp_model, config):
        self.task = task

        self.tsp_model = tsp_model
        self.tsp_model.to(config.DEVICE)
        self.tsp_model.eval()

        self.device = config.DEVICE
        self.dtype = config.DTYPE
        self.model_name = config.MODEL_NAME
        self.image_size = config.IMAGE_SIZE[task]
        self.latent_size = self.image_size // 8

        self.sampling_steps = config.SAMPLING_STEPS
        self.guidance_scale = config.GUIDANCE_SCALE
        self.lambda_s = config.LAMBDA_S

        self.f32 = torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_name, torch_dtype=self.dtype).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(config=self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)

        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae
        self.vae = self.vae.to(dtype=self.f32)
        self.vae.enable_slicing()
        self.vae_scale = self.vae.config.scaling_factor
        self.scheduler = self.pipe.scheduler
        self.unet = self.pipe.unet

        for p in [self.tsp_model, self.text_encoder, self.vae, self.unet]:
            p.requires_grad_(False)

        del self.pipe

        self.uncond_embed = self._encode_prompt("")

    # ------------------------------------------------------------
    # Method
    # ------------------------------------------------------------ 
    def generate_sample(self, label, context):
        cond_embed = self._encode_prompt(context)
        text_embed = torch.cat([self.uncond_embed, cond_embed]).to(self.dtype)

        zt = torch.randn((1, self.unet.config.in_channels, self.latent_size, self.latent_size),
                         device=self.device, dtype=self.dtype) * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(self.sampling_steps, device=self.device)

        for t in self.scheduler.timesteps:
            zt = self._sampling_step(zt, t, text_embed, label)

        image = self._decode_to_image(zt)

        return image

    # ------------------------------------------------------------
    # Helper Method
    # ------------------------------------------------------------
    @torch.no_grad()
    def _encode_prompt(self, prompt):
        tokens = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                truncation=True, return_tensors="pt").to(self.device)

        return self.text_encoder(tokens.input_ids)[0]

    def _decode_to_tensor(self, zt):
        zt = zt.to(dtype=self.vae.dtype)
        image = self.vae.decode(zt / self.vae_scale).sample
        image = ((image / 2) + 0.5).clamp(0, 1)

        return image
    
    @torch.no_grad()
    def _decode_to_image(self, zt):
        image = self._decode_to_tensor(zt)
        image = image.float().cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).round().astype("uint8")

        return Image.fromarray(image)

    def _sampling_step(self, zt, t, text_embed, label):
        latent_input = torch.cat([zt, zt])
        latent_input = self.scheduler.scale_model_input(latent_input, t)

        with torch.no_grad():
            latent_output = self.unet(latent_input, t, encoder_hidden_states=text_embed).sample
        uncond, cond = latent_output.chunk(2)
        pred_eps = uncond + self.guidance_scale * (cond - uncond)

        adjust_zt = self._adjust_prediction(zt, t, pred_eps, label)
        zt = self.scheduler.step(adjust_zt, t, zt).prev_sample

        return zt

    def _adjust_prediction(self, zt, t, pred_eps, label):
        alpha_prod_t = self.scheduler.alphas_cumprod[t].to(device=zt.device, dtype=self.f32)
        s_t = self.lambda_s * ((1 - alpha_prod_t) ** 0.5)
        grad = self._calculate_guidance_gradient(zt, t, pred_eps, label, alpha_prod_t)

        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = grad.flatten(1).norm(dim=1).clamp(min=1e-6).view(-1, 1, 1, 1)
        grad = grad * (1.0 / grad_norm).clamp(max=1.0)

        warmup = alpha_prod_t.sqrt()
        s_t = s_t * warmup
        
        adjusted = pred_eps.float() + s_t * grad
        adjusted = adjusted.clamp(-6e4, 6e4)

        return adjusted.to(pred_eps.dtype)

    def _calculate_guidance_gradient(self, zt, t, pred_eps, label, alpha_prod_t):
        zt_f = zt.detach().float().requires_grad_(True)
        pred_eps_f = pred_eps.detach().float()

        with torch.enable_grad():
            z0_hat = self._predict_clean_sample(zt_f, t, pred_eps_f, alpha_prod_t)
            tensor = self._decode_to_tensor(z0_hat)
            loss = self.tsp_model.proxy_task_loss(tensor, label)
            gradient = grad(outputs=loss, inputs=zt_f)[0]

        return gradient

    def _predict_clean_sample(self, zt, t, pred_eps, alpha_prod_t):
        return (zt - ((1 - alpha_prod_t) ** 0.5) * pred_eps) / (alpha_prod_t ** 0.5)