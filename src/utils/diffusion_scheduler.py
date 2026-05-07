import torch


class DiffusionScheduler:
    """Simple epsilon-prediction diffusion scheduler for action denoising."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1, dtype=torch.long)

    def set_timesteps(self, num_inference_steps: int, training: bool = False):
        del training
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long
        )
        return self.timesteps

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor, t_dim: int = 0):
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        t = timesteps.long().to(original_samples.device).clamp(0, self.num_train_timesteps - 1)
        shape = [1] * original_samples.ndim
        shape[t_dim] = t.shape[0]
        sqrt_alpha = self.sqrt_alphas_cumprod.to(original_samples.device)[t].view(shape).to(original_samples.dtype)
        sqrt_om_alpha = self.sqrt_one_minus_alphas_cumprod.to(original_samples.device)[t].view(shape).to(original_samples.dtype)
        return sqrt_alpha * original_samples + sqrt_om_alpha * noise

    def training_target(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor):
        del sample, timestep
        return noise

    def training_weight(self, timestep: torch.Tensor):
        # Keep the same interface as the former scheduler.
        return torch.ones_like(timestep, dtype=torch.float32, device=timestep.device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep,
        sample: torch.Tensor,
        to_final: bool = False,
        prev_timestep=None,
        **kwargs,
    ):
        del kwargs
        if isinstance(timestep, torch.Tensor):
            timestep_value = int(timestep.flatten()[0].item())
        else:
            timestep_value = int(timestep)
        t = max(0, min(self.num_train_timesteps - 1, timestep_value))

        alpha_bar_t = self.alphas_cumprod[t].to(sample.device, sample.dtype)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_om_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        pred_x0 = (sample - sqrt_om_alpha_bar_t * model_output) / sqrt_alpha_bar_t.clamp_min(1e-8)

        if to_final or t <= 0:
            return pred_x0

        if prev_timestep is None:
            prev_t = max(t - 1, 0)
        elif isinstance(prev_timestep, torch.Tensor):
            prev_t = int(prev_timestep.flatten()[0].item())
        else:
            prev_t = int(prev_timestep)
        prev_t = max(0, min(self.num_train_timesteps - 1, prev_t))
        alpha_bar_prev = self.alphas_cumprod[prev_t].to(sample.device, sample.dtype)
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
        sqrt_om_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)
        # DDIM-like deterministic update (eta=0).
        prev_sample = sqrt_alpha_bar_prev * pred_x0 + sqrt_om_alpha_bar_prev * model_output
        return prev_sample
