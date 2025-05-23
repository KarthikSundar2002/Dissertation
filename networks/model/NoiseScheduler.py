import torch
import torch.nn.functional as F
import lightning as L

class NoiseScheduler(L.LightningModule):
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"):
        super(NoiseScheduler, self).__init__()
        self.num_timesteps = num_timesteps
        self.num_train_timesteps = num_timesteps
        
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32, device=self.device
            )
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32, device=self.device
            ) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0, dtype=torch.float32)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t].to(self.device)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].to(self.device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t].to(self.device)
        s2 = self.posterior_mean_coef2[t].to(self.device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance.to(self.device)

    def step(self, model_output, timestep, sample):
        t = timestep.to(self.device)
        pred_original_sample = self.reconstruct_x0(sample, t, model_output).to(self.device)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t).to(self.device)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output).to(self.device)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample.to(self.device)

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod.to(self.device)[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod.to(self.device)[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return (s1 * x_start + s2 * x_noise)

    def __len__(self):
        return self.num_timesteps