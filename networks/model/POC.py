import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DenoiserMLP(nn.Module):
    """
    A simple MLP for the denoiser network that incorporates time embeddings.
    """
    def __init__(self, input_size, output_size, hidden_size=256, time_emb_dim=64):
        super().__init__()
        self.time_embedding = SinusoidalPosEmb(time_emb_dim)
        
        self.net = nn.Sequential(
            nn.Linear(input_size + time_emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, time):
        t_emb = self.time_embedding(time)
        
        # We need to expand the time embedding to match the number of strokes
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
        
        xt = torch.cat([x, t_emb_expanded], dim=-1)
        return self.net(xt)


class StrokeAttentionDiffusion(L.LightningModule):
    """
    A diffusion model that performs attention across strokes during denoising.
    """
    def __init__(self, attention_module, denoiser_model, noise_scheduler, noise_scheduler_sample, learning_rate=1e-4, timesteps_for_sampling=25):
        super().__init__()
        self.attention_module = attention_module
        self.denoiser_model = denoiser_model
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_sample = noise_scheduler_sample
        self.learning_rate = learning_rate
        self.timesteps_for_sampling = timesteps_for_sampling
        
        self.save_hyperparameters(ignore=["attention_module", "denoiser_model"])

    def training_step(self, batch, batch_idx):
        strokes = batch # (Batch, NumStrokes, StrokeDim)
        
        # 1. Create noise
        noise = torch.randn_like(strokes)
        
        # 2. Choose random timesteps
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (strokes.shape[0],), device=self.device).long()
        
        # 3. Add noise to the strokes
        noisy_strokes = self.noise_scheduler.add_noise(strokes, noise, timesteps)
        
        # 4. Get attention context
        # The attention module will create a representation for each stroke informed by others
        attention_context = self.attention_module(noisy_strokes)
        
        # 5. Combine noisy strokes with their context
        combined_input = torch.cat([noisy_strokes, attention_context], dim=-1)
        
        # 6. Predict noise
        predicted_noise = self.denoiser_model(combined_input, timesteps)
        
        # 7. Calculate loss
        loss = F.mse_loss(predicted_noise, noise)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # On validation, lets generate a sample
        if batch_idx == 0:
            num_strokes = batch.shape[1]
            stroke_dim = batch.shape[2]
            
            # Generate random noise to start from
            initial_noise = torch.randn((1, num_strokes, stroke_dim), device=self.device)
            
            self.noise_scheduler_sample.set_timesteps(self.timesteps_for_sampling)
            
            sample = initial_noise
            
            for t in self.noise_scheduler_sample.timesteps:
                with torch.no_grad():
                    # Get attention context
                    attention_context = self.attention_module(sample)
                    # Combine with context
                    combined_input = torch.cat([sample, attention_context], dim=-1)
                    # Predict noise
                    timestep_tensor = torch.tensor([t], device=self.device).long()
                    noise_pred = self.denoiser_model(combined_input, timestep_tensor)
                
                # Update sample
                sample = self.noise_scheduler_sample.step(noise_pred, t, sample).prev_sample

            # For visualization, you would need a function to draw the strokes
            # For example: draw_strokes_to_svg(f'generated_sample_epoch_{self.current_epoch}.svg', sample)
            # Since I don't have the drawing utility, I'll just log that we generated a sample.
            print(f"\nGenerated a sample of shape {sample.shape} at epoch {self.current_epoch}\n")


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer