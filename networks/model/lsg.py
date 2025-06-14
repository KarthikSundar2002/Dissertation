import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch import optim

from utils import l_sample, sample, draw

class LSG(L.LightningModule):
    def __init__(self, model, srm, experiment_name, timesteps, noise_scheduler, noise_scheduler_sample, learning_rate):
        super().__init__()
        self.model = model
        self.srm = srm
        self.experiment_name = experiment_name
        self.timesteps = timesteps
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_sample = noise_scheduler_sample
        self.learning_rate = learning_rate
        #self.save_hyperparameters(ignore=["model", "srm"])

    def training_step(self, batch, batch_idx):
        # Encoder
        latent = batch #[128, 1, 256]
        noise = torch.randn(latent.shape, device=self.device)
        latent =  latent.reshape(latent.shape[1],latent.shape[0],latent.shape[2])
        noise = noise.reshape(noise.shape[1],noise.shape[0],noise.shape[2])
        # print(f"latent shape {latent.shape}")
        # print(f"noise shape {noise.shape}")
        # print(f"noise shape {noise.shape}")
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (latent.shape[0],), device=self.device).long()
        # print(f"timesteps shape {timesteps.shape}")
        noisy = self.noise_scheduler.add_noise(latent, noise, timesteps) #{128,128,256}
        # print(f"noisy shape {noisy.shape}")
        noisy = noisy.reshape(noisy.shape[1],noisy.shape[0],noisy.shape[2])
        noise_pred = self.model(noisy, timesteps)
        noise = noise.reshape(noise.shape[1],noise.shape[0],noise.shape[2])
        loss = F.mse_loss(noise_pred, noise)
        
        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Generate a latent vector
        Latent = l_sample(self.timesteps, self.model, self.noise_scheduler_sample)
        
        # Decode the latent vector into strokes
        stroke = sample(
            self.srm.samples, 
            self.srm.sample_steps, 
            self.srm.decoder, 
            self.srm.noise_scheduler_sample, 
            Latent, 
            self.srm.dim_in
        )
        
        # Save the generated drawing
        filename = f'Results/{self.experiment_name}/{self.current_epoch}.svg'
        draw(self.srm.format, self.srm.sample_size, filename, stroke)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[100000, 1000000, 2000000], 
            gamma=0.1
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
