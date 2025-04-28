import lightning as L
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
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # Encoder
        latent = batch
        noise = torch.randn(latent.shape, device=self.device)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (latent.shape[0],), device=self.device).long()
        noisy = self.noise_scheduler.add_noise(latent, noise, timesteps)
        noise_pred = self.model(noisy, timesteps)
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