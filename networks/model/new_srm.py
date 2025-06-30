import pytorch_lightning as L
from torch import optim
import torch
import torch.nn.functional as F

from utils import sample, draw, draw_points_svg, input_sample

class SRM(L.LightningModule):
    def __init__(self, encoder, decoder, noise_scheduler, noise_scheduler_sample, experiment_name, samples, sample_steps, format_path, sample_size, dim_in, lr):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_sample = noise_scheduler_sample
        self.experiment_name = experiment_name
        self.samples = samples
        self.sample_size = sample_size
        self.sample_steps = sample_steps
        self.format = format_path
        self.learning_rate = lr
        self.dim_in = dim_in
        self.save_hyperparameters(ignore=['encoder', 'decoder', 'noise_scheduler', 'noise_scheduler_sample'])

    def training_step(self, batch, batch_idx):
        #Encoder
        # print(f"batch is a list of length {len(batch)}")
        # Batch is a list of length 2 - Set and Strokes
        Set = batch
        # print(f"Set is a tensor of shape {Set.shape}")
        encoded = self.encoder(Set)
        #Decoder
        #1 instead of 0 to use collate
        Strokes = batch
        # print(f"Strokes is a tensor of shape {Strokes.shape}")
        noise = torch.randn(Strokes.shape, device=self.device)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (Strokes.shape[1],), device=self.device).long()
        noisy = self.noise_scheduler.add_noise(Strokes, noise, timesteps)

        #Train
        noise_pred = self.decoder(noisy, timesteps, encoded)
        # print(f"noise_pred is a tensor of shape {noise_pred.shape}")
        # print(f"noise is a tensor of shape {noise.shape}")
        loss_mse = F.mse_loss(noise_pred, noise)

        #KL
        # KLD = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        # KLS = (0) #* self.current_epoch
        loss = loss_mse #+ (KLS * KLD)
        
        # Log metrics to see in the Lightning dashboard
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #batch = batch.unsqueeze(0)
        encoded = self.encoder(batch)
        for i in range(len(encoded)):
            filename = f'/scratch/ks02450/Results/{self.experiment_name}/{self.current_epoch}_{i}.svg'
            stroke = input_sample(self.decoder, self.encoder, self.noise_scheduler_sample,encoded[0].unsqueeze(0), dim_per_stroke=6, number_of_strokes=self.samples, timesteps=self.sample_steps)
            # print(f"stroke is a tensor of shape {stroke.shape}")
            draw(self.format, self.sample_size, filename, stroke)

    def test_step(self, batch, batch_idx):
        encoded = self.encoder(batch)
        filename = f'/scratch/ks02450/Samples/{self.experiment_name}/{batch_idx}.svg'
        stroke = sample(self.samples, self.sample_steps, self.decoder, self.noise_scheduler, dim_per_stroke=30, number_of_strokes=4, timesteps=self.sample_steps)
        draw_points_svg(self.format, self.sample_size, filename, stroke)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000], gamma=0.5)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]