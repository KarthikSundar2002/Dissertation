import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch import optim

from utils import l_sample, sample, draw, input_sample

class Generator(L.LightningModule):
    def __init__(self, model, set_transformer_encoder, experiment_name, timesteps, noise_scheduler, noise_scheduler_sample, learning_rate,format_path,sample_size, encoded_dim=256, number_of_strokes=512):
        super().__init__()
        self.model = model
        self.set_transformer_encoder = set_transformer_encoder
        self.experiment_name = experiment_name
        self.timesteps = timesteps
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_sample = noise_scheduler_sample
        self.learning_rate = learning_rate
        self.encoded_dim = encoded_dim
        self.number_of_strokes = number_of_strokes
        self.format = format_path
        self.sample_size = sample_size
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # Set Transformer Encoder
        inp = batch[0] #[Batch, 512, 6]
        print(f"inp shape {inp.shape}")
        noise = torch.randn(inp.shape, device=self.device) #[Batch, 512, 6]
        print(f"noise shape {noise.shape}")
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (inp.shape[0],), device=self.device).long() #[Batch]
        print(f"timesteps shape {timesteps.shape}")
        inp = inp.transpose(0,1)
        noise = noise.transpose(0,1)
        noisy = self.noise_scheduler.add_noise(inp, noise, timesteps) #{Batch, 512, 6}
        noise = noise.transpose(0,1)
        inp = inp.transpose(0,1)
        noisy = noisy.transpose(0,1)
        print(f"noisy shape {noisy.shape}")
        noisy_enc = self.set_transformer_encoder(noisy) #[Batch, 512, 256]
        print(f"noisy_enc shape {noisy_enc.shape}")
        noisy_combined = torch.cat((noisy_enc, noisy), dim=-1) #[Batch, 512, 262]
        print(f"noisy_combined shape {noisy_combined.shape}")
        noise_pred = self.model(noisy_combined, timesteps) #[Batch, 512, 6]
        print(f"noise_pred shape {noise_pred.shape}")
        loss = F.mse_loss(noise_pred, noise)
        
        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Generate a latent vector
        stroke = input_sample(self.model, self.set_transformer_encoder, self.noise_scheduler_sample, 6, self.number_of_strokes, self.timesteps)
        print(f"stroke shape {stroke.shape}")
        # Save the generated drawing
        filename = f'Results/{self.experiment_name}/{self.current_epoch}.svg'
        draw(self.format, self.sample_size, filename, stroke)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[100000, 1000000, 2000000], 
            gamma=0.1
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
