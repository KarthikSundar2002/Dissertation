import pytorch_lightning as L
from torch import optim
import torch
import torch.nn.functional as F

from utils import sample, draw, draw_points_svg, input_sample, tensor_to_svg


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
        #batch = batch.unsqueeze(0)
        #print(f"batch is a tensor of shape {batch.shape}")
        # Batch is a list of length 2 - Set and Strokes
        Set, Set_mask = batch
        print(f"Set is a tensor of shape {Set.shape}")
        encoded, condition, mu, sigma = self.encoder(Set, Set_mask)
        print(f"condition is a tensor of shape {condition.shape}")
        # Decoder
        # 1 instead of 0 to use collate
        Strokes, Strokes_mask = batch
        print(f"Strokes is a tensor of shape {Strokes.shape}")
        noise = torch.randn(Strokes.shape, device=self.device)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (Strokes.shape[1],), device=self.device).long()
        noisy = self.noise_scheduler.add_noise(Strokes, noise, timesteps)
        mask = self.decoder.compute_mask(Strokes, timesteps, condition, Strokes_mask)
        mask = F.threshold(mask, 0.5, 0)
        #Train
        noise_pred = self.decoder(noisy, timesteps, condition, mask)
        print(f"noise_pred is a tensor of shape {noise_pred.shape}")
        print(f"noise is a tensor of shape {noise.shape}")
        # cd = list()
        # for o, t in zip(z_out, Set):  # [m, C]
        #     o_ = o.unsqueeze(1).repeat(1, t.size(0), 1)  # [m, m, C]
        #     t_ = t.unsqueeze(0).repeat(o.size(0), 1, 1)  # [m, m, C]
        #     l2 = (o_ - t_).pow(2).sum(dim=-1)  # [m, m]
        #     tdist = l2.min(0)[0].sum()  # min over outputs
        #     odist = l2.min(1)[0].sum()  # min over targets
        #     cd.append(odist + tdist)
        # loss_mse = sum(cd) / float(len(cd))

        #KL
        KLD = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        KLS = (1.0/100) * self.current_epoch
        print(f"mask.shape: {mask.shape}")
        print(f"Strokes_mask.shape: {Strokes_mask.shape}")
        Strokes_mask = Strokes_mask.float()
        mask = mask.squeeze(-1)
        print(f"mask.dtype: {mask.dtype}")
        print(f"Strokes_mask.dtype: {Strokes_mask.dtype}")
        bce_loss = F.binary_cross_entropy(mask, Strokes_mask)
        loss_mse = F.mse_loss(noise_pred, noise)
        loss = loss_mse + bce_loss
        #loss = loss_mse + (KLS * KLD)

        
        # Log metrics to see in the Lightning dashboard
        self.log("train_loss_bce", bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_mse", loss_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train_loss_kld", KLD, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train_loss_kls", KLS, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #batch = batch.unsq
        encoded,condition, mu, sigma = self.encoder(batch[0], batch[1])
        for i in range(len(condition)): 
            filename = '/scratch/ks02450/Results/{}/{}_{}.svg'.format(self.experiment_name, self.current_epoch, i)
            stroke = sample(self.samples, self.sample_steps, self.decoder, self.noise_scheduler_sample, mu[i], self.dim_in, batch[1])
            draw(self.format, self.sample_size, filename, stroke)
        


    def test_step(self, batch, batch_idx):
        encoded,condition, mu, sigma = self.encoder(batch[0])
        print(mu)
        filename = f'/scratch/ks02450/Samples/{self.experiment_name}/{batch_idx}.svg'
        stroke = sample(self.samples, self.sample_steps, self.decoder, self.noise_scheduler, dim_per_stroke=30, number_of_strokes=4, timesteps=self.sample_steps)
        draw_points_svg(self.format, self.sample_size, filename, stroke)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000], gamma=0.5)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]