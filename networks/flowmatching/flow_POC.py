import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch import optim
import os
from utils import l_sample, sample, draw

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

class SRM(L.LightningModule):
    def __init__(self, encoder, decoder, noise_scheduler, noise_scheduler_sample, experiment_name, samples, sample_steps, format_path, sample_size, dim_in, lr, weight_mse=100):
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
        self.prob_path = AffineProbPath(scheduler=CondOTScheduler())
        self.weight_mse = weight_mse
        self.save_hyperparameters(ignore=['encoder', 'decoder', 'noise_scheduler', 'noise_scheduler_sample'])

    def training_step(self, batch, batch_idx):
        Set, Set_mask = batch
        #print(f"Set is a tensor of shape {Set.shape}")
        encoded = self.encoder(Set, Set_mask)
        #print(f"condition is a tensor of shape {condition.shape}")
        # Decoder
        # 1 instead of 0 to use collate
        Strokes, Strokes_mask = batch
        #print(f"Strokes is a tensor of shape {Strokes.shape}")
        noise = torch.randn(Strokes.shape, device=self.device)
        t = torch.rand((Strokes.shape[0],), device=self.device)
       
        path_sample = self.prob_path.sample(t=t, x_0=noise, x_1=Strokes)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t
        
        t = t.unsqueeze(1)
        velocity_field = self.decoder(x_t,t,Strokes_mask,encoded)
       
        #out_mask = self.decoder.compute_mask(x_t,t,Strokes_mask,encoded)
        #Train
        loss = F.mse_loss(velocity_field, u_t)
        #self.log("mse_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #print(f"out_mask.dtype: {out_mask.dtype}")
        #print(f"Strokes_mask.dtype: {Strokes_mask.dtype}")
        #Strokes_mask = Strokes_mask.float()
        #bce_loss = F.binary_cross_entropy(out_mask, Strokes_mask)
        #loss = self.weight_mse * loss + bce_loss
        #self.log("bce_loss", bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #batch = batch.unsqueeze(0)
        Set, Set_mask = batch
        encoded = self.encoder(Set, Set_mask)
        solver = ODESolver(self.decoder)
        x_0 = torch.randn(Set.shape, device=self.device)
        time_grid = torch.tensor([0.0,1.0],device=self.device)
        model_args = {'x_mask':Set_mask,'y':encoded}      
        Output = solver.sample(x_0,None,"euler",1e-5,1e-5,time_grid,False,False,**model_args)
        t = torch.ones((Set.shape[0],1), device=self.device)
        Output_mask = self.decoder.compute_mask(Output,t,Set_mask,encoded)
        # for i in range(len(condition)):
        #     filename = '/scratch/ks02450/Results/{}/{}_{}.svg'.format(self.experiment_name, self.current_epoch, i)
        #     stroke = sample(self.samples, self.sample_steps, self.decoder, self.noise_scheduler_sample, mu[i], self.dim_in)
        #     tensor_to_svg(stroke, filename)
        
        # x = batch
        # encoded, condition, mu, sigma = self.encoder(x)
        filename = f'/scratch/ks02450/Results/{self.experiment_name}/{self.current_epoch}.svg'
        #print(f"Output.shape: {Output.shape}")
        #print(f"Output_mask.shape: {Output_mask.shape}")
        Output_mask = Output_mask.unsqueeze(-1)
        # output = Output.permute(0,2,1)
        output_masked = Output * Output_mask
        # output_masked = output_masked.permute(0,2,1)
        draw(self.format, self.sample_size, filename, output_masked)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000], gamma=0.5)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]