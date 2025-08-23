import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch import optim
import os
from utils import l_sample, sample, draw
from flow_matching.path import GeodesicProbPath, AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler, VPScheduler
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
from chamferdist import ChamferDistance
from scipy.optimize import linear_sum_assignment

class SRM(L.LightningModule):
    def __init__(self, encoder, experiment_name, samples, sample_steps, format_path, sample_size, dim_in, lr, weight_mse=100):
        super().__init__()
        self.encoder = encoder
        self.experiment_name = experiment_name
        self.samples = samples
        self.sample_size = sample_size
        self.sample_steps = sample_steps
        self.format = format_path
        self.learning_rate = lr
        self.dim_in = dim_in
        #self.prob_path = AffineProbPath(scheduler=CondOTScheduler())
        self.prob_path = AffineProbPath(scheduler=CondOTScheduler())
        self.weight_mse = weight_mse
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        Strokes, noise = batch
        Strokes, Strokes_mask = Strokes
        #print(f"Strokes is a tensor of shape {Strokes.shape}")
        # Strokes = Strokes.transpose(1,2)
        # noise = noise.transpose(1,2)
        
        Strokes_mask = Strokes_mask.float()
        Strokes_mask = Strokes_mask.unsqueeze(-1)
        #noise = noise * Strokes_mask
        Strokes = Strokes * Strokes_mask
       
        # Strokes = Strokes.transpose(1,2)
        # noise = noise.transpose(1,2)

        t = torch.rand((Strokes.shape[0],), device=self.device)
        #noise = torch.randn(Strokes.shape, device=self.device)
        # cost_matrix = torch.cdist(noise, Strokes, p=1)
        # for i in range(cost_matrix.shape[0]):
        #     row_ind, col_ind = linear_sum_assignment(cost_matrix[i].cpu().numpy(), maximize=False)
        #     noise[i] = noise[i][row_ind]
        #     Strokes[i] = Strokes[i][col_ind]
        #     Strokes_mask[i] = Strokes_mask[i][col_ind]


        #path_sample = self.prob_path.sample(t=t, x_0=noise, x_1=Strokes)
        path_sample = self.prob_path.sample(t=t, x_0=noise, x_1=Strokes)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t
        

        t = t.unsqueeze(1)
        #Strokes_mask = Strokes_mask.float()
        #Strokes_mask = Strokes_mask.unsqueeze(-1)
        #x_t_masked = x_t * Strokes_mask
        #u_t_masked = u_t * Strokes_mask
        
       
        #mask = F.threshold(mask,0.5,0.0)
        
        velocity_field = self.encoder(x_t,t)
        #mask = self.encoder.compute_mask(x_t_masked,t)
        #out_mask = self.decoder.compute_mask(x_t,t,Strokes_mask,encoded)
        #Train
        # solver = ODESolver(self.encoder)
        # time_grid = torch.tensor([0.0,1.0],device=self.device)   
        # Output = solver.sample(noise,0.1,"euler",1e-5,1e-5,time_grid,False,False)
        # Output = Output * mask
        # chamfer = ChamferDistance()
        # loss_chamfer = chamfer(Output, x_t_masked)
        # self.log("chamfer_loss", loss_chamfer, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        loss_mse = F.mse_loss(velocity_field, u_t)
        self.log("mse_loss", loss_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #print(f"out_mask.dtype: {out_mask.dtype}")
        #print(f"Strokes_mask.dtype: {Strokes_mask.dtype}")
        #Strokes_mask = Strokes_mask.float()
        #bce_loss = F.binary_cross_entropy(mask, Strokes_mask)
        #loss = bce_loss + loss_mse
       # self.log("bce_loss", bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss_mse

    def validation_step(self, batch, batch_idx):
        #batch = batch.unsqueeze(0)
        Strokes, noise = batch
        Strokes, Strokes_mask = Strokes
        #Strokes, Strokes_mask = batch
        solver = ODESolver(self.encoder)
        Strokes_mask = Strokes_mask.float()
        Strokes_mask = Strokes_mask.unsqueeze(-1)
        #noise = noise * Strokes_mask
        Strokes = Strokes * Strokes_mask
        # Strokes = Strokes.transpose(1,2)
        # noise = noise.transpose(1,2)
       
        # noise = noise.transpose(1,2)
        # Strokes = Strokes.transpose(1,2)
        #noise = torch.randn(Strokes.shape, device=self.device)
        noise = torch.randn(Strokes.shape, device=self.device)
        x_0 = noise
        time_grid = torch.tensor([0.0,1.0],device=self.device)   
        Output = solver.sample(x_0,0.001,"euler",1e-5,1e-5,time_grid,False,False)
        t = torch.tensor([1.0],device=self.device)
        #mask = self.encoder.compute_mask(Output,t)
        #mask = F.threshold(mask,0.5,0.0)
        #Output = Output * mask
        #filename = f'./{self.experiment_name}/{self.current_epoch}.svg'
        filename = f'/scratch/ks02450/Results/{self.experiment_name}/{self.current_epoch}.svg'
        draw(self.format, self.sample_size, filename, Output)

    def test_step(self, batch, batch_idx):
        Strokes, noise = batch
        Strokes, Strokes_mask = Strokes
        solver = ODESolver(self.encoder)
        Strokes_mask = Strokes_mask.float()
        Strokes_mask = Strokes_mask.unsqueeze(-1)
        x_0 = noise * Strokes_mask
        time_grid = torch.tensor([0.0,1.0],device=self.device)   
        Output = solver.sample(x_0,0.01,"euler",1e-5,1e-5,time_grid,False,False)
        t = torch.tensor([1.0],device=self.device)
        mask = self.encoder.compute_mask(Output,t)
        Output = Output * mask
        filename = 'output.svg'
        draw(self.format, self.sample_size, filename, Output)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, min_lr=5e-5)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000,20000,30000,40000,50000,60000,70000,80000,90000,100000], gamma=0.5)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "train_loss"}]