import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
from utils import l_sample, sample, draw

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

from networks.model.positional_embeddings import PositionalEmbedding
from networks.flowmatching.simple_mlp import SimpleMLP
class SRM(L.LightningModule):
    def __init__(self, experiment_name, num_of_strokes, embed_dim, dim_in, lr, num_layers, dim_hidden, format_path, sample_size):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        dim_output = num_of_strokes * dim_in
        self.model = SimpleMLP(num_of_strokes, embed_dim, dim_in, lr, num_layers, dim_hidden, dim_output)
        self.experiment_name = experiment_name
        self.num_of_strokes = num_of_strokes
        self.format = format_path
        self.dim_in = dim_in
        self.learning_rate = lr
        self.embed_dim = embed_dim
        self.sample_size = sample_size
        self.prob_path = AffineProbPath(scheduler=CondOTScheduler())
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        Set, Set_mask = batch
        
        #print(f"Strokes is a tensor of shape {Strokes.shape}")
        #input_vec = torch.flatten(Set, start_dim=1)
        noise = torch.randn(Set.shape, device=self.device)
        t = torch.rand((Set.shape[0],), device=self.device)
       
        path_sample = self.prob_path.sample(t=t, x_0=noise, x_1=Set)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t

        velocity_field = self.model(Set, t)
       
        #out_mask = self.decoder.compute_mask(x_t,t,Strokes_mask,encoded)
        #Train
        loss = F.mse_loss(velocity_field, u_t)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #batch = batch.unsqueeze(0)
        Set, Set_mask = batch
        #input_vec = torch.flatten(Set, start_dim=1)
        x_0 = torch.randn(Set.shape, device=self.device)
        time_grid = torch.tensor([0.0,1.0], device=self.device)
        solver = ODESolver(self.model)
        Output = solver.sample(x_init=x_0,
                               time_grid=time_grid,
                               method="euler",
                               step_size=20)
        Output = Output.reshape(Set.shape)
        filename = f'/scratch/ks02450/Results/{self.experiment_name}/{self.current_epoch}.svg'
       
        draw(self.format, self.sample_size, filename, Output)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000], gamma=0.5)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
        #return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "train_loss"}]
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]