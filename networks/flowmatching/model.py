import lightning as L
import torch
import torch.nn.functional as F
from torch import optim

from utils import l_sample, sample, draw

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

class LSG(L.LightningModule):
    def __init__(self, model, srm, experiment_name, timesteps, learning_rate):
        super().__init__()
        self.model = model
        self.srm = srm
        self.experiment_name = experiment_name
        self.timesteps = timesteps
        self.prob_path = AffineProbPath(scheduler=CondOTScheduler())
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # Get the data
        x = batch
        t = torch.rand(x.shape[0], device=self.device)
        noise = torch.randn_like(x, device=self.device)
        # Get the flow field
        path_sample = self.prob_path.sample(t=t, x_0=noise, x_t=x)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t
        
        flow_field = self.model(x_t, t)
        
        # Compute the loss using the flow matching objective
        loss = F.mse_loss(flow_field, self.prob_path.vector_field(x, t))
        
        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Generate a latent vector using flow matching
        solver = ODESolver(self.model, self.prob_path)
        Latent = solver.sample(self.timesteps)
        
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