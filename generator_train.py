from ast import arg
from networks.model.generator import Generator 
from networks.model.set_transformer import SetTransformer
from networks.model.mlp import MLP as L_MLP
import torch
from torch.utils.data import DataLoader
import os
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from diffusers import DDIMScheduler, DDPMScheduler
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint, LearningRateMonitor
import wandb
from Data_Set import Tensor, my_collate
from options.args import argument_parser




device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps"
experiment_name = 'Generator-Train-run'
format_path = 'format.svg'
train_path = '10k.pt'
sample_size = 512
learning_rate = 1e-4
size = 512
BATCH_SIZE = 256

dim_in = 6
encoded_dim = 256
number_of_strokes = 512
hidden_size = 32
samples = 1000
steps = 4000
sample_steps = 25
beta_schedule = 'scaled_linear'
wand_b_key = '117905e69dff43b1635103618ba74a5593104105'
gpu_num = 1
wandb.login(key=wand_b_key)
wandb_logger = WandbLogger(name=experiment_name,project='Your Stroke Cloud',save_dir="/wandb")
train_set = Tensor(train_path)

train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, collate_fn= my_collate, pin_memory=True)
torch.set_float32_matmul_precision("medium")

checkpoint_callback = ModelCheckpoint(
    dirpath="Models/{}/".format(experiment_name),
    filename="{epoch:02d}-{global_step}",
    save_last=False,
    every_n_epochs=10000,
    save_on_train_epoch_end=True,
)

model= L_MLP(
        hidden_size=hidden_size,
        hidden_layers=6,
        emb_size=64,
        time_emb= "sinusoidal",
        input_emb = "sinusoidal")
set_transformer = SetTransformer(
        dim_input=dim_in,
        num_outputs=1,
        dim_output=256,
        num_inds=32,
        dim_hidden=256,
        num_heads=16,
        ln=True)

scheduler = DDPMScheduler(beta_end=2e-2, beta_start=1e-4, num_train_timesteps = steps, beta_schedule=beta_schedule)
ddim_s = DDIMScheduler(beta_end=2e-2, beta_start=1e-4, num_train_timesteps = steps, beta_schedule=beta_schedule)
ddim_s.set_timesteps(sample_steps)
sample_steps = list(range(25))
lr_monitor = LearningRateMonitor(logging_interval='epoch')
generator = Generator(model, set_transformer.enc, experiment_name, sample_steps, scheduler, ddim_s, learning_rate, format_path, sample_size, encoded_dim, number_of_strokes)

if not os.path.exists("Results/{}".format(experiment_name)):
        os.makedirs("Results/{}".format(experiment_name))

if not os.path.exists("Models/{}".format(experiment_name)):
        os.makedirs("Models/{}".format(experiment_name))

trainer = L.Trainer(accelerator='gpu', devices=gpu_num, strategy='auto',logger=wandb_logger, max_epochs= 5000000,
                    check_val_every_n_epoch=10000, enable_progress_bar=True, profiler="simple",
                    callbacks=[StochasticWeightAveraging(swa_lrs=learning_rate),checkpoint_callback, lr_monitor], benchmark=True)
trainer.fit(model=generator, train_dataloaders=train_loader, val_dataloaders=train_loader)
