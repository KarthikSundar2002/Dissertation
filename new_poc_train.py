import wandb
from Data_Set  import  my_collate, Tensor, Val_Dataset
from networks.flowmatching.simple_model import SRM as srm
import torch
from torch.utils.data import DataLoader
import os
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint, LearningRateMonitor

device = "cuda" if torch.cuda.is_available() else "cpu"
experiment_name = 'Simple POC Model-2048-Val-2'
format_path = 'format.svg'
train_path = '10k_512.pt'
val_path = '10k_512.pt'


learning_rate = 1e-4
size = 512
BATCH_SIZE = 256
num_of_strokes = 512
embed_dim = 64
dim_in = 6
num_layers = 4
dim_hidden = 2048
gpu_num = 1

#Add WB key here
wand_b_key = '117905e69dff43b1635103618ba74a5593104105'
wandb.login(key=wand_b_key)
wandb_logger = WandbLogger(name=experiment_name,project='Your Stroke Cloud',save_dir='/scratch/ks02450')
trainer = Trainer(logger=wandb_logger)
train_set = Tensor(train_path)
val_set = Val_Dataset(val_path)
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False, pin_memory=True)
torch.set_float32_matmul_precision("medium")
lr_monitor = LearningRateMonitor(logging_interval='epoch')

checkpoint_callback = ModelCheckpoint(
    dirpath="/scratch/ks02450/Models/{}/".format(experiment_name),
    filename="{epoch:02d}-{global_step}",
    save_last=False,
    every_n_epochs=500,
    save_on_train_epoch_end=True,
)



if not os.path.exists("/scratch/ks02450/Results/{}".format(experiment_name)):
        os.makedirs("/scratch/ks02450/Results/{}".format(experiment_name))

if not os.path.exists("/scratch/ks02450/Models/{}".format(experiment_name)):
        os.makedirs("/scratch/ks02450/Models/{}".format(experiment_name))


srm = srm(experiment_name, num_of_strokes, embed_dim, dim_in, learning_rate, num_layers, dim_hidden, format_path, size)

trainer = L.Trainer(accelerator='gpu', devices=gpu_num, strategy='auto' ,logger=wandb_logger, max_epochs=-1,
                    check_val_every_n_epoch=2, enable_progress_bar=True, profiler="simple",
                    callbacks=[StochasticWeightAveraging(swa_lrs=learning_rate),checkpoint_callback, lr_monitor], benchmark=True)
trainer.fit(model=srm, train_dataloaders=train_loader, val_dataloaders=val_loader)
	
