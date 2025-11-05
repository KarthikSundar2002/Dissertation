
import wandb
from Data_Set  import Tensor
import torch
from torch.utils.data import DataLoader
import os
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from diffusers import DDIMScheduler, DDPMScheduler
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint, LearningRateMonitor
from networks.model.ae import ae
from networks.flowmatching.flow_final import SRM as srm
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps"
experiment_name = 'AE-Train-run'
format_path = 'format.svg'
train_path = '10k_512.pt'

learning_rate = 1e-4
size = 512
BATCH_SIZE = 512
hidden_size = 64
samples = 1000
steps = 4000
sample_steps = 25
beta_schedule = 'scaled_linear'
wand_b_key = '117905e69dff43b1635103618ba74a5593104105'
gpu_num = 1
wandb.login(key=wand_b_key)
wandb_logger = WandbLogger(name=experiment_name,project='Your Stroke Cloud',save_dir="/scratch/kas02450/wandb")
trainer = Trainer(logger=wandb_logger)
train_set = Tensor(train_path)
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
torch.set_float32_matmul_precision("medium")

checkpoint_callback = ModelCheckpoint(
    dirpath="/scratch/ks02450/Models/{}/".format(experiment_name),
    filename="{epoch:02d}-{global_step}",
    save_last=True,
    every_n_epochs=100,
    save_on_train_epoch_end=True,
)

srm = srm.load_from_checkpoint("epoch=149-global_step=0.ckpt")





lr_monitor = LearningRateMonitor(logging_interval='epoch')
ae = ae(experiment_name=experiment_name, learning_rate=learning_rate, srm=srm)
if not os.path.exists("/scratch/ks02450/Results/{}".format(experiment_name)):
        os.makedirs("/scratch/ks02450/Results/{}".format(experiment_name))

if not os.path.exists("/scratch/ks02450/Models/{}".format(experiment_name)):
        os.makedirs("/scratch/ks02450/Models/{}".format(experiment_name))

trainer = L.Trainer(accelerator='gpu', devices=gpu_num, strategy='auto' ,logger=wandb_logger, max_epochs=10000,
                    check_val_every_n_epoch=200, enable_progress_bar=True, profiler="simple",
                    callbacks=[StochasticWeightAveraging(swa_lrs=learning_rate),checkpoint_callback, lr_monitor], benchmark=True)
trainer.fit(model=ae, train_dataloaders=train_loader, val_dataloaders=train_loader)
