import wandb
from Data_Set  import  my_collate, Tensor, Val_Dataset, OT_Dataset, OT_Dataset_Val
from networks.flowmatching.flow_Set_Transformer import SRM
from networks.flowmatching.set_transformer_enc import SetTransformer
import torch
from torch.utils.data import DataLoader
import os
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from diffusers import DDIMScheduler, DDPMScheduler
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint, LearningRateMonitor

device = "cuda" if torch.cuda.is_available() else "cpu"
experiment_name = 'MLP Hungarian Preprocessed OT Latent'
format_path = 'format.svg'
train_path = '10k_512.pt'
val_path = '10k_512.pt'
noise_path = 'result_noise_hungarian.pt'

learning_rate = 2e-4
size = 512
BATCH_SIZE = 128
hidden_size = 1024
samples = 512
steps = 200
sample_steps = 30
beta_schedule = 'linear'
dim_in = 6
gpu_num = 1

#Add WB key here
wand_b_key = '117905e69dff43b1635103618ba74a5593104105'
wandb.login(key=wand_b_key)
wandb_logger = WandbLogger(name=experiment_name,project='Your Stroke Cloud',save_dir='/scratch/ks02450')
trainer = Trainer(logger=wandb_logger)
train_set = OT_Dataset(train_path, noise_path)
val_set = OT_Dataset_Val(val_path, noise_path)
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, pin_memory=False)
val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False, pin_memory=False)
torch.set_float32_matmul_precision("medium")
lr_monitor = LearningRateMonitor(logging_interval='epoch')

checkpoint_callback = ModelCheckpoint(
    dirpath="/scratch/ks02450/Models/{}/".format(experiment_name),
    filename="{epoch:02d}-{global_step}",
    save_last=False,
    every_n_epochs=100,
    save_on_train_epoch_end=True,
)

encoder = SetTransformer(
        dim_input=dim_in,
        num_outputs=1,
        num_inputs=size,
        dim_output=6,
        num_inds=32,
        dim_hidden=256,
        num_heads=16,
        emb_size=64,
        ln=True)

if not os.path.exists("/scratch/ks02450/Results/{}".format(experiment_name)):
        os.makedirs("/scratch/ks02450/Results/{}".format(experiment_name))

if not os.path.exists("/scratch/ks02450/Models/{}".format(experiment_name)):
        os.makedirs("/scratch/ks02450/Models/{}".format(experiment_name))

sample_steps = list(range(sample_steps))
srm = SRM(encoder, experiment_name, samples, sample_steps, format_path, size,dim_in, learning_rate, weight_mse=1.0)
trainer = L.Trainer(accelerator='gpu', devices=gpu_num, strategy='auto' ,logger=wandb_logger, max_epochs=-1,
                    check_val_every_n_epoch=10, enable_progress_bar=True, profiler="simple",
                    callbacks=[StochasticWeightAveraging(swa_lrs=learning_rate),checkpoint_callback, lr_monitor], benchmark=True)
# trainer.fit(model=srm, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(model=srm, dataloaders=val_loader, ckpt_path="epoch=499-global_step=0.ckpt")