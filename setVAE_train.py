import wandb
from Data_Set  import  my_collate, Tensor, Val_Dataset
from networks.SetVAE.SetVAE import SetVAE
import torch
from torch.utils.data import DataLoader
import os
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from diffusers import DDIMScheduler, DDPMScheduler
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
experiment_name = 'SetVAE train'
format_path = 'format.svg'
train_path = '10k_512_inv.pt'
val_path = '10k_val.pt'

BATCH_SIZE = 32
dim_in = 6 
learning_rate = 1e-5
torch.autograd.set_detect_anomaly(True)
#Add WB key here
wand_b_key = '117905e69dff43b1635103618ba74a5593104105'
wandb.login(key=wand_b_key)
wandb_logger = WandbLogger(name=experiment_name,project='Your Stroke Cloud',save_dir='/scratch/ks02450')
train_set = Tensor(train_path)
val_set = Val_Dataset(train_path)
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_set,1, shuffle=False, pin_memory=True)

torch.set_float32_matmul_precision("medium")

checkpoint_callback = ModelCheckpoint(
    dirpath="/scratch/ks02450/Models/{}/".format(experiment_name),
    filename="{epoch:02d}-{global_step}",
    save_last=True,
    every_n_epochs=10,
    save_on_train_epoch_end=True,
)

if not os.path.exists("/scratch/ks02450/Results/{}".format(experiment_name)):
        os.makedirs("/scratch/ks02450/Results/{}".format(experiment_name))

if not os.path.exists("/scratch/ks02450/Models/{}".format(experiment_name)):
        os.makedirs("/scratch/ks02450/Models/{}".format(experiment_name))

class ModelConfig:
    def __init__(self, input_dim, n_mixtures, hidden_dim, z_dim, max_outputs, z_scales, experiment_name, lr, kl_warmup_epochs, train_gmm, init_dim, num_heads, slot_att, i_net, i_net_layers, d_net, enc_in_layers, dec_in_layers, dec_out_layers, isab_inds, ln, dropout_p, activation, use_bn, residual, optimizer, beta, sample_size):
        self.input_dim = input_dim
        self.n_mixtures = n_mixtures
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.max_outputs = max_outputs
        self.z_scales = z_scales
        self.experiment_name = experiment_name
        self.lr = lr
        self.kl_warmup_epochs = kl_warmup_epochs
        self.train_gmm = train_gmm
        self.init_dim = init_dim
        self.num_heads = num_heads
        self.slot_att = slot_att
        self.i_net = i_net
        self.i_net_layers = i_net_layers
        self.d_net = d_net
        self.enc_in_layers = enc_in_layers
        self.dec_in_layers = dec_in_layers
        self.dec_out_layers = dec_out_layers
        self.isab_inds = isab_inds
        self.ln = ln
        self.dropout_p = dropout_p
        self.activation = activation
        self.use_bn = use_bn
        self.residual = residual
        self.optimizer = optimizer
        self.beta = beta
        self.sample_size = sample_size
       
model_args = ModelConfig(
    input_dim=dim_in,
    n_mixtures=1,
    hidden_dim=128,
    z_dim=32,
    max_outputs=512,
    z_scales=[128, 64, 32, 32],
    kl_warmup_epochs=100,
    train_gmm=True,
    init_dim=64,
    num_heads=4,
    slot_att=True,
    i_net='set_transformer',
    i_net_layers=2,
    d_net='set_transformer',
    enc_in_layers=2,
    dec_in_layers=2,
    dec_out_layers=2,
    isab_inds=16,
    ln=True,
    dropout_p=0.,
    activation='relu',
    use_bn=False,
    residual=False,
    optimizer='adam',
    lr=learning_rate,
    beta=0.0,
    sample_size=512,
    experiment_name=experiment_name
)

model = SetVAE(model_args)

trainer = L.Trainer(accelerator='gpu', devices=1, strategy='auto' ,logger=wandb_logger, max_epochs=-1,
                    check_val_every_n_epoch=10, enable_progress_bar=True, profiler="simple",
                    callbacks=[StochasticWeightAveraging(swa_lrs=learning_rate),checkpoint_callback ], benchmark=True, gradient_clip_val=0.5, gradient_clip_algorithm='norm')
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
	
