import wandb
from Data_Set  import  my_collate, Tensor, Val_Dataset, OT_Dataset
from networks.flowmatching.flow_final import SRM
from networks.flowmatching.encoder import SetTransformer
import torch
from torch.utils.data import DataLoader
import os
import pytorch_lightning as L
from flow_matching.solver import ODESolver
from utils import draw
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint, LearningRateMonitor
from tqdm import tqdm
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from networks.flowmatching.decoder import Decoder

device = "cuda" if torch.cuda.is_available() else "cpu"
experiment_name = 'MLP Set Transformer Mask 1024 Hidden Dim Sinkhorn OT'
format_path = 'format.svg'
train_path = '10k_512.pt'
val_path = '10k_512.pt'
# noise_path = 'result_noise_hungarian.pt'
noise_path = 'masked_noise_hungarian_10k_512.pt'
learning_rate = 2e-4
size = 1000
BATCH_SIZE = 64
hidden_size = 1024
samples = 1000
steps = 200
sample_steps = 30
beta_schedule = 'linear'
dim_in = 6
gpu_num = 1

train_dataset = OT_Dataset(train_path,noise_path)
val_dataset = OT_Dataset(val_path,noise_path)


torch.set_float32_matmul_precision("medium")


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

decoder = Decoder(
        dim_input=dim_in,
        num_outputs=1,
        num_inputs=size,
        dim_output=6,
        num_inds=32,
        dim_hidden=256,
        num_heads=16,
        emb_size=64,
        ln=True)

ckpt_path = "epoch=149-global_step=0.ckpt"
sample_steps = list(range(sample_steps))
srm = SRM(encoder,decoder, experiment_name, samples, sample_steps, format_path, size,dim_in, learning_rate, weight_mse=1.0)
srm.load_state_dict(torch.load(ckpt_path, weights_only=False)["state_dict"])

srm.eval()
srm.to(device)
solver = ODESolver(srm.decoder)
num_of_strokes = 512
x_0 = torch.randn((1,512,6), device="cuda")

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False)
#Strokes_mask = torch.ones((1,num_of_strokes,1), device="cuda")
#mask_zeros = torch.zeros((1,512-num_of_strokes,1), device="cuda")
#Strokes_mask = torch.cat((Strokes_mask, mask_zeros), dim=1)
#x_0 = x_0 * Strokes_mask
#time_grid = torch.linspace(0.0,1.0,1000,device="cuda") 

for i, (Strokes,x_0) in enumerate(tqdm(train_loader)):
    
    x_0 = torch.randn((1,512,6), device="cuda")
    time_grid = torch.tensor([0.0,0.96,0.97,0.98,0.99,1.0],device="cuda")
    mask = Strokes[1].float()
    mask = mask.unsqueeze(-1)
    Stroke = Strokes[0]
    Stroke = Stroke * mask
    print(f"Stroke.shape: {Stroke.shape}")
    Stroke = Stroke.to(device)
    attention_latent = srm.encoder(Stroke)
    torch.save(attention_latent, f'/scratch/ks02450/attention_latent_{i}.pt')




	
