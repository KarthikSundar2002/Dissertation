import wandb
from Data_Set  import  my_collate, Tensor, Val_Dataset, OT_Dataset
from networks.flowmatching.flow_Set_Transformer import SRM
from networks.flowmatching.set_transformer_enc import SetTransformer
import torch
from torch.utils.data import DataLoader
import os
import pytorch_lightning as L
from flow_matching.solver import ODESolver
from utils import draw
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint, LearningRateMonitor
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
experiment_name = 'MLP Set Transformer Mask 1024 Hidden Dim Sinkhorn OT'
format_path = 'format.svg'
train_path = '18k_600.pt'
val_path = '10k_512.pt'
noise_path = 'result_noise_hungarian_18k_600.pt'

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

ckpt_path = "/scratch/ks02450/Models/MLP Hungarian Preprocessed OT Set Latent 18k 600 No Mask/epoch=10099-global_step=0.ckpt"
sample_steps = list(range(sample_steps))
srm = SRM(encoder, experiment_name, samples, sample_steps, format_path, size,dim_in, learning_rate, weight_mse=1.0)
srm.load_state_dict(torch.load(ckpt_path, weights_only=False)["state_dict"])
# weights = torch.load(ckpt_path, weights_only=False)['state_dict']
# weights = {k.replace("encoder.", ""): v for k, v in weights.items() if k.startswith("encoder.")}
# encoder.load_state_dict(weights)
srm.eval()
srm.to(device)
solver = ODESolver(srm.encoder)
num_of_strokes = 600
x_0 = torch.randn((1,600,6), device="cuda")
#Strokes_mask = torch.ones((1,num_of_strokes,1), device="cuda")
#mask_zeros = torch.zeros((1,512-num_of_strokes,1), device="cuda")
#Strokes_mask = torch.cat((Strokes_mask, mask_zeros), dim=1)
#x_0 = x_0 * Strokes_mask
#time_grid = torch.linspace(0.0,1.0,1000,device="cuda") 

for i, (Strokes,x_0) in enumerate(tqdm(train_dataset)):
#     print(f"Strokes.shape: {Strokes[1].shape}")
#     print(f"x_0.shape: {x_0.shape}")
    mask = Strokes[1]
    mask = mask.unsqueeze(-1)
    mask = mask.to(device)
    x_0 = x_0.to(device)
    x_0 = x_0.unsqueeze(0)
    #x_0 = x_0 * mask
    time_grid = torch.tensor([0.0,1.0],device="cuda")
    Output = solver.sample(x_0,0.001,"euler",1e-5,1e-5,time_grid,False,False) 
    draw(format_path, size, f'output{i}.svg', Output)
    break
#     draw(format_path, size, f'output_Strokes.svg', Strokes[0].unsqueeze(0))

# time_grid = torch.tensor([0.0,1.0],device="cuda")
# Output = solver.sample(x_0,0.001,"euler",1e-5,1e-5,time_grid,False,False) 
# t = torch.tensor([1.0],device="cuda")
#for i in range(len(Output)):
    #mask = srm.encoder.compute_mask(Output[i],t)
    #Output[i] = Output[i] * mask
    #draw(format_path, size, f'output_{i}.svg', Output[i])

# mask = srm.encoder.compute_mask(Output,t)
# Output = Output * mask
# draw(format_path, size, f'output.svg', Output)




	
