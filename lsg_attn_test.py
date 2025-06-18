from networks.model.models import srm
from networks.model.lsg import LSG as lsg
from networks.model.mlp import MLP as L_MLP
import os
import torch
from utils import sample, draw, l_sample


torch.set_float32_matmul_precision('medium')
experiment_name = 'Your name here'
SRM  = srm.load_from_checkpoint("/users/karthiksundarsenthilkumar/SRM17149.ckpt", map_location='cpu') 
LSG  = lsg.load_from_checkpoint("/users/karthiksundarsenthilkumar/epoch=999999-global_step=0.ckpt", map_location='cpu')    
dim_in = 6
samples = 1000
size = 512
#Number of sampling steps for the LSG
steps = list(range(50))

if not os.path.exists("Samples/{}".format(experiment_name)):
        os.makedirs("Samples/{}".format(experiment_name))

with torch.no_grad():
    for i in range(1000):
        #LSG
        Latent = l_sample(steps, LSG.model, LSG.noise_scheduler_sample)
        #SRM
        stroke = sample(samples, SRM.sample_steps, SRM.decoder, SRM.noise_scheduler_sample, Latent, dim_in)
        #Render
        filename = 'Samples/{}/{}.svg'.format(experiment_name, i)
        draw(SRM.format, size, filename, stroke)
