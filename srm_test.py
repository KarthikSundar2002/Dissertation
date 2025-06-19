from Data_Set  import  my_collate, Tensor
from networks.model.models import  srm
from torch.utils.data import DataLoader
import os
import torch
from utils import sample, draw

from options.args import argument_parser

args = argument_parser()
args = args.parse_args()

print(type(args))
experiment_name = "Test"
if not experiment_name:
    experiment_name = 'SRM-Test-run'
torch.set_float32_matmul_precision('medium')
test_path = "10k.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(os.getcwd())
print(os.path.join(os.getcwd(), "/Models/SRM17149.ckpt"))
model = srm.load_from_checkpoint("Models/SRM.ckpt")
size = 512
dim_in = 6
samples = 1000
L = []
#To help train the lsg we create our latent data by sampling the data set multiple times.
reps = 1
if not os.path.exists(args.rootdir + "Samples/{}".format(experiment_name)):
        os.makedirs(args.rootdir + "Samples/{}".format(experiment_name))

test_set = Tensor(test_path)
loader = DataLoader(test_set, 1, shuffle=False, collate_fn= my_collate, pin_memory=True)
Encoder = model.encoder
Decoder = model.decoder

for i in range(reps):
    with torch.no_grad():
        for i, data in enumerate(loader):
            Latent = Encoder(data[0].to(device))[0]
            L.append(Latent)
            stroke = sample(samples, model.sample_steps, Decoder, model.noise_scheduler_sample, Latent, dim_in)
            filename = args.rootdir + 'Samples/{}/{}.svg'.format(experiment_name, i)
            draw(model.format, size, filename, stroke)


Latents = [item for sublist in L for item in sublist]
os.makedirs("Latent", exist_ok=True)
torch.save(Latents, os.path.join('Latent/{}.pt'.format(experiment_name)))
