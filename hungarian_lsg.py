import torch

from torch.utils.data import DataLoader
from Data_Set import OT_Dataset, Tensor, Tensor_LSG
from networks.model.ae import ae
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
# Create two distributions with different numbers of samples
train_set = Tensor_LSG("/scratch/ks02450/ SRM Latent 4096 Dim MLP 10k 512 No Mask Test/")

BATCH_SIZE = 10093
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(len(train_set))
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=False, pin_memory=False)
print(train_set.data[0][0].shape)
result_noise = torch.randn((10093,512,6), device=device)
latent_result = torch.randn((10093,512,6), device=device)
# optimizer = torch.optim.Adam([result_noise], lr=0.1)
for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Batches"):
    latent = batch
    latent = latent.to(device)
    print(f"latent.shape: {latent.shape}")
    noise = result_noise[idx:idx+latent.shape[0]]
    cost_matrix = torch.cdist(noise, latent, p=1)
    for i in tqdm(range(cost_matrix.shape[0]), total=cost_matrix.shape[0], desc="Rows"):
        row_ind, col_ind = linear_sum_assignment(cost_matrix[i].cpu().numpy(), maximize=False)
        result_noise[idx+i] = result_noise[idx+i][row_ind]
        latent_result[idx+i] = latent_result[idx+i][col_ind]
        # if i % 10 == 0:
        #     print(f"Loss: {loss.item()}")

torch.save(latent_result, '/scratch/ks02450/latent_hungarian.pt')
torch.save(result_noise, '/scratch/ks02450/latent_noise_hungarian.pt')
   
    