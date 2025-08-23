import torch
from geomloss import SamplesLoss
from torch.utils.data import DataLoader
from Data_Set import OT_Dataset, Tensor
from geomloss import SamplesLoss
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
# Create two distributions with different numbers of samples
train_set = Tensor('18k_600.pt')
BATCH_SIZE = 18620
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(len(train_set))
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, pin_memory=True)
loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
print(train_set.data[0][0].shape)
result_noise = torch.randn((18620,600,6), device=device)
# optimizer = torch.optim.Adam([result_noise], lr=0.1)

for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Batches"):
    Strokes, Strokes_mask = batch
    Strokes = Strokes.to(device)
    Strokes_mask = Strokes_mask.to(device)
    Strokes_mask = Strokes_mask.unsqueeze(-1)
   # Strokes = Strokes.transpose(1,2)
    print(f"Strokes.shape: {Strokes.shape}")
    print(f"Strokes_mask.shape: {Strokes_mask.shape}")
    Strokes = Strokes * Strokes_mask
    noise = result_noise[idx:idx+Strokes.shape[0]]
    cost_matrix = torch.cdist(noise, Strokes, p=1)
    for i in tqdm(range(cost_matrix.shape[0]), total=cost_matrix.shape[0], desc="Rows"):
        row_ind, col_ind = linear_sum_assignment(cost_matrix[i].cpu().numpy(), maximize=False)
        result_noise[idx+i] = result_noise[idx+i][row_ind]
        Strokes[i] = Strokes[i][col_ind]
        Strokes_mask[i] = Strokes_mask[i][col_ind]
        # if i % 10 == 0:
        #     print(f"Loss: {loss.item()}")

torch.save(result_noise, 'masked_noise_hungarian_18k_600.pt')
   
    