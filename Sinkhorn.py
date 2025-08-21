import torch
from geomloss import SamplesLoss
from torch.utils.data import DataLoader
from Data_Set import OT_Dataset
from geomloss import SamplesLoss
from tqdm import tqdm

# Create two distributions with different numbers of samples
train_set = OT_Dataset('10k_512.pt')
BATCH_SIZE = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, pin_memory=True)
loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
print(train_set.data[0][0].shape)
result_noise = torch.nn.Parameter(torch.randn((10093,512,6), device=device))
optimizer = torch.optim.Adam([result_noise], lr=0.1)

for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Batches"):
    Strokes, Strokes_mask = batch
    Strokes = Strokes.to(device)
    noise = result_noise[idx:idx+Strokes.shape[0]]
    for i in range(100):
        loss = loss_fn(Strokes, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # if i % 10 == 0:
        #     print(f"Loss: {loss.item()}")

torch.save(result_noise, 'result_noise.pt')
   
    
# data = torch.load('10k_512.pt')
# print(len(data))

