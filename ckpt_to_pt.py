import torch

weights = torch.load('9599.ckpt',map_location=torch.device('cpu'), weights_only=False)["state_dict"]
torch.save(weights, 'model.pt')