import torch
import random

input_path = '10k.pt'
output_path = '10k_512.pt'
num_strokes = 512

# Load the data
print(f'Loading {input_path}...')
data = torch.load(input_path)

processed = []
for idx, tensor in enumerate(data):
    # tensor: [N, 6], N can vary
    n, d = tensor.shape
    if n < num_strokes:
        # Pad with zeros
        pad = torch.zeros((num_strokes - n, d), dtype=tensor.dtype, device=tensor.device)
        new_tensor = torch.cat([tensor, pad], dim=0)
        mask = torch.cat([
            torch.ones(n, dtype=torch.bool, device=tensor.device),
            torch.zeros(num_strokes - n, dtype=torch.bool, device=tensor.device)
        ], dim=0)
    elif n > num_strokes:
        # Randomly sample without replacement if possible
        indices = torch.randperm(n)[:num_strokes]
        new_tensor = tensor[indices]
        mask = torch.ones(num_strokes, dtype=torch.bool, device=tensor.device)
    else:
        new_tensor = tensor
        mask = torch.ones(num_strokes, dtype=torch.bool, device=tensor.device)
    processed.append((new_tensor, mask))
    if idx % 1000 == 0:
        print(f'Processed {idx} samples...')

# Save the processed data
print(f'Saving processed data to {output_path}...')
torch.save(processed, output_path)
print('Done.') 