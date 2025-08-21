import torch

# Load the data
print('Loading 18k_600.pt...')
data = torch.load('18k_600.pt')

# Count the number of samples
num_samples = len(data)
print(f'Number of SVGs in 18k_600.pt: {num_samples}')

# Also show some additional info about the data structure
if num_samples > 0:
    first_sample = data[0]
    if isinstance(first_sample, tuple):
        tensor, mask = first_sample
        print(f'Each sample contains:')
        print(f'  - Tensor shape: {tensor.shape}')
        print(f'  - Mask shape: {mask.shape}')
        print(f'  - Tensor dtype: {tensor.dtype}')
        print(f'  - Mask dtype: {mask.dtype}') 