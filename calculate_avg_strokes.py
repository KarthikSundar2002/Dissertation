import torch

# Load the validation data
print('Loading val.pt...')
val_data = torch.load('val.pt')

# Calculate the number of strokes for each sample
stroke_counts = []
num_samples = 0
for idx, tensor in enumerate(val_data):
    # tensor: [N, 6], N is the number of strokes
    n, d = tensor.shape
    stroke_counts.append(n)
    num_samples += n
    if idx % 1000 == 0:
        print(f'Processed {idx} samples...')

# Calculate average
avg_strokes = sum(stroke_counts) / len(stroke_counts)
min_strokes = min(stroke_counts)
max_strokes = max(stroke_counts)

print(f'\nStatistics for val.pt:')
print(f'Total samples: {len(stroke_counts)}')
print(f'Average strokes per sample: {avg_strokes:.2f}')
print(f'Minimum strokes: {min_strokes}')
print(f'Maximum strokes: {max_strokes}') 