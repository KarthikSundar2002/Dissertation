from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
class Tensor(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data
def my_collate(batch):
    # B x Set(S) x Strokes(s)
    # print(f"Batch is a list of length {len(batch)}")
    # print(f"Batch[0] is a tensor of shape {batch[0].shape}")
    # print(f"Batch[1] is a tensor of shape {batch[1].shape}")
    # Takes a batch of length Batch Size where each entry is an image and returns a Set and Strokes tensors... 
    # Set is a tensor of shape (Batch Size, Max Set Size, 6 - Number of Parameters to describe a stroke (3 control points - So 3 * 2 parameters))
    # In the Modified version, we pad the Set tensor to a predefined maximum number of strokes
    # Strokes is a tensor of shape (Batch Size, Sampled Strokes Size, 6 - Number of Parameters to describe a stroke (3 control points - So 3 * 2 parameters))

    number_of_strokes = 512
    batch[0] = torch.nn.functional.pad(batch[0], (0, 0, number_of_strokes - batch[0].shape[0], 0))
    Set = pad_sequence(batch, batch_first=True)
    #Strokes
    R = []

    samp = 512
    for idx, val in enumerate(batch):
        if len(batch[idx]) <= samp:
            Randomly_sampled_strokes = random.choices(batch[idx], k=samp)
        else:
            indices = torch.randperm(len(batch[idx]))[:samp]
            Randomly_sampled_strokes = batch[idx][indices]
            Randomly_sampled_strokes = [t for t in Randomly_sampled_strokes]

        R.append(torch.stack(Randomly_sampled_strokes, dim=0))

    Strokes = torch.stack(R, dim=0)
    return Set, Strokes

