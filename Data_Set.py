from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
import pickle
import os

class LSG_Dataset(Dataset):
    def __init__(self, path, noise_path):
        data_1 = torch.load(path + '/attention_latent_0.pt')
        data_2 = torch.load(path + '/attention_latent_1.pt')
        data_3 = torch.load(path + '/attention_latent_2.pt')
        data_4 = torch.load(path + '/attention_latent_3.pt')
        data_5 = torch.load(path + '/attention_latent_4.pt')
        self.data = torch.cat((data_1, data_2, data_3, data_4, data_5), dim=0)
        self.noise = torch.load(noise_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        noise = self.noise[idx]
        return data, noise

class Tensor(Dataset):
    def __init__(self, path):
        self.data = torch.load(path) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

class Tensor_LSG(Dataset):
    def __init__(self, path):
        self.data = torch.tensor([])
        for file in os.listdir(path):
            if file.endswith('.pt'):
                data = torch.load(os.path.join(path, file))
                data = data.to("cpu")
                self.data = torch.cat((self.data, data), dim=0)
        self.data = self.data.view(-1, 6)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data
    

class One_Image(Dataset):
    def __init__(self, path, noise_path):
        self.data = torch.load(path)
        self.data = self.data[13:14]
        self.noise = torch.load(noise_path)
        self.noise = self.noise[13:14]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        noise = self.noise[idx]
        return data, noise
        
class OT_Dataset(Dataset):
    def __init__(self, path, noise_path):
        self.data = torch.load(path)
        #self.data = self.data[:1]
        self.noise = torch.load(noise_path)
        #self.noise = self.noise[:1]
        self.data = self.data.view(-1, 6)
        self.noise = self.noise.view(-1, 6)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        noise = self.noise[idx]
        return data, noise

class OT_Dataset_Val(Dataset):
    def __init__(self, path, noise_path):
        self.data = torch.load(path)
        self.data = self.data[:1]
        self.noise = torch.load(noise_path)
        self.noise = self.noise[:1]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        noise = self.noise[idx]
        return data, noise

class Val_Dataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)
        self.data = self.data[:1]
    
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

