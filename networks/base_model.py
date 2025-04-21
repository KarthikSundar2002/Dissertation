import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.isTrain = opt.isTrain
        self.lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    def save_networks(self, epoch):
        save_filename = '{}_net.pth'.format(epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.state_dict(), save_path)
        print(f'Saving model {save_path}')

    def load_networks(self, epoch):
        load_filename = '{}_net.pth'.format(epoch)
        load_path = os.path.join(self.save_dir, load_filename)
        print(f'Loading model {load_path}')
        self.load_state_dict(torch.load(load_path))
        
