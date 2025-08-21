import torch
import torch.nn as nn
from networks.model.positional_embeddings import PositionalEmbedding

class l_Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(size)

    def forward(self, x: torch.Tensor):
        return x + self.act(self.norm(self.ff(x)))

class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal",):
        super(MLP, self).__init__()
        self.num_embeddings = 6
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        for i in range(self.num_embeddings):
            setattr(self, f"input_mlp{i+1}", PositionalEmbedding(emb_size, input_emb, scale=25.0))
        self.input_mlp = nn.ModuleList([getattr(self, f"input_mlp{i+1}") for i in range(self.num_embeddings)])
        concat_size = (self.num_embeddings+1)*emb_size + 256
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(l_Block(hidden_size))
        layers.append(nn.Linear(hidden_size, self.num_embeddings))
        self.joint_mlp = nn.Sequential(*layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_mlp = nn.Sequential(nn.Linear(concat_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, x, t, y, x_mask):

        t = t.to(self.device)
        print(f"x.shape in MLP: {x.shape}")
        print(f"t.shape in MLP: {t.shape}")
        print(f"y.shape in MLP: {y.shape}")
        print(f"x_mask.shape in MLP: {x_mask.shape}")
        x_emb = []
        x_mask = x_mask.squeeze(-1)
        for i in range(self.num_embeddings):
            input_emb = x[:,:, i]
            print(f"input_emb.shape in MLP: {input_emb.shape}")
            print(f"x_mask.shape in MLP: {x_mask.shape}")
            input_emb = input_emb * x_mask
            print(f"input_emb.shape in MLP: {input_emb.shape}")
            x_emb.append(self.input_mlp[i](input_emb))
        t_emb = self.time_mlp(t)
        t_emb = t_emb.repeat(x_emb[0].shape[0], 1, 1)
        y = y.repeat(1,x_emb[0].shape[1],1 )
        print(f"x_emb[0].shape: {x_emb[0].shape}")
        print(f"t_emb.shape: {t_emb.shape}")
        print(f"y.shape: {y.shape}")
        x = torch.cat((*x_emb, t_emb, y), dim=-1)
        print(f"x.shape in MLP: {x.shape}")
        x = self.joint_mlp(x)

        return x
    
    def compute_mask(self, x, t, y, x_mask):
        t = t.to(self.device)
        x_emb = []
        for i in range(self.num_embeddings):
            input_emb = x[:,:, i]
            input_emb = input_emb * x_mask
            x_emb.append(self.input_mlp[i](input_emb))
        t_emb = self.time_mlp(t)
        t_emb = t_emb.repeat(x_emb[0].shape[0], 1, 1)
        y = y.repeat(1,x_emb[0].shape[1],1 )
        print(f"x_emb[0].shape: {x_emb[0].shape}")
        print(f"t_emb.shape: {t_emb.shape}")
        print(f"y.shape: {y.shape}")
        x = torch.cat((*x_emb, t_emb, y), dim=-1)
        mask = self.mask_mlp(x)
        return mask