import torch
import torch.nn as nn
from networks.model.positional_embeddings import PositionalEmbedding
import math

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
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal", output_size: int = 6):
        super(MLP, self).__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)

        concat_size = emb_size + 262

        layers = [nn.Linear(concat_size, hidden_size),nn.LayerNorm(hidden_size), nn.GELU()]
        attention_size = 256
        for _ in range(hidden_layers):
            layers.append(l_Block(hidden_size))
        layers.append(nn.Linear(hidden_size,attention_size))
        
        self.joint_mlp = nn.Sequential(*layers)
        self.query = nn.Linear(attention_size,attention_size)
        self.key = nn.Linear(attention_size,attention_size)
        self.value = nn.Linear(attention_size,attention_size)
        self.softmax = nn.Softmax(dim=-1)

        self.output_layer = nn.Sequential(nn.Linear(attention_size,256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256,64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64,output_size))  

    def forward(self, x, t):
        # x shape: [Batch, 512, 262]
        # t shape: [Batch]
        # print(f"x shape {x.shape}")
        # print(f"t shape {t.shape}")
        t_emb = self.time_mlp(t) #[Batch, 64]
        # print(f"t_emb shape {t_emb.shape}")
        t_emb = t_emb.unsqueeze(1).repeat(1, x.shape[1], 1) #[Batch, 512, 64]
        # print(f"t_emb shape {t_emb.shape}")
        x = torch.cat((x, t_emb), dim=-1) #[Batch, 512, 326]
        # print(f"x shape {x.shape}")
        x = self.joint_mlp(x) #[Batch, 512, 256]
        # print(f"x shape{x.shape}")
        q = self.query(x)
        # print(f'q shape{q.shape}')
        k = self.key(x)
        # print(f'k shape{k.shape}')
        v = self.value(x)
        # print(f'v shape{v.shape}')
        weight = torch.matmul(q.transpose(-1,-2),k)
        # print(f'weight shape{weight.shape}')
        qk = self.softmax(weight/q.shape[-1]**0.5)
        # print(f'qk shape{qk.shape}')
        x = torch.matmul(v,qk)
        # print(f'x shape{x.shape}')
        x = self.output_layer(x)
        return x
