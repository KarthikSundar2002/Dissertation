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

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)

        concat_size = emb_size + 256

        layers = [nn.Linear(concat_size, hidden_size),nn.LayerNorm(hidden_size), nn.GELU()]

        for _ in range(hidden_layers):
            layers.append(l_Block(hidden_size))
        layers.append(nn.Linear(hidden_size,256))
        self.output_layer = nn.Linear(256,256)
        self.joint_mlp = nn.Sequential(*layers)
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

    def forward(self, x, t):

        t_emb = self.time_mlp(t)
        t_emb = t_emb.repeat(x.shape[0], 1, 1)
        x = torch.cat((x, t_emb), dim=-1)
        x = self.joint_mlp(x)
        x, _ = self.attn(x, x, x)
        x = self.output_layer(x)
        return x
