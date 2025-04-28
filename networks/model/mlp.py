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
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp3 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp4 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp5 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp6 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        concat_size = (7 * emb_size) + 256
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(l_Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 6))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t, y):

        t = t.to(x.device)
        x1_emb = self.input_mlp1(x[:, :, 0])
        x2_emb = self.input_mlp2(x[:, :, 1])
        x3_emb = self.input_mlp3(x[:, :, 2])
        x4_emb = self.input_mlp4(x[:, :, 3])
        x5_emb = self.input_mlp5(x[:, :, 4])
        x6_emb = self.input_mlp6(x[:, :, 5])
        t_emb = self.time_mlp(t)
        t_emb = t_emb.repeat(x1_emb.shape[0], 1, 1)
        y = y.repeat(1,x1_emb.shape[1],1 )
        x = torch.cat((x1_emb, x2_emb, x3_emb, x4_emb, x5_emb, x6_emb, t_emb, y), dim=-1)
        x = self.joint_mlp(x)

        return x