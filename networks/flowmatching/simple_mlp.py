import torch
import torch.nn as nn
from networks.model.positional_embeddings import PositionalEmbedding
import torch.nn.functional as F

from networks.flowmatching.set_transformer_enc import SetTransformer

class SimpleMLP(nn.Module):
    def __init__(self, num_of_strokes, embed_dim, dim_in, lr, num_layers, dim_hidden, dim_output):
        super().__init__()
        self.num_of_strokes = num_of_strokes
        self.embed_dim = embed_dim
        self.dim_in = dim_in
        self.lr = lr
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.encoder = SetTransformer(dim_in, num_of_strokes, num_of_strokes, 1,
                 32, 1, 1, ln=False)
        self.time_embedding = PositionalEmbedding(embed_dim, "sinusoidal")
        layers = []
        concat_size = embed_dim + num_of_strokes
        layers.append(nn.Linear(concat_size, dim_hidden))
        layers.append(nn.GELU())
        for i in range(self.num_layers):
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(nn.GELU())
        layers.append(nn.Linear(dim_hidden, dim_output))
        self.model = nn.Sequential(*layers)

    def forward(self, x, t):
        #print(f"x.shape: {x.shape}")
        encoded = self.encoder(x)
        encoded = encoded.squeeze(-1)
        #print(f"encoded.shape: {encoded.shape}")
        t_emb = self.time_embedding(t)
        #print(f"t_emb.shape: {t_emb.shape}")
        x_emb = torch.cat((encoded, t_emb), dim=-1)
        out = self.model(x_emb)
        #print(f"out.shape: {out.shape}")
        out = out.reshape(x.shape)
        return out