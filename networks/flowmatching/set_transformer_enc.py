import torch
import torch.nn as nn
from networks.model.modules import ISAB, PMA, SAB, MAB
from networks.model.positional_embeddings import PositionalEmbedding

class l_Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(size)

    def forward(self, x: torch.Tensor):
        return x + self.act(self.norm(self.ff(x)))

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs,num_inputs, dim_output,
                 num_inds, dim_hidden, num_heads, emb_size=64,ln=False):
        super(SetTransformer, self).__init__()
        
        # self.enc = nn.Sequential(nn.Linear(dim_input, dim_output))
        
        # self.dec_enc = nn.Sequential(
        #     ISAB(dim_output, dim_hidden, num_heads, num_inds, ln=ln),
        #     ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        # )

        # self.dec_dec = nn.Sequential(
        #     PMA(dim_hidden, num_heads, num_inputs, ln=ln),
        #     SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        #     SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        # )

        # self.final_out = nn.Linear(dim_hidden, dim_input)
        self.time_mlp = PositionalEmbedding(emb_size, "sinusoidal")
        self.num_embeddings = 6
        for i in range(self.num_embeddings):
            setattr(self, f"input_mlp{i+1}", PositionalEmbedding(emb_size, "sinusoidal", scale=25.0))
        self.input_mlp = nn.ModuleList([getattr(self, f"input_mlp{i+1}") for i in range(self.num_embeddings)])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device in set transformer: {self.device}")

        concat_size = (dim_input + 1) * emb_size
        
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),  # Not Present in the First Run titled "Flow Set Transformer Encoder POC"
            #SAB(dim_hidden, dim_output, 2, num_inds, ln=ln),
        )


        # self.conv = nn.Sequential(
        #     nn.Conv1d(dim_hidden*2, dim_hidden*2, kernel_size=3, padding=1),
        #     nn.GELU(),
        #     nn.Conv1d(dim_hidden*2, dim_hidden*2, kernel_size=3, padding=1),
        #     nn.GELU(),
        #     nn.Conv1d(dim_hidden*2, dim_hidden*2, kernel_size=3, padding=1),
        #     nn.GELU(),
        #     nn.Conv1d(dim_hidden*2, dim_hidden*2, kernel_size=3, padding=1),
        # )

        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        )

        # self.linear_mu = nn.Linear(dim_hidden, dim_hidden)
        # self.linear_sigma = nn.Linear(dim_hidden, dim_hidden)
        # self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.to(self.device)
        # self.N.scale = self.N.scale.to(self.device)

        dim_mlp_hidden = 4096
        layers = []
        layers.append(nn.Linear(concat_size + dim_hidden, dim_mlp_hidden))
        for i in range(6):
            layers.append(l_Block(dim_mlp_hidden))
        layers.append(nn.Linear(dim_mlp_hidden, dim_output))
        self.mlp = nn.Sequential(*layers)

        # self.mask_mlp = nn.Sequential(
        #     nn.Linear(concat_size + dim_hidden, dim_hidden),
        #     nn.GELU(),
        #     nn.Linear(dim_hidden, dim_hidden),
        #     nn.GELU(),
        #     nn.Linear(dim_hidden, dim_hidden),
        #     nn.GELU(),
        #     nn.Linear(dim_hidden, 1),
        #     nn.Sigmoid(),
        # )

    def forward(self, x, t):
        #print(f"X.shape in set transformer: {X.shape}")
        #print(f"X_mask.shape in set transformer: {X_mask.shape}")
        # X_mask = X_mask.unsqueeze(-1)
        # X = X * X_mask
        #print(f"X.shape in set transformer: {X.shape}")
        encoded = self.enc(x)
        #c = self.dec(encoded)
        # b,n,d = encoded.shape
        #encoded = encoded.reshape(b*n, d)
        # encoded = self.conv(encoded)
        #encoded = encoded.reshape(b, n, d)
        #encoded = encoded.permute(0,2,1)
        x_emb = []
        # # print(f"x.shape: {x.shape}")
       
        if t.dim() == 0:
            t = torch.full((x.shape[0],1), t, device=self.device)
            # t_emb = t_emb.unsqueeze(-1)
        #else:
            #t_emb = t.unsqueeze(-1)
        for i in range(self.num_embeddings):
            input_emb = x[:,:, i]
            #input_emb = input_emb * x_mask
            x_emb.append(self.input_mlp[i](input_emb))
        t_emb = self.time_mlp(t)
        t_emb = t_emb.repeat(1,x.shape[1],1)
        #c = c.repeat(1,x.shape[1],1)
        x_emb = torch.cat((*x_emb, t_emb), dim=-1)
        #x_emb = self.mlp_in(x_emb)
        

        # c = self.dec(encoded)
        # mu = self.linear_mu(c)
        # sigma = self.linear_sigma(c)
        # sigma = torch.exp(sigma)
        # z = mu + sigma*self.N.sample(mu.shape).to(self.device)
        # z = z.repeat(1,x.shape[1],1)
        # print(f"z.shape: {z.shape}")
        # print(f"x_emb.shape: {x_emb.shape}")
        encoded = torch.cat((x_emb,encoded), dim=-1)
        #mask = self.mask_mlp(encoded)
        #encoded = encoded * mask
        out = self.mlp(encoded)
        #out = out * mask
        return out
    
    def compute_mask(self, x, t):
        #print(f"X.shape in set transformer: {X.shape}")
        #print(f"X_mask.shape in set transformer: {X_mask.shape}")
        # X_mask = X_mask.unsqueeze(-1)
        # X = X * X_mask
        #print(f"X.shape in set transformer: {X.shape}")
        encoded = self.enc(x)
        #c = self.dec(encoded)
        # b,n,d = encoded.shape
        #encoded = encoded.reshape(b*n, d)
        # encoded = self.conv(encoded)
        #encoded = encoded.reshape(b, n, d)
        #encoded = encoded.permute(0,2,1)
        x_emb = []
        # # print(f"x.shape: {x.shape}")
    
        if t.dim() == 0:
            t = torch.full((x.shape[0],1), t, device=self.device)
            # t_emb = t_emb.unsqueeze(-1)
        #else:
            #t_emb = t.unsqueeze(-1)
        for i in range(self.num_embeddings):
            input_emb = x[:,:, i]
            #input_emb = input_emb * x_mask
            x_emb.append(self.input_mlp[i](input_emb))
        t_emb = self.time_mlp(t)
        t_emb = t_emb.repeat(1,x.shape[1],1)
        #c = c.repeat(1,x.shape[1],1)
        x_emb = torch.cat((*x_emb, t_emb), dim=-1)
        #x_emb = self.mlp_in(x_emb)
        

        c = self.dec(encoded)
        mu = self.linear_mu(c)
        sigma = self.linear_sigma(c)
        sigma = torch.exp(sigma)
        z = mu + sigma*self.N.sample(mu.shape).to(self.device)
        z = z.repeat(1,x.shape[1],1)
        # print(f"z.shape: {z.shape}")
        # print(f"x_emb.shape: {x_emb.shape}")
        encoded = torch.cat((x_emb,encoded), dim=-1)
        mask = self.mask_mlp(encoded)
        return mask