import torch
import torch.nn as nn
from networks.model.modules import ISAB, PMA, SAB

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs,num_inputs, dim_output,
                 num_inds, dim_hidden, num_heads, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )
        # self.enc = nn.Sequential(nn.Linear(dim_input, dim_output))
        
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        )

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
        
        self.linear_mu = nn.Linear(dim_hidden, dim_output)
        self.linear_sigma = nn.Linear(dim_hidden, dim_output)
        
        self.N = torch.distributions.Normal(0, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device in set transformer: {self.device}")
        self.N.loc = self.N.loc.to(self.device)
        self.N.scale = self.N.scale.to(self.device)

    def forward(self, X):
        print(f"X.shape in set transformer: {X.shape}")
        encoded = self.enc(X)
        y = self.dec(encoded)
        print(f"y.shape in set transformer: {y.shape}")
        mu = self.linear_mu(y)  
        print(f"mu.shape in set transformer: {mu.shape}")
        sigma = torch.exp(self.linear_sigma(y))
        print(f"sigma.shape in set transformer: {sigma.shape}")
        z = mu + sigma * self.N.sample(mu.shape).to(mu.device)
        print(f"z.shape in set transformer: {z.shape}")
        # z_dec = self.dec_enc(z)
        # z_dec = self.dec_dec(z_dec)
        # z_out = self.final_out(z_dec)
        return encoded, z, mu, sigma
        #return encoded