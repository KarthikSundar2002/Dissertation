import torch
import torch.nn as nn
import lightning as L
from networks.model.modules import ISAB, PMA, SAB

class SetTransformer(L.LightningModule):
    def __init__(self, dim_input, num_outputs, dim_output,
                 num_inds, dim_hidden, num_heads, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )
        
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        )
        
        self.linear_mu = nn.Linear(dim_hidden, dim_output)
        self.linear_sigma = nn.Linear(dim_hidden, dim_output)
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(self.device)
        self.N.scale = self.N.scale.to(self.device)

    def forward(self, X):
        encoded = self.enc(X)

        y = self.dec(encoded)

        mu = self.linear_mu(y)
        sigma = torch.exp(self.linear_sigma(y))

        z = mu + sigma * self.N.sample(mu.shape).to(mu.device)
        
        return z, mu, sigma