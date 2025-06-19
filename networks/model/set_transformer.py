import torch
import torch.nn as nn
from networks.model.modules import ISAB, PMA, SAB

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
                 num_inds, dim_hidden, num_heads, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )
        
        # self.dec = nn.Sequential(
        #     PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        #     SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        #     SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        # )
        
        # self.linear_mu = nn.Linear(dim_hidden, dim_output)
        # self.linear_sigma = nn.Linear(dim_hidden, dim_output)
        
        # self.N = torch.distributions.Normal(0, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.N.loc = self.N.loc.to(self.device)
        # self.N.scale = self.N.scale.to(self.device)

    def forward(self, X):
        encoded = self.enc(X)
        print(f"encoded.shape in set transformer: {encoded.shape}")
        # y = self.dec(encoded)
        # print(f"y.shape in set transformer: {y.shape}")
        # mu = self.linear_mu(y)  
        # print(f"mu.shape in set transformer: {mu.shape}")
        # sigma = torch.exp(self.linear_sigma(y))
        # print(f"sigma.shape in set transformer: {sigma.shape}")
        # z = mu + sigma * self.N.sample(mu.shape).to(mu.device)
        # print(f"z.shape in set transformer: {z.shape}")
        
        # return encoded, z, mu, sigma
        return encoded