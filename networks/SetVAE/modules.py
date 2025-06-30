import torch
import torch.nn as nn
from .layers import AttentiveBlock, ResidualAttention, ISAB

class DeterministicNetwork(nn.Module):
    def __init__(self, net_type, num_inds, dim, dim_out, n_hidden, num_heads=0, ln=False, dropout_p=0.,
                 activation='relu', use_bn=False, residual=False):
        super().__init__()
        self.num_inds = num_inds
        self.net_type = net_type
        if self.net_type == 'elem_mlp':
            self.net = ElementwiseMLP(dim, dim, dim_out, n_hidden, activation, residual, use_bn, use_masked_bn=True, dropout_p=dropout_p)
        elif self.net_type == 'set_transformer':
            self.net = nn.ModuleList()
            for i in range(n_hidden):
                self.net.append(ISAB(dim, dim_out, num_heads, num_inds, ln=ln, dropout_p=dropout_p))

    def forward(self, x, x_mask):
        if self.net_type == 'elem_mlp':
            return self.net(x, x_mask)
        elif self.net_type == 'set_transformer':
            for layer in self.net:
                x = layer(x, x_mask)
            return x


class InducedNetwork(nn.Module):
    def __init__(self, num_inds, dim_in, dim_hidden, dim_out, n_hidden, num_heads=0, ln=False, dropout_p=0.):
        super().__init__()
        self.num_inds = num_inds
      
        self.input = nn.Linear(dim_in, dim_hidden)
        self.net = nn.ModuleList()
        for i in range(n_hidden):
            self.net.append(ResidualAttention(dim_hidden, dim_hidden, dim_hidden, num_heads, ln=ln, dropout_p=dropout_p))
        self.output = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        bs, num_inds, dim_in = x.shape
        assert num_inds == self.num_inds, f"num_inds: {num_inds} != {self.num_inds}"
        x = self.input(x)
        for layer in self.net:
            x = layer(x, x, None, None)
        x = self.output(x)
        return x

class EncoderBlock(AttentiveBlock):
    """ISAB in Set Transformer"""
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln, dropout_p, slot_att):
        super().__init__(dim_in, dim_out, num_heads, num_inds, ln, dropout_p, slot_att)


class DecoderBlock(AttentiveBlock):
    """Attentive Bottleneck Layer"""
    def __init__(self, dim_in, dim_out, dim_z, num_heads, num_inds, ln, dropout_p,
                 slot_att, i_net, i_net_layers, cond_prior=True):
        super().__init__(dim_in, dim_out, num_heads, num_inds, ln, dropout_p, slot_att)
        self.cond_prior = cond_prior
        if cond_prior:
            self.prior = InducedNetwork(num_inds, dim_out, dim_out, 2*dim_z, i_net_layers, num_heads, ln, dropout_p)
        else:
            self.register_parameter(name='prior', param=nn.Parameter(torch.randn(1, num_inds, 2*dim_z)))  # [1, M, 2Dz]
            nn.init.xavier_uniform_(self.prior)
        self.posterior = InducedNetwork(num_inds, dim_out, dim_out, 2*dim_z, i_net_layers, num_heads, ln, dropout_p)
        self.fc = nn.Linear(dim_z, dim_out)

    def compute_prior(self, h):
        """
        Sample from prior
        :param h: Tensor([B, M, D])
        :return: Tensor([B, M, Dz])
        """
        bs, num_inds, dim_in = h.shape
        if self.cond_prior:  # [B, M, 2Dz]
            prior = self.prior(h)
        else:
            prior = self.prior.repeat(bs, 1, 1)
        mu = prior[..., :prior.shape[-1]//2]  # [B, M, Dz]
        logvar = prior[..., prior.shape[-1]//2:].clamp(-4., 3.)
        eps = torch.randn(mu.shape).to(h)
        z = mu + torch.exp(logvar / 2.) * eps  # [B, M, Dz]
        return z, mu, logvar

    def compute_posterior(self, mu, logvar, bottom_up_h, h=None):
        """
        Estimate residual posterior parameters from prior parameters and top-down features
        :param mu: Tensor([B, M, D])
        :param logvar: Tensor([B, M, D])
        :param bottom_up_h: Tensor([B, M, D])
        :param h: Tensor([B, M, D])
        :return: Tensor([B, M, Dz]), Tensor([B, M, Dz])
        """
        bs, num_inds, dim_in = bottom_up_h.shape
        assert self.num_inds == num_inds
        bottom_up_h = bottom_up_h + h if h is not None else bottom_up_h
        posterior = self.posterior(bottom_up_h)
        mu2 = posterior[..., :posterior.shape[-1]//2]  # [B, M, Dz]
        logvar2 = posterior[..., posterior.shape[-1]//2:].clamp(-4., 3.)
        sigma = torch.exp(logvar / 2.)
        sigma2 = torch.exp(logvar2 / 2.)
        eps = torch.randn(mu.shape).to(mu)
        z = (mu + mu2) + (sigma * sigma2) * eps
        kl = -0.5 * (logvar2 + 1. - mu2.pow(2) / sigma.pow(2) - sigma2.pow(2)).view(mu.shape[0], -1).sum(dim=-1)  # [B,]
        return z, kl, mu2, logvar2

    def broadcast_latent(self, z, h, x, x_mask=None):
        return self.broadcast(self.fc(z), x, x_mask)  # No residual
