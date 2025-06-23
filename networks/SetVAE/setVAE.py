import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import torch.distributions as D
from utils import draw
# Helper functions from metrics and ops, assuming they exist and are correct
def get_mask(output_sizes, max_outputs):
    """Creates a boolean mask of shape [B, N]"""
    bs = output_sizes.shape[0]
    mask = torch.arange(max_outputs, device=output_sizes.device).expand(bs, max_outputs) >= output_sizes.unsqueeze(1)
    return mask

def sample_mask(output_sizes, max_outputs):
    """Creates a boolean mask with sampled indices."""
    return get_mask(output_sizes, max_outputs) # simplified for now

def masked_fill(tensor, mask, value):
    if mask is None:
        return tensor
    return tensor.masked_fill(mask.unsqueeze(-1), value)

def chamfer_loss(a, a_mask, b, b_mask, accelerate=False):
    # a: [B, N, D], b: [B, M, D]
    # simplified version
    a = a.float()
    b = b.float()
    a_norm = (a**2).sum(-1, keepdim=True)
    b_norm = (b**2).sum(-1, keepdim=True)
    dist = a_norm + b_norm.transpose(1, 2) - 2 * torch.bmm(a, b.transpose(1, 2))
    dist = torch.sqrt(torch.clamp(dist, min=0))
    
    if a_mask is not None:
      dist = torch.where(a_mask.unsqueeze(2), torch.full_like(dist, float('inf')), dist)
    if b_mask is not None:
      dist = torch.where(b_mask.unsqueeze(1), torch.full_like(dist, float('inf')), dist)

    dist_ab, _ = torch.min(dist, dim=2)
    dist_ba, _ = torch.min(dist, dim=1)

    if a_mask is not None:
        dist_ab.masked_fill_(a_mask, 0)
    if b_mask is not None:
        dist_ba.masked_fill_(b_mask, 0)

    loss = dist_ab.sum(1) / ((~a_mask).float().sum(1) + 1e-8) + dist_ba.sum(1) / ((~b_mask).float().sum(1) + 1e-8)
    return loss.mean()

def emd_loss(a, a_mask, b, b_mask):
    # This is hard to mock. I will assume it returns a scalar loss.
    return chamfer_loss(a, a_mask, b, b_mask)

# --- From layers.py ---
class InitialSet(nn.Module):
    def __init__(self, dim_seed, n_mixtures, dim_out, max_outputs, fixed_gmm, train_gmm):
        super().__init__()
        self.dim_seed = dim_seed
        self.n_mixtures = n_mixtures
        self.dim_out = dim_out
        self.max_outputs = max_outputs
        self.fixed_gmm = fixed_gmm
        self.train_gmm = train_gmm
        self.tau = 1.
        
        self.register = self.register_parameter if train_gmm else self.register_buffer

        if n_mixtures == 1:
            self.register('mu', nn.Parameter(torch.randn(1, 1, dim_seed)))
            self.register('logvar', nn.Parameter(torch.randn(1, 1, dim_seed)))
            nn.init.xavier_normal_(self.mu)
            nn.init.xavier_normal_(self.logvar)
        elif fixed_gmm:
            # Sphere lattice is not available, using random
            logits = torch.ones(n_mixtures,)
            mu = torch.randn(n_mixtures, dim_seed)
            sig = torch.ones(n_mixtures, dim_seed) * 0.1
            self.register('logits', nn.Parameter(logits))
            self.register('mu', nn.Parameter(mu))
            self.register('sig', nn.Parameter(sig))
        else:
            self.register('logits', nn.Parameter(torch.ones(n_mixtures,)))
            self.register('mu', nn.Parameter(torch.randn(n_mixtures, dim_seed)))
            self.register('sig', nn.Parameter(torch.ones(n_mixtures, dim_seed)))
        
        self.output = nn.Linear(dim_seed, dim_out)

    def forward(self, output_sizes, hold_seed=None, hold_initial_set=False):
        bsize = output_sizes.shape[0]
        if hold_initial_set:
            x_mask = get_mask(output_sizes, self.max_outputs)
        else:
            x_mask = sample_mask(output_sizes, self.max_outputs)

        if hold_seed is not None:
            torch.random.manual_seed(hold_seed)
            eps = torch.randn([1, self.max_outputs, self.dim_seed]).to(x_mask.device).repeat(bsize, 1, 1)
        else:
            eps = torch.randn([bsize, self.max_outputs, self.dim_seed]).to(x_mask.device)

        if self.n_mixtures == 1:
            x = self.mu + torch.exp(self.logvar / 2.) * eps
        else:
            mix = D.Categorical(self.logits)
            comp = D.Independent(D.Normal(self.mu, self.sig.abs()), 1)
            mixture = D.MixtureSameFamily(mix, comp)
            x = mixture.sample((bsize, self.max_outputs))

        x = self.output(x)
        return x, x_mask

class ResidualAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, num_heads, ln=False, dropout_p=0., slot_att=False):
        super().__init__()
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.dim_split = dim_v // num_heads if dim_v >= num_heads else 1
        self.dropout_p = dropout_p
        self.slot_att = slot_att
        self.fc_q = nn.Linear(dim_q, dim_v, bias=False)
        self.fc_k = nn.Linear(dim_k, dim_v, bias=False)
        self.fc_v = nn.Linear(dim_k, dim_v, bias=False)
        self.fc_o = nn.Linear(dim_v, dim_q, bias=False)
        self.ffn1 = nn.Linear(dim_q, dim_q)
        self.ffn2 = nn.Linear(dim_q, dim_q)
        if ln:
            self.ln_o1 = nn.LayerNorm(dim_q)
            self.ln_o2 = nn.LayerNorm(dim_q)
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)

    def compute_attention(self, query, key, value, x_mask, y_mask):
        bs, xs, ys = query.shape[0], query.shape[1], value.shape[1]
        
        q_ = torch.cat(query.split(self.dim_split, 2), 0)
        k_ = torch.cat(key.split(self.dim_split, 2), 0)
        v_ = torch.cat(value.split(self.dim_split, 2), 0)
        
        sdp = torch.bmm(q_, k_.transpose(1, 2)) / math.sqrt(self.dim_v) # [H*B, N, M]
        
        if x_mask is not None or y_mask is not None:
            x_m = x_mask.unsqueeze(2) if x_mask is not None else torch.zeros(bs, xs, 1, dtype=torch.bool, device=query.device) # [B, N, 1]
            y_m = y_mask.unsqueeze(1) if y_mask is not None else torch.zeros(bs, 1, ys, dtype=torch.bool, device=query.device) # [B, 1, M]
            mask = (x_m | y_m) # [B, N, M]
            mask = mask.repeat(self.num_heads, 1, 1) # [H*B, N, M]
            sdp.masked_fill_(mask, -float('inf'))

        alpha = torch.softmax(sdp, -1) # [H*B, N, M]

        att = torch.bmm(alpha, v_)
        att = torch.cat(att.split(bs, 0), 2)
        
        return att, alpha.view(self.num_heads, bs, xs, ys)

    def forward(self, x, y, x_mask=None, y_mask=None, get_alpha=False):
        q = self.fc_q(x)
        k = self.fc_k(y)
        v = self.fc_v(y)

        att, alpha = self.compute_attention(q, k, v, x_mask, y_mask)
        att = self.fc_o(att)
        if hasattr(self, 'dropout'):
            att = self.dropout(att)
        
        o = x + att
        if hasattr(self, 'ln_o1'):
            o = self.ln_o1(o)
        
        ff = self.ffn2(F.relu(self.ffn1(o)))
        if hasattr(self, 'dropout'):
            ff = self.dropout(ff)
            
        o = o + ff
        if hasattr(self, 'ln_o2'):
            o = self.ln_o2(o)
        
        if x_mask is not None:
            o = o.masked_fill(x_mask.unsqueeze(-1), 0)
            
        if get_alpha:
            return o, alpha
        return o

class AttentiveBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, dropout_p=0., slot_att=False):
        super().__init__()
        self.num_inds = num_inds
        self.register_parameter(name='i', param=nn.Parameter(torch.randn(1, num_inds, dim_out)))
        nn.init.xavier_uniform_(self.i)
        self.att1 = ResidualAttention(dim_out, dim_in, dim_out, num_heads, ln, dropout_p, slot_att)
        self.att2 = ResidualAttention(dim_in, dim_out, dim_out, num_heads, ln, dropout_p)

    def project(self, x, x_mask=None):
        i = self.i.repeat(x.shape[0], 1, 1)
        h, alpha = self.att1(i, x, None, x_mask, get_alpha=True)
        return h, alpha.transpose(2, 3)

    def broadcast(self, h, x, x_mask=None):
        o, alpha = self.att2(x, h, x_mask, None, get_alpha=True)
        return o, alpha

    def forward(self, x, x_mask=None):
        h, alpha1 = self.project(x, x_mask)
        o, alpha2 = self.broadcast(h, x, x_mask)
        return o, h, alpha1, alpha2

class ISAB(AttentiveBlock):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln, dropout_p=0.):
        super().__init__(dim_in, dim_out, num_heads, num_inds, ln=ln, dropout_p=dropout_p, slot_att=False)
    def forward(self, x, x_mask=None):
        h, alpha1 = self.project(x, x_mask)
        o, alpha2 = self.broadcast(h, x, x_mask)
        return o

# --- From mlp.py ---
class ElementwiseMLP(nn.Module):
    # a simplified version
    def __init__(self, dim_in, dim_hidden, dim_out, n_hidden, activation, residual, use_bn, use_masked_bn, dropout_p):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_hidden):
            self.layers.append(nn.Linear(dim_in if i==0 else dim_hidden, dim_hidden))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(dim_hidden, dim_out))
    def forward(self, x, x_mask=None):
        for layer in self.layers:
            x = layer(x)
        return x

class MLP(nn.Module):
    # a simplified version
    def __init__(self, dim_in, dim_hidden, dim_out, n_hidden, dropout_p):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_hidden):
            self.layers.append(nn.Linear(dim_in if i==0 else dim_hidden, dim_hidden))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(dim_hidden, dim_out))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- From modules.py ---
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
    def __init__(self, net_type, num_inds, dim_in, dim_hidden, dim_out, n_hidden, num_heads=0, ln=False, dropout_p=0.):
        super().__init__()
        self.num_inds = num_inds
        self.net_type = net_type
        if self.net_type == 'full_mlp':
            self.net = MLP(num_inds*dim_in, dim_hidden, num_inds*dim_out, n_hidden, dropout_p=dropout_p)
        elif self.net_type == 'elem_mlp':
            self.net = ElementwiseMLP(dim_in, dim_hidden, dim_out, n_hidden, 'relu', False, False, False, dropout_p=dropout_p)
        elif self.net_type == 'set_transformer':
            self.input = nn.Linear(dim_in, dim_hidden)
            self.net = nn.ModuleList()
            for i in range(n_hidden):
                self.net.append(ResidualAttention(dim_hidden, dim_hidden, dim_hidden, num_heads, ln=ln, dropout_p=dropout_p))
            self.output = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        bs, num_inds, dim_in = x.shape
        assert num_inds == self.num_inds
        if self.net_type == 'full_mlp':
            return self.net(x.reshape([bs, -1])).reshape([bs, num_inds, -1])
        elif self.net_type == 'elem_mlp':
            return self.net(x)
        elif self.net_type == 'set_transformer':
            x = self.input(x)
            for layer in self.net:
                x = layer(x, x, None, None)
            return self.output(x)


class EncoderBlock(AttentiveBlock):
    """ISAB in Set Transformer"""
    pass


class DecoderBlock(AttentiveBlock):
    """ABL (Attentive Bottleneck Layer)"""
    def __init__(self, dim_in, dim_out, dim_z, num_heads, num_inds, ln, dropout_p,
                 slot_att, i_net, i_net_layers, cond_prior=True):
        super().__init__(dim_in, dim_out, num_heads, num_inds, ln, dropout_p, slot_att)
        self.cond_prior = cond_prior
        if cond_prior:
            self.prior = InducedNetwork(i_net, num_inds, dim_out, dim_out, 2*dim_z, i_net_layers, num_heads, ln, dropout_p)
        else:
            self.register_parameter(name='prior', param=nn.Parameter(torch.randn(1, num_inds, 2*dim_z)))  # [1, M, 2Dz]
            nn.init.xavier_uniform_(self.prior)
        self.posterior = InducedNetwork(i_net, num_inds, dim_out, dim_out, 2*dim_z, i_net_layers, num_heads, ln, dropout_p)
        self.fc = nn.Linear(dim_z, dim_out)

    def compute_prior(self, h):
        bs, num_inds, dim_in = h.shape
        if self.cond_prior:
            prior = self.prior(h)
        else:
            prior = self.prior.repeat(bs, 1, 1)
        mu = prior[..., :prior.shape[-1]//2]
        logvar = prior[..., prior.shape[-1]//2:].clamp(-4., 3.)
        eps = torch.randn(mu.shape).to(h)
        z = mu + torch.exp(logvar / 2.) * eps
        return z, mu, logvar

    def compute_posterior(self, mu, logvar, bottom_up_h, h=None):
        bs, num_inds, dim_in = bottom_up_h.shape
        assert self.num_inds == num_inds
        bottom_up_h = bottom_up_h + h if h is not None else bottom_up_h
        posterior = self.posterior(bottom_up_h)
        mu2 = posterior[..., :posterior.shape[-1]//2]
        logvar2 = posterior[..., posterior.shape[-1]//2:].clamp(-4., 3.)
        sigma = torch.exp(logvar / 2.)
        sigma2 = torch.exp(logvar2 / 2.)
        eps = torch.randn(mu.shape).to(mu)
        z = (mu + mu2) + (sigma * sigma2) * eps
        kl = -0.5 * (logvar2 + 1. - mu2.pow(2) / sigma.pow(2) - sigma2.pow(2)).view(mu.shape[0], -1).sum(dim=-1)
        return z, kl, mu2, logvar2

    def broadcast_latent(self, z, h, x, x_mask=None):
        return self.broadcast(self.fc(z), x, x_mask)


class SetVAE(pl.LightningModule):
    def __init__(
        self,
        input_dim=2,
        max_outputs=100,
        z_scales=[256,128,64,32,32],
        hidden_dim=128,
        num_heads=4,
        z_dim=32,
        fixed_gmm=False,
        train_gmm=False,
        init_dim=32,
        n_mixtures=16,
        slot_att=False,
        i_net='set_transformer',
        i_net_layers=2,
        d_net='set_transformer',
        enc_in_layers=2,
        dec_in_layers=2,
        dec_out_layers=2,
        isab_inds=16,
        ln=True,
        dropout_p=0.,
        activation='relu',
        use_bn=False,
        residual=False,
        optimizer='adam',
        lr=1e-4,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        momentum=0.9,
        matcher='chamfer',
        beta=1.0,
        kl_warmup_epochs=0,
        format_path='format.svg',
        sample_size=512,
        experiment_name='setVAE',
    ):
        super().__init__()
        self.save_hyperparameters()
        self.format = format_path
        self.sample_size = sample_size
        self.input_dim = input_dim
        self.max_outputs = max_outputs
        self.z_scales = z_scales
        self.n_layers = len(z_scales)
        self.z_dim = z_dim
        self.enc_inds = list(reversed(self.z_scales))
        self.dec_inds = self.z_scales
        self.experiment_name = experiment_name

        self.input_proj = nn.Linear(self.hparams.input_dim, self.hparams.hidden_dim)
        
        self.init_set = InitialSet(
            dim_seed=self.hparams.init_dim,
            n_mixtures=self.hparams.n_mixtures,
            dim_out=self.hparams.hidden_dim,
            max_outputs=self.hparams.max_outputs,
            fixed_gmm=self.hparams.fixed_gmm,
            train_gmm=self.hparams.train_gmm,
        )

        self.pre_encoder = DeterministicNetwork(self.hparams.d_net, self.hparams.isab_inds, self.hparams.hidden_dim, self.hparams.hidden_dim, self.hparams.enc_in_layers, self.hparams.num_heads, self.hparams.ln, self.hparams.dropout_p, self.hparams.activation, self.hparams.use_bn, self.hparams.residual)
        self.pre_decoder = DeterministicNetwork(self.hparams.d_net, self.hparams.isab_inds, self.hparams.hidden_dim, self.hparams.hidden_dim, self.hparams.dec_in_layers, self.hparams.num_heads, self.hparams.ln, self.hparams.dropout_p, self.hparams.activation, self.hparams.use_bn, self.hparams.residual)
        self.post_decoder = DeterministicNetwork(self.hparams.d_net, self.hparams.isab_inds, self.hparams.hidden_dim, self.hparams.hidden_dim, self.hparams.dec_out_layers, self.hparams.num_heads, self.hparams.ln, self.hparams.dropout_p, self.hparams.activation, self.hparams.use_bn, self.hparams.residual)

        self.encoder = nn.ModuleList([
            EncoderBlock(self.hparams.hidden_dim, self.hparams.hidden_dim, self.hparams.num_heads, self.enc_inds[i], self.hparams.ln, self.hparams.dropout_p, self.hparams.slot_att)
            for i in range(self.n_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(self.hparams.hidden_dim, self.hparams.hidden_dim, self.hparams.z_dim, self.hparams.num_heads, self.dec_inds[i], self.hparams.ln, self.hparams.dropout_p, self.hparams.slot_att, self.hparams.i_net, self.hparams.i_net_layers, cond_prior=i > 0)
            for i in range(self.n_layers)
        ])
        
        self.output_proj = nn.Linear(self.hparams.hidden_dim, self.hparams.input_dim)

    def bottom_up(self, x, x_mask):
        x = self.input_proj(x)

        x = self.pre_encoder(x, x_mask)
        features = []
        for layer in self.encoder:
            x, h, _, _ = layer(x, x_mask)
            features.append(h)
        return features

    def top_down(self, cardinality, bottom_up_h):
        o, o_mask = self.init_set(cardinality)
        o = self.pre_decoder(o, o_mask)
        kls = []
        posteriors = []
        for idx, layer in enumerate(self.decoder):
            h, _ = layer.project(o, o_mask)
            _, mu, logvar = layer.compute_prior(h)
            z, kl, mu2, logvar2 = layer.compute_posterior(mu, logvar, bottom_up_h[idx], None if idx == 0 else h)
            o, _ = layer.broadcast_latent(z, h, o, o_mask)
            kls.append(kl)
            posteriors.append((z, mu2, logvar2))
        o = self.post_decoder(o, o_mask)
     
        o = self.output_proj(o)
        return {'set': o, 'set_mask': o_mask, 'kls': kls, 'posteriors': posteriors}

    def forward(self, x, x_mask):
      
        features = self.bottom_up(x, x_mask)
        output = self.top_down((~x_mask).sum(-1), list(reversed(features)))
        return output

    def training_step(self, batch, batch_idx):
        x, x_mask = batch[0], batch[1]
        output = self(x, x_mask)
        output_set, output_mask, kls = output['set'], output['set_mask'], output['kls']

        if self.hparams.matcher == 'chamfer':
            recon_loss = chamfer_loss(output_set, output_mask, x, x_mask)
        elif self.hparams.matcher == 'emd':
            recon_loss = emd_loss(output_set, output_mask, x, x_mask)
        else: # all
             recon_loss = chamfer_loss(output_set, output_mask, x, x_mask) + emd_loss(output_set, output_mask, x, x_mask)

        kl_loss = torch.stack(kls, dim=1).sum(dim=1).mean()
        
        kl_weight = self.hparams.beta
        if self.hparams.kl_warmup_epochs > 0:
            kl_weight *= torch.clamp(torch.tensor(self.current_epoch / self.hparams.kl_warmup_epochs, device=self.device), 0.0, 1.0)
            
        loss = recon_loss + kl_weight * kl_loss
        
        self.log('train/loss', loss)
        self.log('train/recon_loss', recon_loss)
        self.log('train/kl_loss', kl_loss)
        self.log('train/kl_weight', kl_weight)
        
        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum)
        else:
            raise ValueError("Optimizer should be either 'adam' or 'sgd'")
        return optimizer

    def sample(self, output_sizes, given_latents=None):
        o, o_mask = self.init_set(output_sizes)
        o = self.pre_decoder(o, o_mask)
        priors = []
        for idx, layer in enumerate(self.decoder):
            h, _ = layer.project(o, o_mask)
            z, mu, logvar = layer.compute_prior(h)
            if given_latents is not None:
                z = given_latents[idx]
            o, _ = layer.broadcast_latent(z, h, o, o_mask)
            priors.append((z, mu, logvar))
        o = self.post_decoder(o, o_mask)
        o = self.output_proj(o)
        return {'set': o, 'set_mask': o_mask, 'priors': priors} 
    
    def validation_step(self, batch, batch_idx):
        x, x_mask = batch[0], batch[1]
        output = self(x, x_mask)
        output_set, output_mask, kls = output['set'], output['set_mask'], output['kls']
        filename = f'Results/{self.experiment_name}/{self.current_epoch}.svg'
        draw(self.format, self.sample_size, filename, output_set)
        