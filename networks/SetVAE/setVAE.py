import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from utils import draw

# A standard, simple Set-Transformer Block (ISAB)
class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab1 = nn.MultiheadAttention(dim_out, num_heads, batch_first=True)
        self.mab2 = nn.MultiheadAttention(dim_in, num_heads, batch_first=True)
        self.fc = nn.Linear(dim_in, dim_out) # Added to project input x to the right dimension for attention with I

    def forward(self, X):
        # Project X to the same dimension as the inducing points I
        X_proj = self.fc(X)
        H = self.mab1(self.I.repeat(X.size(0), 1, 1), X_proj, X_proj, need_weights=False)[0]
        return self.mab2(X, H, H, need_weights=False)[0]

# Simplified VAE for sets
class SetVAE(pl.LightningModule):
    def __init__(self, input_dim=6, max_outputs=512, hidden_dim=8, z_dim=4, num_heads=4, num_inds=32, lr=1e-4, beta=1.0, experiment_name='setVAE', sample_size=512, format_path='format.svg', **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.experiment_name = experiment_name
        self.sample_size = sample_size
        self.format = format_path
        # Encoder: Maps a set of strokes to a single latent vector
        self.encoder_pre = nn.Linear(input_dim, hidden_dim)
        self.encoder_isab1 = ISAB(hidden_dim, hidden_dim, num_heads, num_inds)
        self.encoder_isab2 = ISAB(hidden_dim, hidden_dim, num_heads, num_inds)
        self.encoder_pool = nn.AdaptiveAvgPool1d(1) # Pool across strokes to get a single vector
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

        # Decoder: Maps a latent vector back to a set of strokes
        self.decoder_pre = nn.Linear(z_dim, hidden_dim)
        self.decoder_isab1 = ISAB(hidden_dim, hidden_dim, num_heads, num_inds)
        self.decoder_isab2 = ISAB(hidden_dim, hidden_dim, num_heads, num_inds)
        self.decoder_post = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = F.relu(self.encoder_pre(x))
        print(f"h.shape: {h.shape}")
        h = self.encoder_isab1(h)
        print(f"h.shape: {h.shape}")
        h = self.encoder_isab2(h)
        print(f"h.shape: {h.shape}")
        # Pool across the set dimension (dim=1) to get a single feature vector per batch item
        # h = self.encoder_pool(h.transpose(1, 2)).squeeze(-1)
        print(f"h.shape: {h.shape}")
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_pre(z)
        print(f"h.shape: {h.shape}")
        # h = h.view(-1, self.hparams.max_outputs, self.hparams.hidden_dim)
        # print(f"h.shape: {h.shape}")
        h = F.relu(h)
        print(f"h.shape: {h.shape}")
        h = self.decoder_isab1(h)
        print(f"h.shape: {h.shape}")
        h = self.decoder_isab2(h)
        print(f"h.shape: {h.shape}")
        return torch.tanh(self.decoder_post(h)) # Use tanh to keep outputs in [-1, 1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        print(f"z.shape: {z.shape}")
        print(f"z: {z[0]}")
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def training_step(self, batch, batch_idx):
        x, x_mask = batch[0], batch[1]
        x_recon, mu, logvar = self(x)
        print(f"x_recon.shape: {x_recon.shape}")
        print(f"x_recon: {x_recon[0]}")
        # Use a standard VAE loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        loss = recon_loss + self.hparams.beta * kl_loss

        self.log_dict({'train_loss': loss, 'train_recon_loss': recon_loss, 'train_kl_loss': kl_loss})
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def on_before_optimizer_step(self, optimizer):
        for name, param in self.named_parameters():
            if param.grad is None:
                print(f"!!! GRADIENT IS NONE for parameter: {name}")
            else:
                norm = torch.norm(param.grad)
                if torch.isnan(norm) or torch.isinf(norm):
                    print(f"!!! INVALID GRADIENT (NaN or Inf) for parameter: {name}")
    
    def validation_step(self, batch, batch_idx):
        x, x_mask = batch[0], batch[1]
        x_recon, mu, logvar = self(x)
        filename = f'Results/{self.experiment_name}/{self.current_epoch}.svg'
        print(f"x_recon: {x_recon}")
        draw(self.format, self.sample_size, filename, x_recon)
        