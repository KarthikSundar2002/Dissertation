import torch.nn as nn
import pytorch_lightning as L


def _weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname or 'Linear' in classname:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)

class CB_AE(nn.Module):
    def __init__(self, latent_shape, hidden_dim, concept_dim):
        super().__init__()
        assert len(latent_shape) == 3
        noise_dim = latent_shape[0] * latent_shape[1] * latent_shape[2]
        self.noise_dim = noise_dim
        self.latent_shape = latent_shape
        self.encoder = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, concept_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(concept_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, noise_dim),
        )

        print('number of layers in CB-AE:', len(self.encoder) + len(self.decoder))
        
        self.apply(_weights_init)
    
    def enc(self, x):
        # the latent vector will be like (batch_size, 512, 8, 8) --- 2D conv features
        # so we flatten and keep only batch dimension
        x = x.reshape(x.shape[0], self.noise_dim)
        return self.encoder(x)
    
    def dec(self, x):
        x = self.decoder(x)
        x = x.reshape(x.shape[0], self.latent_shape[0], self.latent_shape[1], self.latent_shape[2])
        return x


    def forward(self, x):
        return self.decoder(self.encoder(x))

class CC(nn.Module):
    def __init__(self, latent_shape, hidden_dim, concept_dim):
        super().__init__()
        assert len(latent_shape) == 3
        noise_dim = latent_shape[0] * latent_shape[1] * latent_shape[2]
        self.noise_dim = noise_dim
        self.latent_shape = latent_shape
        self.encoder = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, concept_dim),
        )

        print('number of layers in CB-E:', len(self.encoder))

        self.apply(_weights_init)

    def enc(self, x):
        # the latent vector will be like (batch_size, 512, 8, 8) --- 2D conv features
        # so we flatten and keep only batch dimension
        x = x.reshape(x.shape[0], self.noise_dim)
        return self.encoder(x)

    def forward(self, x):
        return (self.encoder(x))
    

class CB(L.LightningModule()):
