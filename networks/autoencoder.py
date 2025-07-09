import torch.nn as nn
# from .modules import DeterministicNetwork, EncoderBlock, DecoderBlock
import pytorch_lightning as L
# from .layers import InitialSet
import torch
# from .criterion import ChamferCriterion
# from .ops import get_module
from utils import draw

class SetVAE(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.input_dim = args.input_dim
        self.max_outputs = args.max_outputs
        self.train_gmm = args.train_gmm
        self.init_dim = args.init_dim
        self.n_mixtures = args.n_mixtures
        self.n_layers = len(args.z_scales)
        self.z_dim = args.z_dim
        self.z_scales = args.z_scales
        self.hidden_dim = args.hidden_dim
        self.num_heads = args.num_heads
        self.slot_att = args.slot_att
        self.i_net = args.i_net
        self.i_net_layers = args.i_net_layers
        self.d_net = args.d_net
        self.enc_in_layers = args.enc_in_layers
        self.dec_in_layers = args.dec_in_layers
        self.dec_out_layers = args.dec_out_layers
        self.isab_inds = args.isab_inds
        self.ln = args.ln
        self.dropout_p = args.dropout_p
        self.activation = args.activation
        self.use_bn = args.use_bn
        self.residual = args.residual
        self.enc_inds = list(reversed(self.z_scales))  # bottom-up
        self.dec_inds = self.z_scales  # top-down
        self.input = nn.Linear(self.input_dim, self.hidden_dim)
        self.init_set = InitialSet(self.init_dim, self.n_mixtures, self.hidden_dim, self.max_outputs, self.train_gmm)
        self.pre_encoder = DeterministicNetwork(self.d_net, self.isab_inds, self.hidden_dim, self.hidden_dim,
                                                self.enc_in_layers, self.num_heads, self.ln, self.dropout_p,
                                                self.activation, self.use_bn, self.residual)
        self.pre_decoder = DeterministicNetwork(self.d_net, self.isab_inds, self.hidden_dim, self.hidden_dim,
                                                self.dec_in_layers, self.num_heads, self.ln, self.dropout_p,
                                                self.activation, self.use_bn, self.residual)
        self.post_decoder = DeterministicNetwork(self.d_net, self.isab_inds, self.hidden_dim, self.hidden_dim,
                                                 self.dec_out_layers, self.num_heads, self.ln, self.dropout_p,
                                                 self.activation, self.use_bn, self.residual)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.args = args
        self.format = "format.svg"
        self.sample_size = args.sample_size
        for i in range(self.n_layers):
            self.encoder.append(EncoderBlock(self.hidden_dim, self.hidden_dim, self.num_heads,
                                             self.enc_inds[i], self.ln, self.dropout_p, self.slot_att))
            self.decoder.append(DecoderBlock(self.hidden_dim, self.hidden_dim, self.z_dim, self.num_heads,
                                             self.dec_inds[i], self.ln, self.dropout_p, self.slot_att, self.i_net,
                                             self.i_net_layers, cond_prior=i > 0))
        self.output = nn.Linear(self.hidden_dim, self.input_dim)
        self.lr = args.lr
    
    def bottom_up(self, x, x_mask):
        """ Deterministic bottom-up encoding
        :param x: Tensor([B, N, Di])
        :param x_mask: BoolTensor([B, N])
        :return: List([Tensor([B, M, D])]), List([Tensor([H, B, N, M]), Tensor([H, B, N, M])])
        """
        x = self.input(x)  # [B, N, D]
        x = self.pre_encoder(x, x_mask)
        features = list()
        alphas = list()
        for layer in get_module(self.encoder):
            x, h, alpha1, alpha2 = layer(x, x_mask)  # [B, N, D], [B, M, D], [H, B, N, M], [H, B, N, M]
            features.append(h)
            alphas.append((alpha1, alpha2))
        for i in range(len(features)):
            print(f"features[{i}]: {features[i].shape}")
            print(f"alphas[{i}][0]: {alphas[i][0].shape}")
            print(f"alphas[{i}][1]: {alphas[i][1].shape}")
        return {'features': features, 'alphas': alphas}

    def top_down(self, cardinality, bottom_up_h):
        """ Stochastic top-down decoding
        :param cardinality: Tensor([B,])
        :param bottom_up_h: List([Tensor([B, M, D])]) in top-down order
        :return:
        """
        o, o_mask = self.init_set(cardinality)
        o = self.pre_decoder(o, o_mask)
        alphas, posteriors, kls = [], [(o, None, None)], []
        for idx, layer in enumerate(get_module(self.decoder)):
            h, alpha1 = layer.project(o, o_mask)
            _, mu, logvar = layer.compute_prior(h)
            z, kl, mu2, logvar2 = layer.compute_posterior(mu, logvar, bottom_up_h[idx], None if idx == 0 else h)
            o, alpha2 = layer.broadcast_latent(z, h, o, o_mask)
            alphas.append((alpha1, alpha2))
            posteriors.append((z, mu2, logvar2))
            kls.append(kl)
        o = self.post_decoder(o, o_mask)
        o = self.output(o)  # [B, N, Do]
        return {'set': o, 'set_mask': o_mask,
                'posteriors': posteriors, 'kls': kls, 'alphas': alphas}

    def forward(self, x, x_mask):
        """ Bidirectional inference
        :param x: Tensor([B, N, Di])
        :param x_mask: BoolTensor([B, N])
        :return: Tensor([B, N, Do]), Tensor([B, N]), List([Tensor([H, B, N, M]), Tensor([H, B, N, M])]) * 2
        """
        print(f"x: {x.shape}")
        print(f"x_mask: {x_mask.shape}")
        bup = self.bottom_up(x, x_mask)
        print(bup['features'][0].shape)
        tdn = self.top_down((~x_mask).sum(-1), list(reversed(bup['features'])))
        print(tdn['set'].shape)
        print(tdn['set_mask'].shape)
        o, o_mask = self.postprocess(tdn['set'], tdn['set_mask'])
        return {'set': o, 'set_mask': o_mask,
                'posteriors': tdn['posteriors'], 'kls': tdn['kls'],
                'alphas': (bup['alphas'], tdn['alphas'])}

    def sample(self, output_sizes, hold_seed=None, hold_initial_set=False, given_latents=None):
        """ Top-down generation
        :param output_sizes: Tensor([B,])
        :param hold_seed
        :param hold_initial_set
        :param given_latents: List([Tensor([B, ?, D])])
        :return: Tensor([B, N, Do]), Tensor([B, N]), List([Tensor([B, M, D])]),
                 List([Tensor([H, B, N, M]), Tensor([H, B, N, M])])
        """
        o, o_mask = self.init_set(output_sizes, hold_seed, hold_initial_set)
        o = self.pre_decoder(o, o_mask)
        priors = [(o, None, None)]
        # if given_latents is not None:
        #    o = given_latents[0]
        #    assert o.shape[1] == self.max_outputs
        alphas = list()
        for idx, layer in enumerate(get_module(self.gpu, self.decoder)):
            h, alpha1 = layer.project(o, o_mask)
            if idx == 0:
                z, mu, logvar = layer.compute_prior(h)
            z, mu, logvar = layer.compute_prior(h)
            if given_latents is not None:
                z = given_latents[idx + 1]
                assert z.shape[1] == mu.shape[1]
            o, alpha2 = layer.broadcast_latent(z, h, o, o_mask)
            priors.append((z, mu, logvar))
            alphas.append((alpha1, alpha2))
        o = self.post_decoder(o, o_mask)
        o = self.output(o)  # [B, N, Do]
        o, o_mask = self.postprocess(o, o_mask)
        return {'set': o, 'set_mask': o_mask,
                'priors': priors, 'alphas': alphas}
    
    @staticmethod
    def postprocess(x, x_mask):
        if x.shape[-1] == 2:  # MNIST, xy
            return (torch.tanh(x) + 1) / 2., x_mask  # [B, N, Do], [0, 1] range
        elif x.shape[-1] == 3:  # ShapeNet, xyz
            return x, x_mask  # [B, N, Do]
        elif x.shape[-1] == 4:  # KITTI, xyzc
            x = x.clone()
            x[..., -1] = (torch.tanh(x[..., -1]) + 1) / 2.
            return x, x_mask  # [B, N, Do]
        else:
            return x, x_mask
    

    def training_step(self, batch, batch_idx):
        criterion = ChamferCriterion(self.args)
        x, x_mask = batch
        output = self(x, x_mask)

        loss = criterion(output, x, x_mask, self.args, self.current_epoch)
        loss, kl_loss, l2_loss, topdown_kl, beta = loss['loss'], loss['kl'], loss['l2'], loss['topdown_kl'], loss['beta']

        self.log('train_loss', loss)
        self.log('train_kl_loss', kl_loss)
        self.log('train_l2_loss', l2_loss)
        #self.log('train_topdown_kl', topdown_kl)
        #self.log('train_beta', beta)
    
    def validation_step(self, batch, batch_idx):
        criterion = ChamferCriterion(self.args)
        x, x_mask = batch
        output = self(x, x_mask)
        loss = criterion(output, x, x_mask, self.args, self.current_epoch)
        loss, kl_loss, l2_loss, topdown_kl, beta = loss['loss'], loss['kl'], loss['l2'], loss['topdown_kl'], loss['beta']
        
        self.log('val_loss', loss)
        self.log('val_kl_loss', kl_loss)
        filename = f'/scratch/ks02450/Results/{self.args.experiment_name}/{self.current_epoch}_{batch_idx}.svg'
        draw(self.format, self.sample_size, filename, output['set'])
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer