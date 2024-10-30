# Adapt from https://github.com/THUDM/CogDL/blob/281f47424d58844b167ccbe41d9829c1f77689f8/examples/graphmae/graphmae/models/edcoder.py

from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.nn.pytorch.conv as dglnn
import dgl
from torch import nn

from edcoders import *
from utils import obtain_act, obtain_norm, obtain_pooler, sce_loss, mask_edge, drop_edge

# ======================================================================
#   Predictive SSL
# ======================================================================

class GraphMAE(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hid_dim: int,
            num_layer: int,
            drop_ratio: float,
            act: str,
            norm: Optional[str],
            residual: bool,
            mask_ratio: float = 0.3,
            encoder_type: str = "gcn",
            decoder_type: str = "gcn",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_ratio: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
         ):
        super(GraphMAE, self).__init__()
        self._mask_ratio = mask_ratio

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = hid_dim
        self._concat_hidden = concat_hidden
        
        self._replace_ratio = replace_ratio
        self._mask_token_rate = 1 - self._replace_ratio

        enc_in_dim = in_dim
        enc_hid_dim = hid_dim
        enc_out_dim = hid_dim

        dec_in_dim = hid_dim
        dec_hid_dim = hid_dim 
        dec_out_dim = in_dim 

        # helpers
        self.in_dim = in_dim
        self.embed_dim = enc_out_dim

        # encoder setting
        if encoder_type == "gcn":
            self.encoder = GCN(
                in_dim=enc_in_dim, 
                num_hidden=enc_hid_dim, 
                out_dim=enc_out_dim,
                num_layers=num_layer,
                dropout=drop_ratio,
                activation=act,
                norm= obtain_norm(norm),
                residual=residual,
                encoding = True,
            )
        elif encoder_type == 'bwgnn':
            self.encoder = BWGNN(
                in_dim=in_dim, 
                num_hidden=enc_hid_dim, 
                encoding = True,
            )
            # update the output dims
            enc_out_dim = enc_hid_dim * len(self.encoder.conv)
            self.embed_dim = enc_out_dim
            dec_in_dim = enc_out_dim
        elif encoder_type == "gin":
            self.encoder = GIN(
                in_dim=enc_in_dim,
                num_hidden=enc_hid_dim,
                out_dim=enc_out_dim,
                num_layers=num_layer,
                dropout=drop_ratio,
                activation=act,
                residual=residual,
                norm=norm,
                encoding=True,
            )
        else:
            raise NotImplementedError
        
        # decoder setting
        if decoder_type == "gcn":
            self.decoder = GCN(
                in_dim=dec_in_dim, 
                num_hidden=dec_hid_dim, 
                out_dim=dec_out_dim,
                num_layers=num_layer,
                dropout=drop_ratio,
                activation=act,
                norm= obtain_norm(norm),
                residual=residual,
                encoding = False,
            )
        elif decoder_type == 'gin':
            self.decoder = GIN(
                in_dim=dec_in_dim,
                num_hidden=dec_hid_dim,
                out_dim=dec_out_dim,
                num_layers=num_layer,
                dropout=drop_ratio,
                activation=act,
                residual=residual,
                norm=norm,
                encoding=True,
            )
        elif decoder_type == 'mlp':
            # 3 layers MLP
            self.decoder = nn.Sequential(
                nn.Linear(dec_in_dim, dec_hid_dim),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(dec_hid_dim, dec_hid_dim),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(dec_hid_dim, dec_out_dim)
            )
        else:
            raise NotImplementedError



        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(enc_out_dim * num_layer, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(enc_out_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, g, x, mask_ratio=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_ratio * num_nodes)

        # random masking
        num_mask_nodes = int(mask_ratio * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_ratio > 0:
            num_noise_nodes = int(self._replace_ratio * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_ratio * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def forward(self, g, x):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(g, x)
        loss_item = {"loss": loss.item()}
        return loss, loss_item
    
    def mask_attr_prediction(self, g, x):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_ratio)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep= self.encoder(use_g, use_x)
        # enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        # if self._concat_hidden:
        #     enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "linear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss

    def embed(self, g, x):
        # rep = self.encoder(g, x)
        return self.encoder(g, x)

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

