from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    sinusoid_encoding, MaskedConv1D, AttNPool1D, TransformerEncoder
)

from .weight_init import trunc_normal_


backbones = dict()
def register_text_net(name):
    def decorator(module):
        backbones[name] = module
        return module
    return decorator


@register_text_net('identity')
class TextIdentity(nn.Module):

    def __init__(
        self, 
        in_dim,                 # text feature dimension
        embd_dim,               # embedding dimension
        max_seq_len,            # max sequence length
        n_heads=4,              # number of attention heads
        use_abs_pe=False,       # whether to apply absolute position encoding
        use_bkgd_token=True,    # whether to add background token
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        if embd_dim is not None:
            self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)
        else:
            embd_dim = in_dim
            self.embd_fc = None

        # position encoding (c, t)
        if use_abs_pe:
            pe = sinusoid_encoding(max_seq_len, embd_dim // 2)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        if use_bkgd_token:
            self.attn_pool = AttNPool1D(embd_dim, n_heads=n_heads)
        else:
            self.attn_pool = None

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        _, _, t = x.size()
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)    # (bs, l) -> (bs, 1, l)

        # embedding projection
        if self.embd_fc is not None:
            x, _ = self.embd_fc(x, mask)

        # position encoding
        if self.pe is not None:
            pe = self.pe.to(x.dtype)
            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=t, mode='linear', align_corners=True
                    )[0]
            x = x + pe[..., :t] * mask.to(x.dtype)

        # attention pooling
        if self.attn_pool is not None:
            x, mask = self.attn_pool(x, mask)

        return x, mask


@register_text_net('transformer')
class TextTransformer(nn.Module):
    """
    A backbone with a stack of transformer encoder layers.

    word embeddings
    -> [embedding projection]
    -> [self-attn transformer x L]
    -> latent word embeddings
    """
    def __init__(
        self,
        in_dim,                 # text feature dimension
        embd_dim,               # embedding dimension
        n_heads,                # number of attention heads
        max_seq_len,            # max sequence length
        n_layers=5,             # number of transformer encoder layers
        attn_pdrop=0.0,         # dropout rate for attention map
        proj_pdrop=0.0,         # dropout rate for projection
        path_pdrop=0.0,         # dropout rate for residual paths
        use_abs_pe=True,        # whether to apply absolute position encoding
        use_bkgd_token=True,    # whether to add background token
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        # embedding projection
        self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)

        # position encoding (c, t)
        if use_abs_pe:
            pe = sinusoid_encoding(max_seq_len, embd_dim // 2)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # (optional) background token
        if use_bkgd_token:
            self.bkgd_token = nn.Parameter(torch.empty(embd_dim, 1))
            trunc_normal_(self.bkgd_token, mean=0.0, std=0.02)
        else:
            self.bkgd_token = None

        # self-attention transformers
        self.transformer = nn.ModuleList()
        for _ in range(n_layers):
            self.transformer.append(
                TransformerEncoder(
                    embd_dim,
                    stride=0,   # no conv
                    n_heads=n_heads,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop
                )
            )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        bs, _, t = x.size()
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)     # (bs, l) -> (bs, 1, l)

        # embedding projection
        x, _ = self.embd_fc(x, mask)

        # position encoding
        if self.pe is not None:
            pe = self.pe.to(x.dtype)
            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=t, mode='linear', align_corners=True
                    )[0]
            x = x + pe[..., :t] * mask.to(x.dtype)

        # prepend background token
        if self.bkgd_token is not None:
            bkgd_token = self.bkgd_token.repeat(bs, 1, 1)
            x = torch.cat((bkgd_token, x), dim=-1)
            mask = torch.cat((mask[..., :1], mask), dim=-1)

        # transformer layers
        for transformer in self.transformer:
            x, _ = transformer(x, mask)
            
        return x, mask


def make_text_net(opt):
    opt = deepcopy(opt)
    return backbones[opt.pop('name')](**opt)
