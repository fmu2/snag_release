from copy import deepcopy
import math

import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    sinusoid_encoding, MaskedConv1D, LayerNorm, TransformerEncoder
)


backbones = dict()
def register_video_net(name):
    def decorator(module):
        backbones[name] = module
        return module
    return decorator


@register_video_net('transformer')
class VideoTransformer(nn.Module):
    """
    A backbone that combines convolutions with transformer encoder layers 
    to build a feature pyramid.
    
    video clip features
    -> [embedding convs x L1]
    -> [stem transformer x L2]
    -> [branch transformer x L3]
    -> latent video feature pyramid
    """
    def __init__(
        self,
        in_dim,             # video feature dimension
        embd_dim,           # embedding dimension
        max_seq_len,        # max sequence length
        n_heads,            # number of attention heads for MHA
        mha_win_size,       # local window size for MHA (0 for global attention)
        stride=1,           # conv stride applied to the input features
        arch=(2, 1, 6),     # (#convs, #stem transformers, #branch transformers)
        attn_pdrop=0.0,     # dropout rate for attention maps
        proj_pdrop=0.0,     # dropout rate for projection
        path_pdrop=0.0,     # dropout rate for residual paths
        use_abs_pe=False,   # whether to apply absolute position encoding
    ):
        super().__init__()

        assert len(arch) == 3, '(embed convs, stem, branch)'
        assert stride & (stride - 1) == 0
        assert arch[0] >= int(math.log2(stride))
        self.max_seq_len = max_seq_len

        # embedding projection
        self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)

        # embedding convs
        self.embd_convs = nn.ModuleList()
        self.embd_norms = nn.ModuleList()
        for _ in range(arch[0]):
            self.embd_convs.append(
                MaskedConv1D(
                    embd_dim, embd_dim,
                    kernel_size=5 if stride > 1 else 3,
                    stride=2 if stride > 1 else 1,
                    padding=2 if stride > 1 else 1,
                    bias=False
                )
            )
            self.embd_norms.append(LayerNorm(embd_dim))
            stride = max(stride // 2, 1)

        # position encoding (c, t)
        if use_abs_pe:
            pe = sinusoid_encoding(max_seq_len, embd_dim // 2)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # stem transformers
        self.stem = nn.ModuleList()
        for _ in range(arch[1]):
            self.stem.append(
                TransformerEncoder(
                    embd_dim,
                    stride=1,
                    n_heads=n_heads,
                    window_size=mha_win_size,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop
                )
            )

        # branch transformers (for FPN)
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerEncoder(
                    embd_dim,
                    stride=2 if idx > 0 else 1,
                    n_heads=n_heads,
                    window_size=mha_win_size,
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
        """
        Args:
            x (float tensor, (bs, c1, t1)): video features.
            mask (bool tensor, (bs, t1)): video mask.
        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)    # (bs, l) -> (bs, 1, l)

        # embedding projection
        x, _ = self.embd_fc(x, mask)

        # embedding convs
        for conv, norm in zip(self.embd_convs, self.embd_norms):
            x, mask = conv(x, mask)
            x = F.relu(norm(x), inplace=True)

        # position encoding
        _, _, t = x.size()
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

        # stem layers
        for block in self.stem:
            x, mask = block(x, mask)

        # branch layers
        fpn, fpn_masks = tuple(), tuple()
        for block in self.branch:
            x, mask = block(x, mask)
            fpn += (x, )
            fpn_masks += (mask, )

        return fpn, fpn_masks


def make_video_net(opt):
    opt = deepcopy(opt)
    return backbones[opt.pop('name')](**opt)