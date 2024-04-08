from copy import deepcopy

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .blocks import MaskedConv1D, LayerNorm, Scale


heads = dict()
def register_head(name):
    def decorator(module):
        heads[name] = module
        return module
    return decorator


@register_head('cls')
class ClsHead(nn.Module):
    """
    1D Conv head for event classification
    """
    def __init__(
        self,
        embd_dim,       # embedding dimension
        n_layers=2,     # number of conv layers
        prior_prob=0.0, # prior probability of positive class
    ):
        super().__init__()

        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                MaskedConv1D(
                    embd_dim, embd_dim,
                    kernel_size=3, stride=1, padding=1, bias=False
                )
            )
            self.norms.append(LayerNorm(embd_dim))

        self.cls_head = MaskedConv1D(
            embd_dim, 1, kernel_size=3, stride=1, padding=1
        )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        bias_init = 0
        assert prior_prob >= 0 and prior_prob < 1
        if prior_prob > 0:
            bias_init = -np.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head.conv.bias, bias_init)

    def forward(self, fpn, fpn_masks):
        out_logits, out_masks = tuple(), tuple()    
        for x, mask in zip(fpn, fpn_masks):
            for conv, norm in zip(self.convs, self.norms):
                x, _ = conv(x, mask)
                x = F.relu(norm(x), inplace=True)
            logits, _ = self.cls_head(x, mask)                  # (bs, 1, p)
            logits, mask = logits.squeeze(1), mask.squeeze(1)   # (bs, p)
            out_logits += (logits, )
            out_masks += (mask, )

        return out_logits, out_masks


@register_head('reg')
class RegHead(nn.Module):
    """
    1D Conv head for offset regression
    """
    def __init__(
        self, 
        embd_dim,       # embedding dimension
        num_fpn_levels, # number of FPN levels
        n_layers=2,     # number of conv layers
    ):
        super().__init__()

        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                MaskedConv1D(
                    embd_dim, embd_dim,
                    kernel_size=3, stride=1, padding=1, bias=False
                )
            )
            self.norms.append(LayerNorm(embd_dim))

        self.reg_head = MaskedConv1D(
            embd_dim, 2, kernel_size=3, stride=1, padding=1
        )
        self.scales = nn.ModuleList([Scale() for _ in range(num_fpn_levels)])

    def forward(self, fpn, fpn_masks):
        out_offsets, out_masks = tuple(), tuple()
        for i, (x, mask) in enumerate(zip(fpn, fpn_masks)):
            for conv, norm in zip(self.convs, self.norms):
                x, _ = conv(x, mask)
                x = F.relu(norm(x), inplace=True)
            offsets, _ = self.reg_head(x, mask)
            offsets = F.relu(self.scales[i](offsets))   # (bs, 2, p)
            offsets = offsets.transpose(1, 2)           # (bs, p, 2)
            mask = mask.squeeze(1)                      # (bs, p)
            out_offsets += (offsets, )
            out_masks += (mask, )
            
        return out_offsets, out_masks


def make_head(opt):
    opt = deepcopy(opt)
    return heads[opt.pop('name')](**opt)