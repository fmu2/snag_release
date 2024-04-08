from copy import deepcopy

import torch.nn as nn

from .blocks import LayerNorm, TransformerDecoder


modules = dict()
def register_fusion(name):
    def decorator(module):
        modules[name] = module
        return module
    return decorator


@register_fusion('xattn')
class XAttNFusion(nn.Module):
    """ Fuse video and text features using attention.
    """

    def __init__(
        self,
        vid_dim,            # video feature dimension
        text_dim,           # text feature dimension
        n_layers=2,         # number of fusion layers
        n_heads=4,          # number of attention heads for MHA
        attn_pdrop=0.0,     # dropout rate for attention maps
        proj_pdrop=0.0,     # dropout rate for projection
        path_pdrop=0.0,     # dropout rate for residual paths
        xattn_mode='adaln', # cross-attention mode (adaln | affine)
    ):
        super(XAttNFusion, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                TransformerDecoder(
                    vid_dim, text_dim, 
                    n_heads=n_heads, 
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    xattn_mode=xattn_mode,
                )
            )

        self.ln_out = LayerNorm(vid_dim)

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward(self, q, q_mask, kv, kv_mask, kv_size=None):
        for layer in self.layers:
            q, q_mask = layer(q, q_mask, kv, kv_mask, kv_size)
        q = self.ln_out(q)

        # repeat query to match the size of key / value
        if kv_size is not None and q.size(0) != kv.size(0):
            q = q.repeat_interleave(kv_size, dim=0)
            q_mask = q_mask.repeat_interleave(kv_size, dim=0)

        return q, q_mask

    def forward(self, vid, vid_masks, text, text_mask, text_size=None):
        if not isinstance(vid, tuple):
            return self._forward(vid, vid_masks, text, text_mask, text_size)
            
        out, out_masks = tuple(), tuple()
        for x, mask in zip(vid, vid_masks):
            x, mask = self._forward(x, mask, text, text_mask, text_size)
            out += (x, )
            out_masks += (mask, )

        return out, out_masks


def make_fusion(opt):
    opt = deepcopy(opt)
    return modules[opt.pop('name')](**opt)