from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def masked_avg_pool1d(
    x: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    x_sum = torch.sum(x * mask.to(x.dtype), dim=-1, keepdim=True)
    n = torch.sum(mask, dim=-1, keepdim=True)
    x = x_sum / n
    return x


class MaskedAvgPool1D(nn.Module):
    """
    Masked 1D average pooling
    """
    def __init__(self):
        super(MaskedAvgPool1D, self).__init__()

    def forward(self, x, mask):
        return masked_avg_pool1d(x, mask)


@torch.jit.script
def masked_max_pool1d(
    x: torch.Tensor,
    mask: torch.Tensor,
    kernel_size: int = 3,
    stride: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_min = x.amin(dim=-1, keepdim=True).detach()
    mask_float = mask.to(x.dtype)
    x = x * mask_float + (~mask).to(x.dtype) * x_min
    
    padding = (kernel_size - 1) // 2 
    x = F.max_pool1d(x, kernel_size, stride, padding)
    mask_float = F.max_pool1d(mask_float, kernel_size, stride, padding)
    x = x * mask_float
    mask = mask_float.to(mask.dtype)
    return x, mask

class MaskedMaxPool1D(nn.Module):
    """
    Masked 1D max pooling
    """
    def __init__(self, kernel_size, stride):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, x, mask):
        return masked_max_pool1d(x, mask, self.kernel_size, self.stride)


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=True,
    ):
        super(MaskedConv1D, self).__init__()

        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias
        )
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        assert x.size(-1) % self.stride == 0

        if mask is None:
            mask = torch.ones_like(x[:, :1], dtype=torch.bool)

        mask_float = mask.to(x.dtype)
        x = self.conv(x * mask_float)

        if self.stride > 1:
            mask_float = F.interpolate(
                mask_float, size=x.size(-1), mode='nearest'
            )
            mask = mask_float.bool()
        return x, mask


class LayerNorm(nn.Module):
    """
    LayerNorm that supports input of size (bs, c, t)
    """
    def __init__(self, n_channels, affine=True, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.n_channels = n_channels
        self.eps = eps

        if affine:
            self.weight = nn.Parameter(torch.ones(n_channels, 1))
            self.bias = nn.Parameter(torch.zeros(n_channels, 1))
        else:
            self.weight = self.bias = None

    def forward(self, x):
        x = x - torch.mean(x, dim=1, keepdim=True)
        sigma = torch.mean(x ** 2, dim=1, keepdim=True)
        x = x / torch.sqrt(sigma + self.eps)
        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


def sinusoid_encoding(seq_len, n_freqs):
    """
    Sinusoid position encoding
    """
    tics = torch.arange(seq_len, dtype=torch.float)
    freqs = 10000 ** torch.linspace(0, 1, n_freqs + 1)[:n_freqs]
    x = tics[None, :] / freqs[:, None]
    pe = torch.cat((torch.sin(x), torch.cos(x)))
    return pe


class MaskedMHA(nn.Module):
    """
    Multi Head Attention with mask
    NOTE: This implementation supports
        - global and local self-attention
        - (global) cross-attention

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        embd_dim,           # embedding dimension
        q_dim=None,         # query dimension
        kv_dim=None,        # key / value dimension
        out_dim=None,       # output dimension
        n_heads=4,          # number of attention heads
        window_size=0,      # local attention window size (0 for global attention)
        attn_pdrop=0.0,     # dropout rate for attention map
        proj_pdrop=0.0,     # dropout rate for projection
    ):
        super(MaskedMHA, self).__init__()

        assert embd_dim % n_heads == 0
        self.embd_dim = embd_dim

        if q_dim is None:
            q_dim = embd_dim
        if kv_dim is None:
            kv_dim = embd_dim
        if out_dim is None:
            out_dim = q_dim

        self.n_heads = n_heads
        self.n_channels = embd_dim // n_heads
        self.scale = 1.0 / np.sqrt(np.sqrt(self.n_channels))
        self.out_dim = out_dim

        self.query = nn.Conv1d(q_dim, embd_dim, 1)
        self.key = nn.Conv1d(kv_dim, embd_dim, 1)
        self.value = nn.Conv1d(kv_dim, embd_dim, 1)
        self.proj = nn.Conv1d(embd_dim, out_dim, 1)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # local attention window size
        assert window_size == 0 or window_size % 2 == 1
        self.window_size = window_size
        self.stride = window_size // 2

        # masks for local attention (left / right paddings)
        if window_size > 0:
            l_mask = torch.ones(self.stride, self.stride + 1).tril().flip(dims=(0,))
            r_mask = torch.ones(self.stride, self.stride + 1).tril().flip(dims=(1,))
            self.register_buffer('l_mask', l_mask.bool(), persistent=False)
            self.register_buffer('r_mask', r_mask.bool(), persistent=False)
        else:
            self.l_mask = self.r_mask = None

    def _chunk(self, x, size):
        """
        Convert feature sequence into temporally overlapping chunks.

        Args:
            x (float tensor, (n, t, d)): feature sequence.
            size (int): chunk size.

        Returns:
            x (float tensor, (n, k, s, d)): chunked features.
        """
        n, t, d = x.size()
        assert (t + self.stride - size) % self.stride == 0
        n_chunks = (t + self.stride - size) // self.stride

        chunk_size = (n, n_chunks, size, d)
        chunk_stride = (x.stride(0), self.stride * x.stride(1), *x.stride()[1:])
        x = x.as_strided(size=chunk_size, stride=chunk_stride)
        return x

    def _query_key_matmul(self, q, k):
        """
        Chunk-wise query-key product.

        Args:
            q (float tensor, (n, t, d)): query tensor.
            k (float tensor, (n, t, d)): key tensor.

        Returns:
            attn (float tensor, (n, t, w)): unnormalized attention scores.
        """
        assert q.size() == k.size()
        n, t, _ = q.size()
        w, s = self.window_size, self.stride

        # chunk query and key tensors: (n, t, d) -> (n, t // s - 1, 2s, d)
        q_chunks = self._chunk(q.contiguous(), size=2 * s)
        k_chunks = self._chunk(k.contiguous(), size=2 * s)
        n_chunks = q_chunks.size(1)

        # chunk-wise attention scores: (n, t // s - 1, 2s, 2s)
        chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (q_chunks, k_chunks))

        # shift diagonals into columns: (n, t // s - 1, 2s, w)
        chunk_attn = F.pad(chunk_attn, (0, 0, 0, 1))
        chunk_attn = chunk_attn.view(n, n_chunks, 2 * s, w)

        # fill in the overall attention matrix: (n, t // s, s, w)
        attn = chunk_attn.new_zeros(n, t // s, s, w)
        attn[:, :-1, :, s:] = chunk_attn[:, :, :s, :s + 1]
        attn[:, -1, :, s:] = chunk_attn[:, -1, s:, :s + 1]
        attn[:, 1:, :, :s] = chunk_attn[:, :, -(s + 1):-1, s + 1:]
        attn[:, 0, 1:s, 1:s] = chunk_attn[:, 0, :s - 1, w - (s - 1):]
        attn = attn.view(n, t, w)

        # mask invalid attention scores
        attn[:, :s, :s + 1].masked_fill_(self.l_mask, float('-inf'))
        attn[:, -s:, -(s + 1):].masked_fill_(self.r_mask, float('-inf'))
        return attn

    def _attn_normalize(self, attn, mask):
        """
        Normalize attention scores over valid positions.

        Args:
            attn (float tensor, (bs, h, t, w)): unnormalized attention scores.
            mask (bool tensor, (bs, t, 1)): mask (1 for valid positions).

        Returns:
            attn (float tensor, (bs, h, t, w)): normalized attention map.
        """
        bs, h, t, w = attn.size()

        # inverse mask (0 for valid positions, -inf for invalid ones)
        inv_mask = torch.logical_not(mask)
        inv_mask_float = inv_mask.to(attn.dtype).masked_fill(inv_mask, -1e4)

        # additive attention mask: (bs, t, w)
        attn_mask = self._query_key_matmul(
            torch.ones_like(inv_mask_float), inv_mask_float
        )
        attn += attn_mask.view(bs, 1, t, w)

        # normalize
        attn = F.softmax(attn, dim=-1)

        # if all key / value positions in a local window are invalid
        # (i.e., when the query position is invalid), softmax returns NaN.
        # Replace NaNs with 0
        attn = attn.masked_fill(inv_mask.unsqueeze(1), 0.0)
        return attn

    def _attn_value_matmul(self, attn, v):
        """
        Chunk-wise attention-value product.

        Args:
            attn (float tensor, (n, t, w)): attention map.
            v (float tensor, (n, t, d)): value tensor.

        Returns:
            out (float tensor, (n, t, d)): attention-weighted sum of values.
        """
        n, t, d = v.size()
        w, s = self.window_size, self.stride

        # chunk attention map: (n, t, w) -> (n, t // s, s, w)
        attn_chunks = attn.view(n, t // s, s, w)

        # shift columns into diagonals: (n, t // s, s, 3s)
        attn_chunks = F.pad(attn_chunks, (0, s))
        attn_chunks = attn_chunks.view(n, t // s, -1)[..., :-s]
        attn_chunks = attn_chunks.view(n, t // s, s, 3 * s)

        # chunk value tensor: (n, t + 2s, d) -> (n, t // s, 3s, d)
        v = F.pad(v, (0, 0, s, s))
        v_chunks = self._chunk(v.contiguous(), size=3 * s)

        # chunk-wise attention-weighted sum: (n, t // s, s, d)
        out = torch.einsum('bcwd,bcdh->bcwh', (attn_chunks, v_chunks))
        out = out.view(n, t, d)
        return out

    def forward(self, q, k=None, v=None, kv_mask=None, kv_size=None):
        """
        Args:
            q (float tensor, (bs, c, t1)): query feature sequence.
            k (float tensor, (bs, c, t2)): key feature sequence.
            v (float tensor, (bs, c, t2)): value feature sequence.
            kv_mask (bool tensor, (bs, 1, t2)): key / value mask.
            kv_size (int tensor, (bs,)): number of times to repeat each sample.
        """
        bs, c = q.size(0), self.embd_dim
        h, d, w = self.n_heads, self.n_channels, self.window_size

        if k is None:
            k = q
        if v is None:
            v = k

        # if mask is not given, assume all positions are valid
        if kv_mask is None:
            kv_mask = torch.ones_like(k[:, :1], dtype=torch.bool)

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # repeat query to match the size of key / value
        if kv_size is not None and k.size(0) != bs:
            q = q.repeat_interleave(kv_size, dim=0)
            bs = q.size(0)

        if self.window_size > 0:
            q = q.view(bs, h, d, -1).flatten(0, 1).transpose(1, 2)
            k = k.view(bs, h, d, -1).flatten(0, 1).transpose(1, 2)
            v = v.view(bs, h, d, -1).flatten(0, 1).transpose(1, 2)

            # attention scores: (bs * h, t, w)
            attn = self._query_key_matmul(q * self.scale, k * self.scale)
            attn = attn.view(bs, h, -1, w)

            # normalized attention map: (bs, h, t, w)
            attn = self._attn_normalize(attn, kv_mask.transpose(1, 2))
            attn = self.attn_drop(attn)
            attn = attn.view(bs * h, -1, w)

            # attention-weighted sum of values: # (bs * h, t, d)
            q = self._attn_value_matmul(attn, v)
            q = q.view(bs, h, -1, d)
        else:
            q = q.view(bs, h, d, -1).transpose(2, 3)
            k = k.view(bs, h, d, -1)
            v = v.view(bs, h, d, -1).transpose(2, 3)

            attn = (q * self.scale) @ (k * self.scale)      # (bs, h, t1, t2)
            attn = attn.masked_fill(
                mask=torch.logical_not(kv_mask[:, :, None, :]),
                value=float('-inf'),
            )
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            q = attn @ v                                    # (bs, h, t1, d)

        q = q.transpose(2, 3).reshape(bs, c, -1)            # (bs, c, t1)
        out = self.proj_drop(self.proj(q))
        return out


class AttNPool1D(nn.Module):

    def __init__(self, embd_dim, n_heads=4):
        super(AttNPool1D, self).__init__()

        self.pool = MaskedAvgPool1D()
        self.attn = MaskedMHA(embd_dim, n_heads=n_heads)

    def forward(self, x, mask):
        x_mean = self.pool(x, mask)
        h = torch.cat((x_mean, x), dim=-1)
        mask = torch.cat((mask[..., :1], mask), dim=-1)

        pool = self.attn(h, kv_mask=mask)[..., :1]
        x = torch.cat((pool, x), dim=-1)
        return x, mask


class ConvAttNLayer(nn.Module):
    """
    Multi Head Conv Self Attention with mask

    With current implementation, the downpsampled features will be aligned with
    every s+1 time steps, where s is the down-sampling stride. This allows us
    to easily interpolate the corresponding position encoding.
    """
    def __init__(
        self,
        embd_dim,           # embedding dimension
        out_dim=None,       # output dimension
        stride=1,           # convolution stride
        n_heads=4,          # number of attention heads
        window_size=0,      # window size for local attention
        attn_pdrop=0.0,     # dropout rate for attention map
        proj_pdrop=0.0,     # dropout rate for projection
    ):
        super(ConvAttNLayer, self).__init__()

        self.use_conv = stride > 0
        if self.use_conv:
            assert stride == 1 or stride % 2 == 0
            self.q_conv = MaskedConv1D(
                embd_dim, embd_dim, 3, stride, 1, groups=embd_dim, bias=False
            )
            self.k_conv = MaskedConv1D(
                embd_dim, embd_dim, 3, stride, 1, groups=embd_dim, bias=False
            )
            self.v_conv = MaskedConv1D(
                embd_dim, embd_dim, 3, stride, 1, groups=embd_dim, bias=False
            )
            self.q_norm = LayerNorm(embd_dim)
            self.k_norm = LayerNorm(embd_dim)
            self.v_norm = LayerNorm(embd_dim)
        else:
            self.q_conv = self.q_norm = None
            self.k_conv = self.k_norm = None
            self.v_conv = self.v_norm = None

        # self-attention
        self.attn = MaskedMHA(
            embd_dim,
            out_dim=out_dim or embd_dim,
            n_heads=n_heads, window_size=window_size,
            attn_pdrop=attn_pdrop, proj_pdrop=proj_pdrop,
        )

    def forward(self, x, mask):
        if self.use_conv:
            k, _ = self.k_conv(x, mask)
            v, _ = self.v_conv(x, mask)
            q, mask = self.q_conv(x, mask)
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = self.v_norm(v)
        else:
            q = k = v = x
        out = self.attn(q, k, v, mask)
        return out, mask


class ConvXAttNLayer(nn.Module):
    """
    Multi Head Conv Cross Attention with mask

    With current implementation, the downpsampled features will be aligned with
    every s+1 time steps, where s is the down-sampling stride. This allows us
    to easily interpolate the corresponding position encoding.
    """
    def __init__(
        self,
        embd_dim,           # embedding dimension
        kv_dim,             # key / value dimension
        out_dim=None,       # output dimension
        stride=1,           # convolution stride
        n_heads=4,          # number of attention heads
        attn_pdrop=0.0,     # dropout rate for attention map
        proj_pdrop=0.0,     # dropout rate for projection
    ):
        super(ConvXAttNLayer, self).__init__()

        self.use_conv = stride > 0
        if self.use_conv:
            assert stride == 1 or stride % 2 == 0
            self.q_conv = MaskedConv1D(
                embd_dim, embd_dim, 3, stride, 1, groups=embd_dim, bias=False
            )
            self.q_norm = LayerNorm(embd_dim)
        else:
            self.q_conv = self.q_norm = None

        # cross-attention
        self.xattn = MaskedMHA(
            embd_dim,
            kv_dim=kv_dim, out_dim=out_dim or embd_dim,
            n_heads=n_heads, attn_pdrop=attn_pdrop, proj_pdrop=proj_pdrop,
        )

    def forward(self, q, q_mask, kv, kv_mask, kv_size=None):
        if self.use_conv:
            q, q_mask = self.q_conv(q, q_mask)
            q = self.q_norm(q)
        out = self.xattn(q, kv, None, kv_mask, kv_size)
        if kv_size is not None and out.size(0) != q_mask.size(0):
            q_mask = q_mask.repeat_interleave(kv_size, dim=0)
        return out, q_mask


class FFN(nn.Module):
    """
    Feed Forward Network (MLP) in Transformer.
    """
    def __init__(self, channels, expansion=4, pdrop=0.0):
        super(FFN, self).__init__()

        self.fc = nn.Conv1d(channels, channels * expansion, 1)
        self.actv = nn.GELU()
        self.proj = nn.Conv1d(channels * expansion, channels, 1)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x):
        x = self.dropout(self.actv(self.fc(x)))
        x = self.dropout(self.proj(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder.
    (optional depth-wise conv -> local self-attn -> FFN)
    """
    def __init__(
        self,
        embd_dim,           # embedding dimension
        stride=1,           # convolution stride (0 if disable convs)
        n_heads=4,          # number of attention heads
        window_size=0,      # MHA window size (0 for global attention)
        expansion=4,        # expansion factor for FFN
        attn_pdrop=0.0,     # dropout rate for attention map
        proj_pdrop=0.0,     # dropout rate for projection
        path_pdrop=0.0,     # dropout rate for residual paths
    ):
        super(TransformerEncoder, self).__init__()

        # self-attention
        self.attn = ConvAttNLayer(
            embd_dim,
            stride=stride, n_heads=n_heads, window_size=window_size,
            attn_pdrop=attn_pdrop, proj_pdrop=proj_pdrop,
        )
        self.ln_attn = LayerNorm(embd_dim)
        self.drop_path_attn = LayerScale(embd_dim, path_pdrop)

        if stride > 1:
            self.attn_skip = MaskedMaxPool1D(3, stride=stride)
        else:
            self.attn_skip = None

        # FFN
        self.ffn = FFN(embd_dim, expansion, proj_pdrop)
        self.ln_ffn = LayerNorm(embd_dim)
        self.drop_path_ffn = LayerScale(embd_dim, path_pdrop)

    def forward(self, x, mask):
        if mask is None:
            mask = torch.ones_like(x[:, :1], dtype=torch.bool)
        x = x * mask.to(x.dtype)

        # local self-attention (optionally with depth-wise conv)
        skip = self.attn_skip(x, mask)[0] if self.attn_skip is not None else x
        h, mask = self.attn(self.ln_attn(x), mask)
        x = skip * mask.to(x.dtype) + self.drop_path_attn(h)

        # FFN
        h = self.ffn(self.ln_ffn(x)) * mask.to(x.dtype)
        x = x + self.drop_path_ffn(h)
        return x, mask


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder (w/o self-attention).
    (optional depth-wise conv -> xattn -> FFN)
    """
    def __init__(
        self,
        embd_dim,           # embedding dimension
        kv_dim,             # key / value dimension
        n_heads=4,          # number of attention heads
        expansion=4,        # expansion factor for FFN
        attn_pdrop=0.0,     # dropout rate for attention map
        proj_pdrop=0.0,     # dropout rate for projection
        path_pdrop=0.0,     # dropout rate for residual paths
        xattn_mode='adaln', # cross-attention mode (affine | adaln)
    ):
        super(TransformerDecoder, self).__init__()

        # cross-attention
        assert xattn_mode in ('affine', 'adaln')
        self.xattn = ConvXAttNLayer(
            embd_dim, kv_dim, embd_dim * 2,
            stride=1, n_heads=n_heads,
            attn_pdrop=attn_pdrop, proj_pdrop=proj_pdrop,
        )
        self.ln_xattn_q = LayerNorm(embd_dim)
        self.ln_xattn_kv = LayerNorm(kv_dim)

        if xattn_mode == 'adaln':
            self.adaln = LayerNorm(embd_dim, affine=False)
        else:
            self.adaln = nn.Identity()

        # FFN
        self.ffn = FFN(embd_dim, expansion, proj_pdrop)
        self.ln_ffn = LayerNorm(embd_dim)
        self.drop_path_ffn = LayerScale(embd_dim, path_pdrop)

    def forward(self, q, q_mask, kv, kv_mask, kv_size=None):
        if q_mask is None:
            q_mask = torch.ones_like(q[:, :1], dtype=torch.bool)
        q = q * q_mask.to(q.dtype)

        # cross-attention (optionally with depth-wise conv)
        h, q_mask = self.xattn(
            self.ln_xattn_q(q), q_mask, self.ln_xattn_kv(kv), kv_mask, kv_size
        )
        if kv_size is not None and q.size(0) != h.size(0):
            q = q.repeat_interleave(kv_size, dim=0)
        q = self.adaln(q * q_mask.to(q.dtype))
        scale, shift = h.chunk(2, dim=1)
        q = q * scale + shift

        # FFN
        h = self.ffn(self.ln_ffn(q)) * q_mask.to(q.dtype)
        q = q + self.drop_path_ffn(h)
        return q, q_mask


class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init=1.0):
        """
        init_value : initial value for the scalar
        """
        super(Scale, self).__init__()

        self.scale = nn.Parameter(torch.as_tensor(init, dtype=torch.float))

    def forward(self, x):
        return x * self.scale.to(x.dtype)


class LayerScale(nn.Module):
    """
    Multiple residual by a per-channel scaling factor (and zero init) before adding.
    https://arxiv.org/abs/2103.17239
    """
    def __init__(self, n_channels, pdrop=0.0, init_scale=1e-4):
        super(LayerScale, self).__init__()

        self.scale = nn.Parameter(init_scale * torch.ones((1, n_channels, 1)))
        self.pdrop = pdrop

    def forward(self, x):
        return drop_path(self.scale.to(x.dtype) * x, self.pdrop, self.training)


def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    x = x.div(keep_prob) * mask.floor_()
    return x