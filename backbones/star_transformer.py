# --------------------------------------------------------
# Star Transformer
# Copyright (c) 2021 Khalifa University
# Written by Alabi Bojesomo
# --------------------------------------------------------
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .layers import HyperLinear, HyperSoftmax, Concatenate, HyperConv2d, multiply, dot_product
from functools import partial


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., n_divs=1):
        assert in_features % n_divs == 0, f'in_features [{in_features}] is not divisible by n_divs [{n_divs}]'
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % n_divs == 0, \
            f'hidden_features [{hidden_features}] is not divisible by n_divs [{n_divs}]'
        assert out_features % n_divs == 0, f'out_features [{out_features}] is not divisible by n_divs [{n_divs}]'

        Linear = nn.Linear if n_divs == 1 else HyperLinear

        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = Linear(in_features, hidden_features, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.act = act_layer()
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = Linear(hidden_features, out_features, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def pad_terms(H, W, window_size=1):
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    return pad_r, pad_b


def pad_input(x, window_size=1):
    _, H, W, _ = x.shape
    # pad feature maps to multiples of window size
    pad_l = pad_t = 0
    # pad_r = (window_size - W % window_size) % window_size
    # pad_b = (window_size - H % window_size) % window_size
    pad_r, pad_b = pad_terms(H, W, window_size)
    x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
    # _, Hp, Wp, _ = x.shape
    return  x


def window_partition(x, direction, window_size=1):
    """
    Args:
        x: (B, H, W, C)
        direction (int): window direction (one of 0, 1, 2, 3 representing North, NorthEast, East, and SouthEast)

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    assert direction in range(4)
    B, H, W, C = x.shape
    if direction == 0:  # 0 -> North
        for i in range(H):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=(i + 1), dims=1)
        # windows = rearrange(x, 'b h (s w) c -> (b w) h s c', s=window_size)
        windows = rearrange(x, 'b h (s w) c -> (b w) s h c', s=window_size)
    elif direction == 1:  # 1 -> NorthEast
        for i in range(H):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=-(i + 1), dims=1)
        # windows = rearrange(x, 'b h (s w) c -> (b w) h s c', s=window_size)
        windows = rearrange(x, 'b h (s w) c -> (b w) s h c', s=window_size)
    elif direction == 2:  # 2 -> East
        windows = rearrange(x, 'b (s h) w c -> (b h) s w c', s=window_size)
    else:  # 3 -> SouthEast
        # windows = rearrange(x, 'b h (s w) c -> (b w) h s c', s=window_size)
        windows = rearrange(x, 'b h (s w) c -> (b w) s h c', s=window_size)

    return windows


def window_reverse(windows, direction, H, W, Hp, Wp, window_size=1):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        direction (int): window direction (one of 0, 1, 2, 3 representing North, NorthEast, East, and SouthEast)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    assert direction in range(4)

    if direction == 0:  # 0 -> North
        # x = rearrange(windows, '(b w) h s c -> b h (s w) c', w=Wp // window_size, s=window_size)
        x = rearrange(windows, '(b w) s h c -> b h (s w) c', w=Wp // window_size)
        for i in range(H):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=-(i + 1), dims=1)
    elif direction == 1:  # 1 -> NorthEast
        # x = rearrange(windows, '(b w) h s c -> b h (s w) c', w=Wp // window_size, s=window_size)
        x = rearrange(windows, '(b w) s h c -> b h (s w) c', w=Wp // window_size)
        for i in range(H):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=(i + 1), dims=1)
    elif direction == 2:  # 2 -> East
        x = rearrange(windows, '(b h) s w c -> b (s h) w c', h=Hp // window_size)
    else:  # 3 -> SouthEast
        # x = rearrange(windows, '(b w) h s c -> b h (s w) c', w=Wp // window_size, s=window_size)
        x = rearrange(windows, '(b w) s h c -> b h (s w) c', w=Wp // window_size)

    pad_r, pad_b = pad_terms(H, W, window_size)
    if pad_r > 0 or pad_b > 0:
        x = x[:, :H, :W, :].contiguous()

    return x


class StarAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (int): The width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim,
                 window_size=1,
                 num_heads=1,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_divs=1):
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # sw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.n_divs = n_divs

        grp_dim = head_dim // 4
        kernel_size = (max(1, min(3, self.window_size)), 3)
        padding = tuple([ks // 2 for ks in kernel_size])
        self.lepe = nn.Conv2d(grp_dim, grp_dim, kernel_size=kernel_size,
                              padding=padding,
                              groups=grp_dim)

        Linear = nn.Linear if n_divs == 1 else HyperLinear
        # self.qkv = Linear(dim, dim * 3, bias=qkv_bias, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.q = Linear(dim, dim, bias=qkv_bias, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.kv = Linear(dim, dim * 2, bias=qkv_bias, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)  # the dot product of two hypercomplex returns real, not hypercomplex

    def forward(self, x, y=None):
        """ Forward function.

        Args:
            x: input features with shape of (B, H, W, C)       (num_windows*B, N, C)
            y: input features with shape of (B, H, W, C)     {if None -> self attention}
        """
        B, H, W, C = x.shape

        if y is None:  # self attention
            y = x

        x = pad_input(x, self.window_size)
        y = pad_input(y, self.window_size)
        _, Hp, Wp, _ = x.shape

        q = self.q(x)
        kv = self.kv(y)

        q = rearrange(q, 'b h w (k c) -> k b h w c', k=4)
        kv = rearrange(kv, 'b h w (t k c) -> t k b h w c', t=2, k=4)

        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale

        xs = []
        for i in range(4):  # attn_, v_ in zip(attn, v_windows):

            # q_ = rearrange(window_partition(q[i], i, self.window_size), 'b w (n c) -> b n w c', n=self.num_heads)
            # k_ = rearrange(window_partition(k[i], i, self.window_size), 'b w (n c) -> b n w c', n=self.num_heads)
            # v_ = rearrange(window_partition(v[i], i, self.window_size), 'b w (n c) -> b n w c', n=self.num_heads)
            q_ = rearrange(window_partition(q[i], i, self.window_size), 'b h w (n c) -> (b n) h w c', n=self.num_heads)
            k_ = rearrange(window_partition(k[i], i, self.window_size), 'b h w (n c) -> (b n) h w c', n=self.num_heads)
            v_ = rearrange(window_partition(v[i], i, self.window_size), 'b h w (n c) -> (b n) h w c', n=self.num_heads)
            attn_ = (q_ @ k_.transpose(-2, -1))
            attn_ = self.attn_drop(attn_)

            attn_ = attn_.type_as(v_)
            x = torch.einsum('bhij,bhjd->bhid', attn_, v_)
            # v_ = self.lepe(rearrange(v_, 'b h w c -> (b h) c () w'))
            v_ = self.lepe(rearrange(v_, 'b h w c -> b c h w'))
            # x = x + rearrange(v_, '(b h) c () w -> b h w c', h=self.num_heads)
            x = x + rearrange(v_, 'b c h w -> b h w c')
            # x = rearrange(x, 'b h w c -> b w (h c)')
            # xs.append(window_reverse(x, i, H=H, W=W))
            xs.append(window_reverse(x, i, H=H, W=W, Hp=Hp, Wp=Wp, window_size=self.window_size))
        x = torch.stack(xs)  # .transpose(0, 1)
        # x = rearrange(x, 'k b h w c -> b h w (k c)')
        x = rearrange(x, 'k (b n) h w c -> b h w (k n c)', n=self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class StarTransformerBlock(nn.Module):
    """ Star Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size. Default: 1.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim,
                 num_heads=1, window_size=1,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_divs=1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = StarAttention(dim, window_size=window_size,
                                  num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, proj_drop=drop, n_divs=n_divs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, n_divs=n_divs)

        self.H = None
        self.W = None

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H, W, C)
        """
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        # x_windows = [window_partition(x, i) for i in range(4)]

        # W-MSA
        if self.use_checkpoint:
            attn_windows = checkpoint.checkpoint(self.attn, x)
        else:
            attn_windows = self.attn(x)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm, n_divs=1):
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        super().__init__()
        self.dim = dim
        Linear = nn.Linear if n_divs == 1 else HyperLinear
        self.reduction = Linear(4 * dim, 2 * dim, bias=False, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.norm = norm_layer(4 * dim)
        self.concat = Concatenate(dim=-1, n_divs=n_divs)
        # self.H, self.W = None, None

    def forward(self, x):  # , H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H, W, C).
            H, W: Spatial resolution of the input feature.
        """
        # H, W = self.H, self.W
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        #
        # x = x.view(B, H, W, C)
        B, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x = rearrange(x, 'b (h p1) (w p2) c -> b h w (p2 p1 c)', p1=2, p2=2)

        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    """ Patch Expanding Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm, ratio=2, factor=1, n_divs=1):
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        super().__init__()
        self.dim = dim
        self.ratio = ratio
        self.factor = factor
        Linear = nn.Linear if n_divs == 1 else HyperLinear
        self.expansion = Linear(dim, self.factor * self.ratio * dim, bias=False, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.norm = norm_layer(dim)
        # self.concat = Concatenate(dim=-1, n_divs=n_divs)
        # self.H, self.W = None, None

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H, W, C).
        """
        # H, W = self.H, self.W
        #
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        B, H, W, C = x.shape
        x = self.norm(x)
        x = self.expansion(x)

        # padding
        # pad_input = (H % self.ratio == 1) or (W % self.ratio == 1)
        # if pad_input:
        #     x = F.pad(x, (0, 0, 0, W % self.ratio, 0, H % self.ratio))

        # x = x.view(B, H*self.ratio, W*self.ratio, C//self.ratio * self.factor)

        x = rearrange(x, 'b h w (p2 p1 c) -> b (h p1) (w p2) c', p1=self.ratio, p2=self.ratio)

        # x = x.view(B, -1, C//self.ratio * self.factor)  # B H*2*W*2 C/2

        return x


class BasicLayer(nn.Module):
    """ A basic Star Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 1.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=1,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 n_divs=1):
        super().__init__()
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size

        # build blocks
        self.blocks = nn.ModuleList([
            StarTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                n_divs=n_divs,
                use_checkpoint=use_checkpoint
            )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, n_divs=n_divs)
        else:
            self.downsample = None

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H, W, C).
        """
        B, H, W, C = x.shape

        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            self.downsample.H, self.downsample.W = H, W
            x_down = self.downsample(x)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, x_down
        else:
            return x, x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, n_divs=1):
        super().__init__()
        assert in_chans % n_divs == 0, f'in_chans [{in_chans}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % n_divs == 0, f'embed_dim [{embed_dim}] is not divisible by n_divs [{n_divs}]'
        # assert embed_dim % 3 == 0, f'embed_dim [{embed_dim}] is not divisible by 3 (self_attention Q, K, V)'
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        Conv2d = nn.Conv2d if n_divs == 1 else HyperConv2d
        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
                           **{'n_divs': n_divs for n in [n_divs] if n > 1})
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class StarTransformer(nn.Module):
    """ Star Transformer backbone.
        A PyTorch impl of : `Star Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Star Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 1.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=1,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 use_checkpoint=False,
                 n_divs=1):
        super().__init__()

        assert in_chans % n_divs == 0, f'in_chans [{in_chans}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % n_divs == 0, f'embed_dim [{embed_dim}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % num_heads[0] == 0, f'embed_dim [{embed_dim}] is not divisible by num_head[0]=={num_heads[0]}'

        self.img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            n_divs=n_divs
        )

        # absolute position embedding
        if self.ape:
            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                n_divs=n_divs
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        assert np.all(k % v for k, v in zip(num_features, num_heads)), 'embed_dim/ num_features not a multiple of heads'
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def init_weights(self):
        """Initialize the weights in backbone.

        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            # x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
            x = rearrange(x + absolute_pos_embed, 'b c h w -> b h w c')
        else:
            # x = x.flatten(2).transpose(1, 2)
            x = rearrange(x, 'b c h w -> b h w c')
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            # layer.H, layer.W = Wh, Ww
            x_out, x = layer(x)
            # H, W, Wh, Ww = layer.H, layer.W, layer.Wh, layer.Ww

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = rearrange(x_out, 'b h w c -> b c h w')
                # out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)


class StarTransformerDecode(nn.Module):
    """ Star Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size. Default: 1.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim,
                 num_heads=1, window_size=1,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_divs=1, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = StarAttention(dim, window_size=window_size,
                                  num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, proj_drop=drop, n_divs=n_divs)

        self.norm_mixed = norm_layer(dim)
        self.attn_mixed = StarAttention(dim, window_size=window_size,
                                  num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, proj_drop=drop, n_divs=n_divs)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       n_divs=n_divs)

    def self_attn(self, x):
        # W-MSA
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.attn, x)
        else:
            x = self.attn(x)

        return x

    def mixed_attn(self, x, y):
        # W-MSA
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.attn_mixed, x, y)
        else:
            x = self.attn_mixed(x, y)

        return x

    def forward(self, x, y):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H, W, C).
            y: Skip feature, tensor size (B, H1, W1, C/2). H1=2*H
        """
        shortcut = x
        x = self.norm1(x)
        # self attention
        x = self.self_attn(x)
        x = shortcut + self.drop_path(x)

        # attention with skip connection
        shortcut = x
        # x = self.norm_mixed(x)
        x = self.norm_mixed(x)  # TODO - modified correctly on 7th July 2021
        x = self.mixed_attn(x, y)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class EncodeLayer(nn.Module):
    """ A basic Star Transformer layer for one stage Encoder.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 1.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the beginning of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads=1,
                 window_size=1,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 n_divs=1):
        super().__init__()
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim//2, norm_layer=norm_layer, n_divs=n_divs)
        else:
            self.downsample = None

        # build blocks
        self.blocks = nn.ModuleList([
            StarTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                n_divs=n_divs,
                use_checkpoint=use_checkpoint
            )
            for i in range(depth)])

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H, W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, H, W, C = x.shape
        if self.downsample is not None:
            x = self.downsample(x)
            H, W = (H + 1) // 2, (W + 1) // 2

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        return x


class DecodeLayer(nn.Module):
    """ A basic Star Transformer layer for one stage Encoder.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 1.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the beginning of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=1,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=None,
                 use_checkpoint=False,
                 n_divs=1):
        super().__init__()
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch expanding layer
        if upsample is not None:
            self.upsample = upsample(dim=dim*2, norm_layer=norm_layer, n_divs=n_divs)
        else:
            self.upsample = None

        self.blocks = nn.ModuleList()
        for i in range(depth):
            layer = StarTransformerDecode(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                n_divs=n_divs,
                use_checkpoint=use_checkpoint
            )
            self.blocks.append(layer)

    def forward(self, x, y):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H, W, C).
            y: skip feature, tensor size (B, H1, W1, C).
            H, W: Spatial resolution of the input feature.
            H1, W1: Spatial resolution of the skip feature.
        """
        B, H, W, _ = x.shape
        B, H1, W1, _ = y.shape
        if self.upsample is not None:
            x = self.upsample(x)
            H, W = H * 2, W * 2

        if H1 != H or W1 != W:
            x = x.view(B, H, W, -1)
            x = x[:, :H1, :W1, :].contiguous()
            x = x.view(B, H1 * W1, -1)
            H, W = H1, W1

        for i, blk in enumerate(self.blocks):
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, y)
            else:
                x = blk(x, y)
        return x


class DecodeLayerUnet(nn.Module):
    """ A basic Star Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 1.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=1,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=None,
                 use_checkpoint=False,
                 n_divs=1,
                 merge_type='concat'):
        super().__init__()
        assert merge_type in ['concat', 'add']

        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        self.upsample = upsample
        if self.upsample is not None:
            self.upsample = upsample(dim=dim*2, norm_layer=norm_layer, n_divs=n_divs)

        self.merge_type = merge_type
        if self.merge_type == 'concat':
            self.concat = Concatenate(dim=-1, n_divs=n_divs)
            Linear = nn.Linear if n_divs == 1 else HyperLinear
            self.fc = Linear(2*dim, dim, **{'n_divs': n_divs for n in [n_divs] if n > 1})

        # build blocks
        self.blocks = nn.ModuleList([
            StarTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                n_divs=n_divs,
                use_checkpoint=use_checkpoint
            )
            for i in range(depth)])

    def forward(self, x, y):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H, W, C).
            y: Input feature, tensor size (B, H1, W1, C).
        """
        if self.upsample is not None:
            x = self.upsample(x)

        B, H, W, C = x.shape
        _, H1, W1, _ = y.shape

        if H1 != H or W1 != W:
            x = x[:, :H1, :W1, :].contiguous()

        if self.merge_type == 'concat':
            x = self.concat([x, y])
            x = self.fc(x)
        else:
            x = x + y

        for blk in self.blocks:
            x = blk(x)
        return x


class Head(nn.Module):
    def __init__(self, decode_dim=96, patch_size=4, out_chans=None, img_size=(224,), n_divs=1):
        super().__init__()
        assert decode_dim % n_divs == 0, f'dim [{decode_dim}] is not divisible by n_divs [{n_divs}]'
        self.expand = PatchExpanding(decode_dim, ratio=patch_size, factor=patch_size, n_divs=n_divs)
        self.final = nn.Sequential(
            nn.Linear(in_features=decode_dim, out_features=out_chans if out_chans else decode_dim),
            nn.Sigmoid()
        )
        self.out_chans = out_chans if out_chans else decode_dim
        self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        # self.H, self.W = None, None

    def forward(self, x):  # , H, W):
        # H, W = self.H, self.W
        # B, _, _ = x.shape
        # H1, W1 = self.img_size
        # self.expand.H, self.expand.W = H, W

        H1, W1 = self.img_size
        _, H, W, _ = x.shape

        x = self.expand(x)
        H, W = H * self.patch_size, W * self.patch_size
        if H1 != H or W1 != W:
            # x = x.view(B, H, W, -1)
            x = x[:, :H1, :W1, :].contiguous()
            # x = x.view(B, H1 * W1, -1)
        # print(x.shape)
        x = self.final(x)
        return x


class LearnVectorBlock(nn.Module):
    def __init__(self, in_channels, featmaps, filter_size, act_layer=nn.GELU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = featmaps

        padding = (filter_size[0] // 2, filter_size[0] // 2)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=featmaps, kernel_size=filter_size, padding=padding),
            act_layer()
        )

    def forward(self, x):
        x = self.layer(x)
        return x

    def extra_repr(self) -> str:
        extra_str = f"in_channels={self.in_channels}, out_channels={self.out_channels}, activation={self.activation}"
        return extra_str


class StarEncoderDecoderTransformer(nn.Module):
    """ Star Transformer backbone.
        A PyTorch impl of :
    Args:
        img_size (int/tuple): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Star Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Local window size. Default: 1.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 img_size=(224,),
                 patch_size=4,
                 in_chans=3,
                 out_chans=None,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=1,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 n_divs=1):
        super().__init__()

        # assert in_chans % n_divs == 0, f'in_chans [{in_chans}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % n_divs == 0, f'embed_dim [{embed_dim}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % (4 * num_heads[
            0]) == 0, f'embed_dim [{embed_dim}] is not divisible by 4*num_head[0]=={4 * num_heads[0]}'

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm

        n_chans = int(np.ceil(in_chans / n_divs) * n_divs)
        self.learn_vector = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=n_chans, kernel_size=3, padding=1),
            nn.GELU()
        ) if n_chans != in_chans else nn.Identity()

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=n_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            n_divs=n_divs
        )

        # absolute position embedding
        if self.ape:
            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) * 2)]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            layer = EncodeLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
                n_divs=n_divs
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        # assert np.all(k % v for k, v in zip(num_features, num_heads)), 'embed_dim/ num_features not a multiple of heads'
        self.num_features = num_features

        # build up layers
        self.uplayers = nn.ModuleList()
        self.decode_dim = num_features  # [::-1]
        for i_layer in range(self.num_layers - 1, -1, -1):
            drop_path = dpr[sum(depths[(1 + i_layer):])+sum(depths):sum(depths[(i_layer):])+sum(depths)]
            decode_dim = self.decode_dim[i_layer]
            layer = DecodeLayer(
                dim=decode_dim,
                depth=depths[i_layer],  # 1,  #depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer=norm_layer,
                upsample=PatchExpanding if i_layer != (self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                n_divs=n_divs
            )
            self.uplayers.append(layer)

        self.head = Head(decode_dim=decode_dim, patch_size=self.patch_size, out_chans=out_chans, n_divs=n_divs,
                         img_size=self.img_size)

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        x = self.learn_vector(x)
        x = self.patch_embed(x)

        H, W = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(H, W), mode='bicubic')
            # x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
            x = rearrange(x + absolute_pos_embed, 'b c h w -> b h w c')
        else:
            # x = x.flatten(2).transpose(1, 2)
            x = rearrange(x, 'b c h w -> b h w c')

        x = self.pos_drop(x)

        features = []
        # hw = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            # layer.H, layer.W = H, W
            x = layer(x)
            # H, W = layer.H, layer.W
            # print(i, H, W, x.shape)
            features.insert(0, x)
            # hw.insert(0, (H, W))

        for i in range(self.num_layers):
            layer = self.uplayers[i]
            y = features[i]
            # H1, W1 = hw[i]
            # layer.H, layer.W, layer.H1, layer.W1 = H, W, H1, W1
            x = layer(x, y)
            # H, W = layer.H, layer.W
        # self.head.H, self.head.W = H, W
        x = self.head(x)
        # H1, W1 = to_2tuple(self.img_size)
        # x = x.view(-1, H1, W1, self.head.out_chans).permute(0, 3, 1, 2).contiguous()
        x = rearrange(x, 'b h w c -> b c h w')

        return x


class StarUNet(nn.Module):
    """ Star Transformer backbone.
        A PyTorch impl of :
    Args:
        img_size (int/tuple): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Star Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_sizes (tuple[int]): Local window size. Default: 1.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self,
                 img_size=(256, 256),
                 patch_size=4,
                 in_chans=3,
                 out_chans=None,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_sizes=(1, 1, 1, 1),
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 ape=False,
                 n_divs=1,
                 merge_type='concat'):
        super().__init__()

        # assert in_chans % n_divs == 0, f'in_chans [{in_chans}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % n_divs == 0, f'embed_dim [{embed_dim}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % (4 * num_heads[
            0]) == 0, f'embed_dim [{embed_dim}] is not divisible by 4*num_head[0]=={4 * num_heads[0]}'

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.ape = ape

        n_chans = int(np.ceil(in_chans / n_divs) * n_divs)
        self.learn_vector = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=n_chans, kernel_size=3, padding=1),
            nn.GELU()
        ) if n_chans != in_chans else nn.Identity()

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=n_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            n_divs=n_divs
        )

        # absolute position embedding
        if self.ape:
            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths)*2)]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
                layer = EncodeLayer(
                    dim=int(embed_dim * 2 ** i_layer),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer != 0) else None,
                    use_checkpoint=use_checkpoint,
                    n_divs=n_divs
                )
                self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        # assert np.all(k % v for k, v in zip(num_features, num_heads)), 'embed_dim/ num_features not a multiple of heads'
        self.num_features = num_features

        # build up layers
        self.uplayers = nn.ModuleList()
        self.decode_dim = num_features  # [::-1]
        for i_layer in range(self.num_layers - 1, -1, -1):
            drop_path = dpr[sum(depths[(1 + i_layer):]) + sum(depths):sum(depths[(i_layer):]) + sum(depths)]
            decode_dim = self.decode_dim[i_layer]
            layer = DecodeLayerUnet(
                dim=decode_dim,
                depth=depths[i_layer],  # 1,  #depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_sizes[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer=norm_layer,
                upsample=PatchExpanding if i_layer != (self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                n_divs=n_divs,
                merge_type=merge_type
            )
            self.uplayers.append(layer)

        self.head = Head(decode_dim=decode_dim, patch_size=self.patch_size, out_chans=out_chans, n_divs=n_divs,
                         img_size=self.img_size)

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        x = self.learn_vector(x)
        x = self.patch_embed(x)

        H, W = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(H, W), mode='bicubic')
            # x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
            x = rearrange(x + absolute_pos_embed, 'b c h w -> b h w c')
        else:
            # x = x.flatten(2).transpose(1, 2)
            x = rearrange(x, 'b c h w -> b h w c')

        x = self.pos_drop(x)

        features = []
        for layer in self.layers:
            x = layer(x)
            features.insert(0, x)

        for layer, y in zip(self.uplayers, features):
            x = layer(x, y)

        x = self.head(x)
        x = rearrange(x, 'b h w c -> b c h w')

        return x


class PPM(nn.ModuleList):
    """Transformer based (Pooling Pyramid Module used in PSPNet).

    Args:
        in_features (int): Input features.
        out_features (int): Output features after modules.
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, in_features, out_features, 
                 pool_scales=(1, 2, 3, 6), align_corners=False,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, n_divs=1  # new params
                 ):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_features = in_features
        self.out_features = out_features
        
        Linear = nn.Linear if n_divs == 1 else HyperLinear
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    Rearrange('b c h w -> b h w c'),
                    Linear(self.in_features, self.out_features, **{'n_divs': n_divs for n in [n_divs] if n > 1}),
                    norm_layer(self.out_features),
                    act_layer(),
                    Rearrange('b h w c -> b c h w')
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        x = rearrange(x, 'b h w c -> b c h w')
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(rearrange(upsampled_ppm_out, 'b c h w -> b h w c'))
        return ppm_outs


class UPerHead(nn.Module):
    """Transformer based (Unified Perceptual Parsing for Scene Understanding).

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
        window_size (int): Local window size. Default: 1.
    """

    def __init__(self, in_dims, dim, pool_scales=(1, 2, 3, 6), align_corners=False,
                 mlp_ratio=4, num_heads=1, window_size=1,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, n_divs=1, use_checkpoint=False,
                 **kwargs):
        super().__init__()
        self.in_dims = in_dims
        self.dim = dim
        self.align_corners = align_corners

        Linear = nn.Linear if n_divs == 1 else partial(HyperLinear, n_divs=n_divs)
        # PSP Module
        self.psp_modules = PPM(
            in_features=self.in_dims[-1],
            out_features=self.dim,
            pool_scales=pool_scales,
            norm_layer=norm_layer,
            act_layer=act_layer,
            align_corners=align_corners,
            n_divs=n_divs)
        self.bottleneck = nn.Sequential(
            Linear(self.in_dims[-1] + len(pool_scales) * self.dim, self.dim),
            norm_layer(self.dim),
            act_layer()
        )

        # FPN Module
        self.lateral_funcs = nn.ModuleList()
        self.fpn_funcs = nn.ModuleList()
        for in_dim in self.in_dims[:-1]:  # skip the top layer
            l_func = nn.Sequential(
                Linear(in_dim, self.dim),
                norm_layer(self.dim),
                act_layer()
            )
            fpn_func = StarTransformerBlock(dim=self.dim,
                                            num_heads=num_heads,
                                            window_size=window_size,
                                            mlp_ratio=mlp_ratio,
                                            act_layer=act_layer, norm_layer=norm_layer,
                                            n_divs=n_divs, use_checkpoint=use_checkpoint,
                                            )

            self.lateral_funcs.append(l_func)
            self.fpn_funcs.append(fpn_func)

        self.fpn_bottleneck = nn.Sequential(
            StarTransformerBlock(dim=len(self.in_dims) * self.dim,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 mlp_ratio=mlp_ratio,
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 n_divs=n_divs, use_checkpoint=use_checkpoint,
                                 ),
            nn.Sequential(
                Linear(len(self.in_dims) * self.dim, self.dim),
                norm_layer(self.dim),
                act_layer()
            )
        )

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=-1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        # build laterals
        laterals = [
            lateral_func(inputs[i])
            for i, lateral_func in enumerate(self.lateral_funcs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[1:-1]
            laterals[i - 1] += rearrange(resize(
                rearrange(laterals[i], 'b h w c -> b c h w'),
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners),
                'b c h w -> b h w c'
            )

        # build outputs
        fpn_outs = [
            self.fpn_funcs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        used_shape = fpn_outs[0].shape[1:-1]
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = rearrange(resize(
                rearrange(fpn_outs[i], 'b h w c -> b c h w'),
                size=used_shape,
                mode='bilinear',
                align_corners=self.align_corners),
                'b c h w -> b h w c'
            )
        fpn_outs = torch.cat(fpn_outs, dim=-1)
        output = self.fpn_bottleneck(fpn_outs)
        # output = self.cls_seg(output)
        return output


class StarUPerNet(nn.Module):
    """ Star Transformer backbone.
        A PyTorch impl of :
    Args:
        img_size (int/tuple): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Star Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
        window_sizes (tuple[int]): Local window sizes. Default: 1.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self,
                 img_size=(256, 256),
                 patch_size=4,
                 in_chans=3,
                 out_chans=None,
                 embed_dim=96,
                 decode_dim=None,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 pool_scales=(1, 2, 3, 6),
                 window_sizes=(1, 1, 1, 1),
                 decode_window_size=None,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 ape=False,
                 n_divs=1):
        super().__init__()

        # assert in_chans % n_divs == 0, f'in_chans [{in_chans}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % n_divs == 0, f'embed_dim [{embed_dim}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % (4 * num_heads[
            0]) == 0, f'embed_dim [{embed_dim}] is not divisible by 4*num_head[0]=={4 * num_heads[0]}'

        decode_dim = decode_dim or embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.ape = ape

        n_chans = int(np.ceil(in_chans / n_divs) * n_divs)
        self.learn_vector = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=n_chans, kernel_size=3, padding=1),
            nn.GELU()
        ) if n_chans != in_chans else nn.Identity()

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=n_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            n_divs=n_divs
        )

        # absolute position embedding
        if self.ape:
            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
                layer = EncodeLayer(
                    dim=int(embed_dim * 2 ** i_layer),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer != 0) else None,
                    use_checkpoint=use_checkpoint,
                    n_divs=n_divs
                )
                self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # build up layers
        decode_window_size = decode_window_size or window_sizes[0]
        self.decode_head = UPerHead(
            in_dims=self.num_features, dim=decode_dim,
            pool_scales=pool_scales, window_size=decode_window_size,
            mlp_ratio=mlp_ratio, num_heads=1,
            norm_layer=nn.LayerNorm, n_divs=n_divs, use_checkpoint=use_checkpoint,
        )

        self.head = Head(decode_dim=decode_dim, patch_size=self.patch_size, out_chans=out_chans, n_divs=n_divs,
                         img_size=self.img_size)

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        x = self.learn_vector(x)
        x = self.patch_embed(x)

        H, W = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(H, W), mode='bicubic')
            # x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
            x = rearrange(x + absolute_pos_embed, 'b c h w -> b h w c')
        else:
            # x = x.flatten(2).transpose(1, 2)
            x = rearrange(x, 'b c h w -> b h w c')

        x = self.pos_drop(x)

        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        x = self.decode_head(features)

        x = self.head(x)
        x = rearrange(x, 'b h w c -> b c h w')

        return x


