##########################################################
# Alabi Bojesomo
# Khalifa University
# Abu Dhabi, UAE
# October 2021
##########################################################

import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .layers import HyperLinear, HyperSoftmax, Concatenate, HyperConv2d, multiply, dot_product
import numpy as np
from einops import rearrange, reduce


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1, expansion=4):
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.expansion = expansion
        self.out_channels = out_channels or in_channels

        mid_channel = in_channels * self.expansion
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channel, kernel_size=(1, 1), bias=False,
                      stride=(stride, stride))
        )

        self.dconv = nn.Sequential(
            nn.BatchNorm2d(num_features=mid_channel),
            nn.GELU(),
            nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=(3, 3),
                      padding=(1, 1), groups=mid_channel, bias=False)
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=mid_channel),
            nn.GELU(),
            nn.Conv2d(in_channels=mid_channel, out_channels=self.out_channels, kernel_size=(1, 1), bias=False)
        )

        self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride,
                                 ceil_mode=True) if self.stride > 1 else nn.Identity()
        self.proj = nn.Identity()
        if self.in_channels != self.out_channels:
            self.proj = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, bias=False,
                                  kernel_size=(1, 1))

    def forward(self, x):
        shortcut = self.proj(self.pool(x))
        x = self.conv1(x)
        x = self.dconv(x)
        x = self.conv2(x)
        return x + shortcut


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, dim_out, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # print(x.shape)
        qkv = self.qkv(x)
        # print(qkv.shape, (B_, N, 3, self.num_heads, C / self.num_heads))
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print(qkv.shape)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # print(q.shape, k.shape, v.shape)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # print(v.type(), attn.type(), attn.device)
        attn = attn.type_as(v)
        x = torch.einsum('bhij,bhjd->bhid', attn, v)
        x = x.transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BasicMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, depth, expansion=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.expansion = expansion

        self.layers = nn.Sequential(
            *[MBConv(in_channels=out_channels, out_channels=out_channels, expansion=expansion) if i != 0 else
              MBConv(in_channels=in_channels, out_channels=out_channels, expansion=expansion, stride=2)
              for i in range(depth)]
        )

    def forward(self, x):
        return self.layers(x)


class Transformer(nn.Module):
    def __init__(self, dim, dim_out, num_heads, window_size=(7, 7), expansion=4, qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., stride=1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.window_size = tuple([np.ceil(w/stride).astype(int) for w in window_size])
        self.expansion = expansion
        self.stride = stride

        self.norm1 = norm_layer(dim)
        self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride,
                                 ceil_mode=True) if self.stride > 1 else nn.Identity()
        self.attn = WindowAttention(
            dim, dim_out, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        mlp_hidden_dim = int(dim_out * expansion)
        self.mlp = Mlp(in_features=dim_out, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride,
                                 ceil_mode=True) if self.stride > 1 else nn.Identity()
        self.proj = nn.Identity()
        if dim != dim_out:
            self.proj = nn.Conv2d(in_channels=dim, out_channels=dim_out, bias=False,
                                  kernel_size=(1, 1))

    def forward(self, x):
        """
        x : (B, C, H, W)
        """
        shortcut = self.proj(self.pool(x))

        x = self.norm1(rearrange(x, 'b c h w -> b h w c'))
        x = self.pool(rearrange(x, 'b h w c -> b c h w'))
        _, _, H, W = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.attn(x)
        x = x + rearrange(shortcut, 'b c h w -> b (h w) c')

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + shortcut

        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class BasicTransformer(nn.Module):
    def __init__(self, dim, dim_out, num_heads, depth, window_size=(7, 7), expansion=4, qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., stride=1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        # self.window_size = tuple([np.ceil(w / stride).astype(int) for w in window_size])
        self.expansion = expansion
        self.stride = stride
        self.depth = depth

        transformers = []
        hidden_window_size = tuple([np.ceil(w / stride).astype(int) for w in window_size])
        for i_layer in range(depth):
            if i_layer == 0:
                transformer = Transformer(dim, dim_out, num_heads, window_size=window_size, expansion=expansion,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[i_layer] if isinstance(drop_path, list) else drop_path,
                                          stride=stride, act_layer=act_layer,
                                          norm_layer=norm_layer)
            else:
                transformer = Transformer(dim_out, dim_out, num_heads, window_size=hidden_window_size,
                                          expansion=expansion, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                                          attn_drop=attn_drop,
                                          drop_path=drop_path[i_layer] if isinstance(drop_path, list) else drop_path,
                                          stride=1, act_layer=act_layer,
                                          norm_layer=norm_layer)
            transformers.append(transformer)
        self.layers = nn.ModuleList(transformers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BasicMixed(nn.Module):
    def __init__(self, dim, dim_out, num_heads, depth, window_size=(7, 7), expansion=4, qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        depth = to_2tuple(depth)

        self.mb_layer = BasicMBConv(dim, dim_out, depth[0], expansion=expansion)

        hidden_window_size = tuple([np.ceil(w / 2).astype(int) for w in window_size])
        self.transformer_layer = BasicTransformer(dim_out, dim_out, num_heads, depth[1], window_size=hidden_window_size,
                                                  expansion=expansion, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                                                  attn_drop=attn_drop, drop_path=drop_path, stride=1,
                                                  act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x):
        x = self.mb_layer(x)
        x = self.transformer_layer(x)
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_init_features):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_init_features, kernel_size=(3, 3),
                      stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=num_init_features),
            nn.GELU(),
            nn.Conv2d(in_channels=num_init_features, out_channels=num_init_features, kernel_size=(3, 3), bias=False,
                      padding=(1, 1)),
            # nn.BatchNorm2d(num_features=num_init_features)
        )

    def forward(self, x):
        return self.layer(x)


class CoAtNet(nn.Module):
    def __init__(self, num_channels=3, num_init_features=64, #in_planes=96,
                 planes=(96, 96*2, 96*4, 96*8), transformer_idx=None,
                 num_heads=32, depths=(2, 3, 5, 2),
                 image_size=(224, 224), #num_classes=1000,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 expansion=4, #drop=0., attn_drop=0., drop_path=0.
                 ):
        super().__init__()
        num_layers = min(len(planes), len(depths))

        self.planes = planes[:num_layers]
        self.depths = depths[:num_layers]

        transformer_idx = min(num_layers, transformer_idx or num_layers // 2)

        self.pre_block = InitialStage(num_channels, num_init_features)

        self.stages = nn.ModuleList()
        in_channels = num_init_features
        for i_layer in range(transformer_idx):
            stage = BasicMBConv(in_channels=in_channels, out_channels=self.planes[i_layer], depth=self.depths[i_layer],
                                expansion=expansion)
            self.stages.append(stage)
            in_channels = self.planes[i_layer]

        # self.stage0 = InitialStage(num_channels, num_init_features)
        #
        # self.stage1 = BasicMBConv(in_channels=num_init_features, out_channels=in_planes, depth=depths[0],
        #                           expansion=expansion)
        # self.stage2 = BasicMBConv(in_channels=in_planes, out_channels=in_planes*2, depth=depths[1],
        #                           expansion=expansion)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(self.depths[transformer_idx:]))]  # stochastic depth decay rule

        """
         dim, dim_out, num_heads, depth, window_size=(7, 7), expansion=4, qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., stride=1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        for i_layer in range(transformer_idx, num_layers):
            divisor = 2 ** (i_layer + 1)
            window_size = tuple([int(np.ceil(w / divisor)) for w in image_size])
            drop_path = dpr[sum(depths[transformer_idx:i_layer]):sum(depths[transformer_idx:i_layer + 1])]

            stage = BasicTransformer(in_channels, self.planes[i_layer], num_heads, self.depths[i_layer],
                                     window_size=window_size,
                                     expansion=expansion, drop=drop_rate, attn_drop=attn_drop_rate,
                                     drop_path=drop_path,
                                     stride=2)
            self.stages.append(stage)
            in_channels = self.planes[i_layer]
        '''
        drop = drop_rate,
        attn_drop = attn_drop_rate,
        drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
        '''
        # window_size = tuple([w/8 for w in image_size])
        # # self.stage3 = BasicTransformer(in_planes*2, in_planes*4, num_heads, depths[2], window_size=window_size,
        # #                                expansion=expansion, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
        # #                                stride=2)
        # self.stage3 = BasicTransformer(in_planes * 2, in_planes * 4, num_heads, depths[2], window_size=window_size,
        #                                expansion=expansion, drop=drop_rate, attn_drop=attn_drop_rate,
        #                                drop_path=dpr[sum(depths[2:2]):sum(depths[2:2 + 1])],
        #                                stride=2)
        #
        # window_size = tuple([w / 16 for w in image_size])
        # # self.stage4 = BasicTransformer(in_planes * 4, in_planes * 8, num_heads, depths[3], window_size=window_size,
        # #                                expansion=expansion, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
        # #                                stride=2)
        # self.stage4 = BasicTransformer(in_planes * 4, in_planes * 8, num_heads, depths[3], window_size=window_size,
        #                                expansion=expansion, drop=drop_rate, attn_drop=attn_drop_rate,
        #                                drop_path=dpr[sum(depths[2:3]):sum(depths[2:3 + 1])],
        #                                stride=2)

        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Linear(in_planes*8, num_classes)

    def forward(self, x):
        # x = self.stage0(x)
        # x = self.stage1(x)
        # x = self.stage2(x)
        # x = self.stage3(x)
        # x = self.stage4(x)

        # print(x.shape)
        # x = self.classifier(self.pool(x))
        # x = self.classifier(reduce(x, 'b c h w -> b c', 'mean'))
        x = self.pre_block(x)
        out = [x]
        for stage in self.stages:
            #print(x.shape)
            x = stage(x)
            out.append(x)
        return out


def CoAtNet0(num_channels=3, num_init_features=64, in_planes=96, image_size=(224, 224), transformer_idx=None):
    return CoAtNet(num_channels=num_channels, num_init_features=num_init_features,
                   planes=tuple([in_planes * 2**i for i in range(4)]),  # in_planes=in_planes,
                   image_size=image_size,  # num_classes=num_classes,
                   depths=[2, 3, 5, 2], transformer_idx=transformer_idx)


def CoAtNet1(num_channels=3, num_init_features=64, in_planes=96, image_size=(224, 224), transformer_idx=None):
    return CoAtNet(num_channels=num_channels, num_init_features=num_init_features,
                   planes=tuple([in_planes * 2**i for i in range(4)]),  # in_planes=in_planes,
                   image_size=image_size,  # num_classes=num_classes,
                   depths=[2, 6, 14, 2], transformer_idx=transformer_idx)


def CoAtNet2(num_channels=3, num_init_features=128, in_planes=128, image_size=(224, 224), transformer_idx=None):
    return CoAtNet(num_channels=num_channels, num_init_features=num_init_features,
                   planes=tuple([in_planes * 2**i for i in range(4)]),  # in_planes=in_planes,
                   image_size=image_size,  # num_classes=num_classes,
                   depths=[2, 6, 14, 2], transformer_idx=transformer_idx)


def CoAtNet3(num_channels=3, num_init_features=192, in_planes=192, image_size=(224, 224), transformer_idx=None):
    return CoAtNet(num_channels=num_channels, num_init_features=num_init_features,
                   planes=tuple([in_planes * 2**i for i in range(4)]),  # in_planes=in_planes,
                   image_size=image_size,  # num_classes=num_classes,
                   depths=[2, 6, 14, 2], transformer_idx=transformer_idx)


def CoAtNet4(num_channels=3, num_init_features=192, in_planes=192, image_size=(224, 224), transformer_idx=None):
    return CoAtNet(num_channels=num_channels, num_init_features=num_init_features,
                   planes=tuple([in_planes * 2**i for i in range(4)]),  # in_planes=in_planes,
                   image_size=image_size,  # num_classes=num_classes,
                   depths=[2, 12, 28, 2], transformer_idx=transformer_idx)
