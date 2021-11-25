# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'cswin_224': _cfg(),
    'cswin_384': _cfg(
        crop_pct=1.0
    ),

}


class Mlp(nn.Module):
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


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0.,
                 proj_drop=0., qk_scale=None):
        super().__init__()
        self.idx = idx
        self.dim = dim
        self.dim_out = dim_out or dim
        # self.resolution = to_2tuple(resolution)
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.get_v = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    def im2cswin(self, x):

        B, H, W, C = x.shape
        x = rearrange(x, 'b h w c -> b c h w')
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x):
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w c -> b c h w')
        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        lepe = self.get_v(x) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B H W C   (before B L C)
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        B, H, W, C = q.shape

        if self.idx == -1:
            self.H_sp, self.W_sp = H, W  # , self.resolution
        elif self.idx == 0:
            self.H_sp, self.W_sp = H, self.split_size
        elif self.idx == 1:
            self.W_sp, self.H_sp = W, self.split_size
        else:
            print("ERROR MODE", self.idx)
            exit(0)
        H_sp, W_sp = self.H_sp, self.W_sp
        pad_l = pad_t = 0
        pad_r = (W_sp - W % W_sp) % W_sp
        pad_b = (H_sp - H % H_sp) % H_sp

        q = F.pad(q, (0, 0, pad_l, pad_r, pad_t, pad_b))
        k = F.pad(k, (0, 0, pad_l, pad_r, pad_t, pad_b))
        v = F.pad(v, (0, 0, pad_l, pad_r, pad_t, pad_b))
        B, H, W, C = q.shape

        ### Img2Window
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        # print(f"b: {x.shape}")
        # x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C
        x = x.transpose(1, 2).reshape(-1, H_sp, W_sp, C)  # B head N N @ B head N C
        # x = rearrange(x, 'b k (h w) c -> b h w (k c)')

        # x = self.proj_drop(self.proj(x))
        ### Window2Img
        x = windows2img(x, H_sp, W_sp, H, W)  # .view(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False, use_chk=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_chk = use_chk
        self.patches_resolution = to_2tuple(reso)
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        # if self.patches_resolution == split_size:
        if split_size in self.patches_resolution:
            last_stage = True
        elif split_size < min(self.patches_resolution):
            last_stage = True
            split_size = min(self.patches_resolution)

        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim//2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H, W, C  (before B, H*W, C)
        """

        B, H, W, C = x.shape
        img = self.norm1(x)
        qkv = rearrange(self.qkv(img), 'B H W (n C) -> n B H W C', n=3)
        # qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        if self.use_chk:
            if self.branch_num == 2:
                x0 = checkpoint.checkpoint(self.attns[0], qkv[..., :C // 2])[:, :H, :W, :].contiguous()
                x1 = checkpoint.checkpoint(self.attns[1], qkv[..., C // 2:])[:, :H, :W, :].contiguous()
                attened_x = torch.cat([x0, x1], dim=-1)
            else:
                attened_x = checkpoint.checkpoint(self.attns[0], qkv)[:, :H, :W, :].contiguous()
        else:
            if self.branch_num == 2:
                x0 = self.attns[0](qkv[..., :C // 2])[:, :H, :W, :].contiguous()
                x1 = self.attns[1](qkv[..., C // 2:])[:, :H, :W, :].contiguous()
                attened_x = torch.cat([x0, x1], dim=-1)
            else:
                attened_x = self.attns[0](qkv)[:, :H, :W, :].contiguous()
        attened_x = self.proj(attened_x)

        x = x + self.proj_drop(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    # B, N, C = img_splits_hw.shape
    # # pad feature maps to multiples of window size
    # pad_l = 0
    # pad_r = (W_sp * H_sp - N % (W_sp*H_sp)) % (H_sp * W_sp)
    # img_splits_hw = F.pad(img_splits_hw, (0, 0, pad_l, pad_r))

    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        # B, new_HW, C = x.shape
        # H = W = int(np.sqrt(new_HW))
        # B, H, W, C = x.shape

        # x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.conv(x)
        # B, C = x.shape[:2]
        # x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, reso, depth, num_heads,
                 split_size=7, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 merge_block=None, use_chk=False):
        super().__init__()
        self.use_chk = use_chk
        self.merge = merge_block
        if merge_block is None:
            self.merge = nn.Identity()

        self.blocks = nn.ModuleList([
            CSWinBlock(
                dim=dim, num_heads=num_heads, reso=reso,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, act_layer=act_layer, use_chk=use_chk)
            for i in range(depth)])

    def forward(self, x):
        x = self.merge(x)
        for blk in self.blocks:
            x = blk(x)
            # if self.use_chk:
            #     x = checkpoint.checkpoint(blk, x)
            # else:
            #     x = blk(x)
        return x


class CSWinTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, extract_features=False,
                 embed_dim=96, depth=[2,2,32,2], split_size=[1,2,7,7],
                 num_heads=[2, 4, 8, 16], mlp_ratio=4, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, use_checkpoint=False, **kwargs):
        super().__init__()
        self.use_chk = use_checkpoint
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads = num_heads

        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, (7, 7), (4, 4), 2),
            # Rearrange('b c h w -> b (h w) c', h=img_size//4, w=img_size//4),
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(embed_dim)
        )

        img_size = to_2tuple(img_size)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule

        # build stages
        # self.stages = nn.ModuleList()
        # self.merges = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.embed_dims = []
        self.num_stages = len(depth)
        curr_dim = embed_dim
        curr_patch_size = patch_size
        for i_stage in range(self.num_stages):
            resolution = [np.ceil(n / curr_patch_size).astype(int) for n in img_size]
            self.embed_dims.append(curr_dim)
            layer = BasicLayer(dim=curr_dim, reso=resolution, depth=depth[i_stage], num_heads=heads[i_stage],
                               split_size=split_size[i_stage], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depth[:i_stage]):sum(depth[:i_stage + 1])],
                               norm_layer=norm_layer,
                               merge_block=Merge_Block(curr_dim//2, curr_dim) if i_stage > 0 else nn.Identity(),
                               use_chk=self.use_chk
                               )
            self.layers.append(layer)
            curr_dim *= 2  # embed_dim * 2 ** i_stage
            curr_patch_size *= 2  # patch_size * 2 ** i_stage

        curr_dim //= 2
        self.norm = norm_layer(curr_dim)

        self.extract_features = extract_features
        if not self.extract_features:
            # Classifier head
            self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()

            trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)

    def get_features(self):
        self.extract_features = True
        delattr(self, 'head')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def reset_classifier(self, num_classes, global_pool=''):
        if self.num_classes != num_classes:
            print ('reset head to', num_classes)
            self.num_classes = num_classes
            self.head = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head = self.head.cuda()
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x):
        # B = x.shape[0]
        x = self.stage1_conv_embed(x)
        out = []
        # for pre, blocks in zip(self.merges, self.stages):
        #     x = pre(x)
        #     for blk in blocks:
        #         if self.use_chk:
        #             x = checkpoint.checkpoint(blk, x)
        #         else:
        #             x = blk(x)
        for layer in self.layers:
            x = layer(x)
            out.append(rearrange(x, 'b h w c -> b c h w'))
        if not self.extract_features:
            # x = self.norm(x)
            return x  # torch.mean(x, dim=1)
        else:
            return out

    def forward(self, x):
        x = self.forward_features(x)
        if not self.extract_features:
            x = self.norm(x)
            x = reduce(x, 'b h w c -> b c', 'mean')
            x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


default_cfg = {
'tiny_224':
    {'patch_size': 4, 'embed_dim': 64, 'depth': [1, 2, 21, 1], 'split_size': [1, 2, 7, 7], 'num_heads': [2, 4, 8, 16],
     'mlp_ratio': 4, 'img_size': 224, 'drop_path_rate': 0.1},
'small_224':
    {'patch_size': 4, 'embed_dim': 64, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 7, 7], 'num_heads': [2, 4, 8, 16],
     'mlp_ratio': 4, 'img_size': 224, 'drop_path_rate': 0.3},
'base_224':
    {'patch_size': 4, 'embed_dim': 96, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 7, 7], 'num_heads': [4, 8, 16, 32],
     'mlp_ratio': 4, 'img_size': 224, 'drop_path_rate': 0.5},
'large_224':
    {'patch_size': 4, 'embed_dim': 144, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 7, 7],
     'num_heads': [6, 12, 24, 24], 'mlp_ratio': 4, 'img_size': 224, 'drop_path_rate': 0.7},
'base_384':
    {'patch_size': 4, 'embed_dim': 96, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 12, 12],
     'num_heads': [4, 8, 16, 32], 'mlp_ratio': 4, 'img_size': 384, 'drop_path_rate': 0.5},
'large_384':
    {'patch_size': 4, 'embed_dim': 144, 'depth': [2, 4, 32, 2], 'split_size': [1, 2, 12, 12],
     'num_heads': [6, 12, 24, 24], 'mlp_ratio': 4, 'img_size': 384, 'drop_path_rate': 0.7},
}


class CSWin(CSWinTransformer):
    def __init__(self, name='base_224', **kwargs):
        super().__init__(**default_cfg[name], **kwargs)
