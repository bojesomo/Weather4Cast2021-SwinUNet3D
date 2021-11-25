# coding:utf-8
##########################################################
# pytorch v1.0
# Alabi Bojesomo
# Khalifa University
# Abu Dhabi, UAE
# April 2021
##########################################################

import os
import sys
sys.path.extend(os.getcwd())

from utils import compute_padding
import numpy as np
import copy

import torch
from torch import nn, optim
import torch_optimizer as extra_optim
from torch.nn import Conv2d, BatchNorm2d, Upsample, GroupNorm
from torch.nn import Conv2d as RealConv2D

from hypercomplex import ComplexConv2D, QuaternionConv2D, OctonionConv2D, SedanionConv2D, get_c
import torch.utils.checkpoint as cp


ROW_AXIS = 2
COL_AXIS = 3
CHANNEL_AXIS = 1
BATCH_AXIS = 0
activation_dict = {'relu': nn.ReLU(),
                   'relu6': nn.ReLU6(),
                   'prelu': nn.PReLU(),
                   'hardtanh': nn.Hardtanh(),
                   'tanh': nn.Tanh(),
                   'elu': nn.ELU(),
                   'leakyrelu': nn.LeakyReLU(),
                   'selu': nn.SELU(),
                   'gelu': nn.GELU(),
                   'glu': nn.GLU(),
                   'swish': nn.SiLU(),
                   'sigmoid': nn.Sigmoid(),
                   'hardsigmoid': nn.Hardsigmoid(),
                   'softsign': nn.Softsign(),
                   'softplus': nn.Softplus,
                   'softmin': nn.Softmin(),
                   'softmax': nn.Softmax()}
optimizer_dict = {'adadelta': optim.Adadelta,
                  'adagrad': optim.Adagrad,
                  'adam': optim.Adam,
                  'adamw': optim.AdamW,
                  'swats': extra_optim.SWATS,
                  'sparse_adam': optim.SparseAdam,
                  'adamax': optim.Adamax,
                  'asgd': optim.ASGD,
                  'sgd': optim.SGD,
                  'rprop': optim.Rprop,
                  'rmsprop': optim.RMSprop,
                  'lbfgs': optim.LBFGS}
criterion_dict = {'mae': nn.L1Loss(),
                  'mse': nn.MSELoss(),
                  'bce': nn.BCELoss(),
                  'binary_crossentropy': nn.BCELoss(),
                  'categorical_crossentropy': nn.CrossEntropyLoss(),
                  }
conv_dict = {'sedenion': SedanionConv2D,
             'octonion': OctonionConv2D,
             'quaternion': QuaternionConv2D,
             'complex': ComplexConv2D,
             'real': RealConv2D}
n_div_dict = {'sedenion': 16,
              'octonion': 8,
              'quaternion': 4,
              'complex': 2,
              'real': 1}


class Concatenate(nn.Module):
    def __init__(self, n_divs):
        super(Concatenate, self).__init__()
        self.n_divs = n_divs

    def forward(self, x):
        O_components = [torch.cat([get_c(x_i, component, self.n_divs) for x_i in x], dim=1) for component in range(self.n_divs)]
        return torch.cat(O_components, dim=1)


class Activation(nn.Sequential):
    def __init__(self, args, activation):
        super(Activation, self).__init__()
        act = activation_dict[activation]
        if hasattr(act, 'inplace'):
            act.inplace = True if args.inplace_activation else act.inplace
        if hasattr(act, 'min_val'):
            act.min_val = 0 if args.modify_activation else act.min_val
        self.add_module('activation', act)


class LearnVectorBlock(nn.Module):
    def __init__(self, args, input_shape, featmaps, filter_size, block_i=1):
        super(LearnVectorBlock, self).__init__()
        self.block_i = block_i
        self.input_shape = input_shape
        [_, num_features1, H_in, W_in] = input_shape
        out_channels = featmaps
        args_ = copy.copy(args)
        args_.n_divs = 1

        # self.bn1 = BatchNorm2d(num_features=num_features1, **args.bnArgs)

        self.norm1 = NormBlock(args_, num_channels=num_features1)
        self.act = Activation(args, **args.actArgs)

        in_channels1 = num_features1  # self.bn1.num_features
        pH = filter_size[0] // 2  # compute_padding(H_in, H_in, filter_size[0], 1)
        pW = filter_size[1] // 2  # compute_padding(W_in, W_in, filter_size[-1], 1)
        padding1 = (pH, pW)
        self.conv1 = Conv2d(in_channels=in_channels1, out_channels=featmaps, kernel_size=filter_size,
                            bias=False, padding=padding1)

        in_channels2 = self.conv1.out_channels
        # self.bn2 = BatchNorm2d(num_features=in_channels2, **args.bnArgs)
        self.norm2 = NormBlock(args_, num_channels=in_channels2)
        self.conv2 = Conv2d(in_channels=in_channels2, out_channels=featmaps, kernel_size=filter_size,
                            bias=False, padding=padding1)
        self.output_shape = [None, featmaps, H_in, W_in]

    def forward(self, x):
        # print(x.dtype)
        # e1 = self.act(self.bn1(x))
        e1 = self.act(self.norm1(x))
        e1 = self.conv1(e1)

        # e1 = self.act(self.bn2(e1))
        e1 = self.act(self.norm2(e1))

        e1 = self.conv2(e1)

        return e1

    def name(self):
        name_p = self.__str__().split('(')[0]
        return f"{name_p}_{self.block_i}"


def Add(x_in):
    x_sum = 0
    for x in x_in:
        x_sum += x
    return x_sum


class GroupNormBlock(nn.Module):  # Sequential):
    def __init__(self, args, num_channels):
        super(GroupNormBlock, self).__init__()
        if args.n_divs == 1:  # net_type is real
            gn = GroupNorm(num_groups=32, num_channels=num_channels, eps=args.bnArgs['eps'])
        else:
            gn = GroupNorm(num_groups=args.n_divs, num_channels=num_channels, eps=args.bnArgs['eps'])
        self.num_features = num_channels
        # self.add_module('gn', gn)
        setattr(self, 'gn', gn)

    def forward(self, x):
        return cp.checkpoint(self.gn, x)


class NormBlock(nn.Sequential):
    def __init__(self, args, num_channels):
        super(NormBlock, self).__init__()
        self.num_channels = num_channels
        if args.use_group_norm:
            norm = GroupNormBlock(args, num_channels=num_channels)
        else:
            norm = BatchNorm2d(num_features=num_channels, **args.bnArgs)
        self.add_module('norm', norm)


class IdentityShortcut(nn.Module):
    def __init__(self, args, input_shape, residual_shape):
        super(IdentityShortcut, self).__init__()
        ModelConv2D = conv_dict[args.net_type.lower()]
        self.input_shape = input_shape
        self.residual_shape = residual_shape
        self.stride_width = int(np.ceil(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
        self.stride_height = int(np.ceil(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
        self.equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

        self.conv = nn.Identity()  # None
        if self.stride_width > 1 or self.stride_height > 1 or not self.equal_channels:
            convArgs_ = args.convArgs.copy()
            convArgs_['padding'] = (0, 0)
            self.conv = ModelConv2D(in_channels=input_shape[CHANNEL_AXIS],
                                    out_channels=residual_shape[CHANNEL_AXIS], kernel_size=(1, 1),
                                    stride=(self.stride_width, self.stride_height), **convArgs_)
        drop_layer = nn.Identity()
        if args.dropout > 0:
            drop_layer = nn.Dropout2d(p=args.dropout, inplace=False)
        self.dropout = drop_layer
        self.output_shape = residual_shape

    def forward(self, x):  # x = [input_I, residual_R]
        [input_I, residual_R] = x
        # shortcut_I = input_I
        # if self.stride_width > 1 or self.stride_height > 1 or not self.equal_channels:
        #     shortcut_I = self.conv(input_I)
        shortcut_I = self.dropout(self.conv(input_I))
        out = Add([shortcut_I, residual_R])
        return out

    def name(self):
        return f'i_shortcut_'

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + self.conv.__repr__() + '\n' + ')'


class ProjectionShortcut(nn.Module):
    def __init__(self, args, input_shape, residual_shape, featmaps):
        super(ProjectionShortcut, self).__init__()
        ModelConv2D = conv_dict[args.net_type.lower()]
        self.input_shape = input_shape
        self.residual_shape = residual_shape

        [N_in, C_in, H_out, W_out] = residual_shape
        pH = compute_padding(H_out, input_shape[ROW_AXIS], 3, 2)
        pW = compute_padding(W_out, input_shape[COL_AXIS], 3, 2)
        convArgs_ = args.convArgs.copy()
        convArgs_['padding'] = (pH, pW)
        self.conv = ModelConv2D(in_channels=input_shape[CHANNEL_AXIS], out_channels=featmaps,
                                kernel_size=(3, 3), stride=(2, 2), **convArgs_)  # TODO -  modify kernel to 3 20Apr21
        # modification fails to learn after 4 epochs
        drop_layer = nn.Identity()
        if args.dropout > 0:
            drop_layer = nn.Dropout2d(p=args.dropout, inplace=False)
        self.dropout = drop_layer
        self.concatenate = Concatenate(args.n_divs)
        self.output_shape = [N_in, C_in + self.conv.out_channels*args.n_divs, H_out, W_out]

    def forward(self, x):  # [input_I, residual_R]
        input_I, residual_R = x
        e1 = self.dropout(self.conv(input_I))
        return self.concatenate([e1, residual_R])

    def name(self):
        return 'p_shortcut_'

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + self.conv.__repr__() + '\n' + ')'


class ResidualBlock(nn.Module):
    def __init__(self, args, input_shape, filter_size, featmaps, shortcut):
        super(ResidualBlock, self).__init__()

        ModelConv2D = conv_dict[args.net_type.lower()]

        self.input_shape = input_shape
        self.featmaps = featmaps
        self.shortcut_type = shortcut
        pad_type = args.convArgs['padding']
        convArgs_ = args.convArgs.copy()
        [N_in, num_features1, H_in, W_in] = input_shape

        # self.norm1 = NormBlock(args, num_channels=num_features1)
        # self.act = Activation(args, args.actArgs['activation'])
        norm1 = NormBlock(args, num_channels=num_features1)
        act = Activation(args, args.actArgs['activation'])
        drop_layer = nn.Identity()
        if args.dropout > 0:
            drop_layer = nn.Dropout2d(p=args.dropout, inplace=False)

        if self.shortcut_type == 'regular':
            convArgs_['padding'] = (1, 1) if pad_type is 'same' else (0, 0)
            stride = (1, 1)
            H_out = H_in
            W_out = W_in
        elif self.shortcut_type == 'projection':
            H_out = int(np.ceil(H_in / 2))
            W_out = int(np.ceil(W_in / 2))
            pH = compute_padding(H_out, H_in, filter_size[0], 2)
            pW = compute_padding(W_out, W_in, filter_size[-1], 2)
            convArgs_['padding'] = (pH, pW)
            stride = (2, 2)
        # self.conv1 = ModelConv2D(in_channels=num_features1, out_channels=featmaps,
        #                          kernel_size=filter_size, stride=stride, **convArgs_)
        conv1 = ModelConv2D(in_channels=num_features1, out_channels=featmaps,
                            kernel_size=filter_size, stride=stride, **convArgs_)
        self.layer1 = nn.Sequential(norm1, act, conv1, drop_layer)

        # self.norm2 = NormBlock(args, num_channels=self.conv1.out_channels*args.n_divs)
        norm2 = NormBlock(args, num_channels=conv1.out_channels * args.n_divs)
        convArgs_['padding'] = (1, 1) if pad_type is 'same' else (0, 0)
        # self.conv2 = ModelConv2D(in_channels=self.conv1.out_channels * args.n_divs, out_channels=featmaps,
        #                          kernel_size=filter_size, stride=(1, 1), **convArgs_)
        conv2 = ModelConv2D(in_channels=conv1.out_channels * args.n_divs, out_channels=featmaps,
                            kernel_size=filter_size, stride=(1, 1), **convArgs_)
        self.layer2 = nn.Sequential(norm2, act, conv2, drop_layer)
        residual_shape = [N_in, featmaps, H_out, W_out]

        if shortcut == 'regular':
            self.shortcut = IdentityShortcut(args, input_shape, residual_shape)
        elif shortcut == 'projection':
            self.shortcut = ProjectionShortcut(args, input_shape, residual_shape, featmaps)
        self.output_shape = self.shortcut.output_shape
        self.residual_shape = residual_shape

    def forward(self, x):

        # e1 = self.conv1(self.act(self.norm1(x)))
        # e1 = self.conv2(self.act(self.norm2(e1)))
        e1 = self.layer2(self.layer1(x))
        e1 = self.shortcut([x, e1])
        return e1  # out

    def name(self):
        return f'residual_block_'


class SubDenseBlock(nn.Module):
    def __init__(self, args, in_channels, filter_size, featmaps, stride, convArgs_, bottleneck=False, concatenate=True):
        super(SubDenseBlock, self).__init__()
        ModelConv2D = conv_dict[args.net_type.lower()]
        if bottleneck:
            self.bottle_neck = nn.Sequential()
            self.bottle_neck.add_module('conv', ModelConv2D(in_channels=in_channels, out_channels=featmaps * 4,
                                                            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                                            bias=False))  # **convArgs_))
            self.bottle_neck.add_module('act', Activation(args, args.actArgs['activation']))
            self.bottle_neck.add_module('norm', NormBlock(args, num_channels=featmaps * 4))
            if args.dropout > 0:
                self.bottle_neck.add_module('dropout', nn.Dropout2d(p=args.dropout, inplace=False))
            in_channels = 4 * featmaps
        self.layer = nn.Sequential()
        self.layer.add_module('conv', ModelConv2D(in_channels=in_channels, out_channels=featmaps,
                                                  kernel_size=filter_size, stride=stride, **convArgs_))
        self.layer.add_module('act', Activation(args, args.actArgs['activation']))
        self.layer.add_module('norm', NormBlock(args, num_channels=featmaps))
        if args.dropout > 0:
            self.layer.add_module('dropout', nn.Dropout2d(p=args.dropout, inplace=False))

        if concatenate:
            self.concatenate = Concatenate(args.n_divs)

    def forward(self, x):
        # return self.norm(self.act(self.conv(x)))
        x1 = x
        if hasattr(self, 'bottle_neck'):
            x1 = cp.checkpoint(self.bottle_neck, x)
            # x1 = self.bottle_neck(x)

        e = self.layer(x1)  # self.norm(self.act(self.conv(x1)))
        if hasattr(self, 'concatenate'):
            e = self.concatenate([x, e])
        # print(e.shape, x.shape, x1.shape)
        # print(self.conv)
        return e


class _Transition(nn.Sequential):
    def __init__(self, args, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        ModelConv2D = conv_dict[args.net_type.lower()]
        self.add_module('norm', NormBlock(args, num_channels=num_input_features))
        self.add_module('relu', Activation(args, args.actArgs['activation']))
        self.add_module('conv', ModelConv2D(in_channels=num_input_features, out_channels=num_output_features,
                                            kernel_size=(1, 1), stride=(1, 1), bias=False))
        if args.dropout > 0:
            self.add_module('dropout', nn.Dropout2d(p=args.dropout, inplace=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))


class DenseBlock(nn.Module):
    def __init__(self, args, input_shape, filter_size, featmaps, pool=True):
        super(DenseBlock, self).__init__()
        self.nb_layers = args.nb_layers
        self.input_shape = input_shape
        self.featmaps = featmaps
        pad_type = args.convArgs['padding']
        convArgs_ = args.convArgs.copy()
        [N_in, num_features1, H_in, W_in] = input_shape
        self.pool = pool
        if self.pool:
            H_out = int(np.ceil(H_in / 2))
            W_out = int(np.ceil(W_in / 2))
            self.avgpool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)  # TODO - transition
            # num_output_features = int(np.ceil(num_features1 / args.n_divs / 2) * args.n_divs)
            # self.transition = _Transition(args, num_features1, num_output_features)
            # num_features1 = num_output_features
        else:
            H_out = H_in
            W_out = W_in
        featmaps_local = int(np.ceil(featmaps / (args.nb_layers * args.n_divs)) * args.n_divs)  # args.growth_rate
        convArgs_['padding'] = (1, 1) if pad_type is 'same' else (0, 0)
        stride = (1, 1)
        in_channels = num_features1
        # self.layer1 = SubDenseBlock(args, in_channels, filter_size, featmaps, stride, convArgs_, bottleneck=True)
        # in_channels = 0
        for idx in range(0, args.nb_layers):
            in_channels += featmaps_local if idx else 0
            setattr(self, f'layer{idx + 1}', SubDenseBlock(args, in_channels, filter_size, featmaps_local,
                                                           stride, convArgs_))  # , bottleneck=True))

        convArgs_['padding'] = (0, 0)
        in_channels += featmaps_local
        # setattr(self, f'layer{args.nb_layers + 1}', SubDenseBlock(args, in_channels, (1, 1), featmaps,
        #                                                      stride, convArgs_))
        # ModelConv2D = conv_dict[args.net_type.lower()]
        # self.out_conv = ModelConv2D(in_channels=in_channels, out_channels=self.featmaps,
        #                         kernel_size=(1, 1), stride=stride, **convArgs_)
        self.out_block = SubDenseBlock(args, in_channels, (1, 1), featmaps,
                                       stride, convArgs_, concatenate=False)
        # in_channels += featmaps
        # self.concatenate = Concatenate(args.n_divs)
        self.output_shape = [N_in, featmaps, H_out, W_out]
        # self.output_shape = [N_in, in_channels, H_out, W_out]

    def forward(self, x):
        if self.pool:  # hasattr(self, 'avgpool'):
            x = self.avgpool(x)
            # print(x.shape)
            # print(self.transition)
            # x = self.transition(x)  # TODO - transition

        for idx in range(1, self.nb_layers + 1):
            # print(idx, x.shape)
            x = eval(f'self.layer{idx}(x)')
            # print(x.shape)
            # exec(f'x = cp.checkpoint(self.layer{idx}, x)')  # checkpoint for memory efficiency
        return self.out_block(x)  # e1  # out

    def name(self):
        return f'residual_block_'


class EncodeBlock(nn.Module):
    def __init__(self, args, input_shape, num_filters, layer_i):
        super(EncodeBlock, self).__init__()

        ModelBlock = {'resnet': ResidualBlock, 'densenet': DenseBlock}[args.blk_type.lower()]

        nb_layers = args.nb_layers
        self.nb_layers = nb_layers
        self.layer_i = layer_i
        self.blocks = []
        self.input_shape = input_shape
        present_shape = input_shape
        if args.blk_type == 'resnet':
            for i in range(1, args.nb_layers):
                blk = ModelBlock(args, present_shape, (3, 3), num_filters, 'regular')
                setattr(self, f"block_{i}", blk)
                self.blocks.append(blk)
                present_shape = self.blocks[i-1].output_shape

            blk = ModelBlock(args, present_shape, (3, 3), num_filters, 'projection')
            setattr(self, f"block_{nb_layers}", blk)
            self.blocks.append(blk)
        else:  # densenet
            pool = layer_i != 1
            blk = ModelBlock(args, present_shape, (3, 3), num_filters, pool)
            setattr(self, f"block_{nb_layers}", blk)
            self.blocks.append(blk)

        exec(f"self.output_shape = self.block_{nb_layers}.output_shape")
        # if args.dropout > 0:
        #     self.dropout = nn.Dropout2d(p=args.dropout, inplace=False)

    def forward(self, x):
        # for i in range(1, self.nb_layers+1):
        for i in range(1, len(self.blocks) + 1):
            x = self.blocks[i-1](x)
        # if hasattr(self, 'dropout'):
        #     x = self.dropout(x)
        return x  # self.x

    def name(self):
        return f'encode_block_{self.layer_i}'


class CreateConvBnLayer(nn.Module):
    def __init__(self, args, input_shape, num_filters, layer_i):
        super(CreateConvBnLayer, self).__init__()
        self.layer_i = layer_i
        self.input_shape = input_shape
        [N_in, C_in, H_in, W_in] = input_shape
        pad_type = args.convArgs['padding']
        convArgs_ = args.convArgs.copy()
        convArgs_['padding'] = (1, 1) if pad_type is 'same' else (0, 0)
        ModelConv2D = conv_dict[args.net_type.lower()]
        # if args.net_type.lower() == 'real':
        #     convArgs_.pop('weight_init')
        self.conv = ModelConv2D(in_channels=input_shape[CHANNEL_AXIS], out_channels=num_filters,
                                kernel_size=(3, 3), stride=(1, 1), **convArgs_)
        self.norm = NormBlock(args, num_channels=self.conv.out_channels * args.n_divs)
        self.act = Activation(args, **args.actArgs)
        drop_layer = nn.Identity()
        if args.dropout > 0:
            drop_layer = nn.Dropout2d(p=args.dropout, inplace=False)
        self.dropout = drop_layer
        self.output_shape = [N_in, self.norm.num_channels, H_in, W_in]

    def forward(self, x):
        return self.dropout(self.act(self.norm(self.conv(x))))

    def name(self):
        return f'conv_bn_{self.layer_i}'


class DecodeSubBlock(nn.Module):
    def __init__(self, args, x_shape, y_shape, num_filters, layer_i, scale_factor=2):
        super(DecodeSubBlock, self).__init__()
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.layer_i = layer_i
        self.up = Upsample(scale_factor=scale_factor) if scale_factor > 1 else nn.Identity()
        [N_in, C_y, H_y, W_y] = y_shape

        ModelConv2D = conv_dict[args.net_type.lower()]

        pad_type = args.convArgs['padding']
        convArgs_ = args.convArgs.copy()
        convArgs_['padding'] = (1, 1) if pad_type is 'same' else (0, 0)
        # if args.net_type.lower() == 'real':
        #     convArgs_.pop('weight_init')
        self.conv = ModelConv2D(in_channels=x_shape[CHANNEL_AXIS], out_channels=num_filters,
                                kernel_size=(3, 3), stride=(1, 1), **convArgs_)
        drop_layer = nn.Identity()
        if args.dropout > 0:
            drop_layer = nn.Dropout2d(p=args.dropout, inplace=False)
        self.dropout = drop_layer
        self.norm = NormBlock(args, num_channels=self.conv.out_channels*args.n_divs + C_y)
        self.act = Activation(args, **args.actArgs)
        self.output_shape = [N_in, C_y+num_filters, H_y, W_y]
        self.concatenate = Concatenate(args.n_divs)

    def forward(self, x):  # [x_in, y_in]
        [x_in, y_in] = x
        # x1 = Upsample(size=(2, 2))(x_in)
        x1 = self.up(x_in)
        x1 = self.dropout(self.conv(x1))

        # hy, wy = self.y_shape[2:]
        y_shape = list(y_in.shape)
        hy, wy = y_shape[2:]
        x1 = x1[:, :, :hy, :wy]

        # x1 = self.conv(x1)
        # print(f'x: {x1.shape}, y: {y_in.shape}')
        x1 = self.concatenate([x1, y_in])  # xy
        # x1 = self.act(self.bn(x1))
        x1 = self.act(self.norm(x1))
        return x1  # xy

    def name(self):
        return f'decode_subblock_{self.layer_i}'


class DecodeBlock(nn.Module):
    def __init__(self, args, x_shape, y_shape, num_filters, layer_i, scale_factor=2):
        super(DecodeBlock, self).__init__()
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.input_shape = [x_shape, y_shape]
        self.layer_i = layer_i

        self.decode_subblock = DecodeSubBlock(args, x_shape, y_shape, num_filters, layer_i, scale_factor=scale_factor)
        self.create_conv = CreateConvBnLayer(args, self.decode_subblock.output_shape, num_filters, layer_i)
        self.output_shape = self.create_conv.output_shape
        if args.dropout > 0:
            self.dropout = nn.Dropout2d(p=args.dropout, inplace=False)

    def forward(self, x):  # [x_in, y_in]
        e1 = self.decode_subblock(x)
        e1 = self.create_conv(e1)
        if hasattr(self, 'dropout'):
            e1 = self.dropout(e1)
        # e1 = self.create_conv(e1) + e1  # TODO - convert to Deep ResUnet by turning the decoders to ResNet blocks
        return e1

    def name(self):
        return f'decode_block_{self.layer_i}'


class DecodeBlock3p(nn.Module):
    def __init__(self, args, x_channels: list, y_channels: list, num_filters, layer_i, scale_factor=2):
        super().__init__()
        self.layer_i = layer_i
        ModelConv2D = conv_dict[args.net_type.lower()]
        pad_type = args.convArgs['padding']
        convArgs_ = args.convArgs.copy()
        convArgs_['padding'] = (1, 1) if pad_type is 'same' else (0, 0)
        self.norm = NormBlock(args, num_channels=self.conv.out_channels * args.n_divs + C_y)
        self.act = Activation(args, **args.actArgs)

        self.x_layers = nn.ModuleList()
        in_channels = 0
        for i, x_channel in enumerate(x_channels):
            # stride = tuple([2 ** (i + 1)] * 2)
            scale_factor = 2 ** (i + 1)
            x_layer = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                ModelConv2D(in_channels=x_channel, out_channels=num_filters,
                            kernel_size=(3, 3), stride=(1, 1), **convArgs_),
                Activation(args, **args.actArgs),
                NormBlock(args, num_channels=num_filters),
                nn.Dropout2d(p=args.dropout, inplace=False) if args.dropout > 0.0 else nn.Identity()
            )
            in_channels += num_filters
            self.x_layers.append(x_layer)


            # self.decode_subblock = DecodeSubBlock(args, x_shape, y_shape, num_filters, layer_i, scale_factor=scale_factor)
            # self.create_conv = CreateConvBnLayer(args, self.decode_subblock.output_shape, num_filters, layer_i)
            # self.output_shape = self.create_conv.output_shape
            # if args.dropout > 0:
            #     self.dropout = nn.Dropout2d(p=args.dropout, inplace=False)

    def forward(self, x):  # [x_in, y_in]
        e1 = self.decode_subblock(x)
        e1 = self.create_conv(e1)
        if hasattr(self, 'dropout'):
            e1 = self.dropout(e1)
        # e1 = self.create_conv(e1) + e1  # TODO - convert to Deep ResUnet by turning the decoders to ResNet blocks
        return e1

    def name(self):
        return f'decode_block_{self.layer_i}'
