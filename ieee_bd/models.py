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

import warnings
from hypercomplex_patch.dpn import DPN
from backbones.swin_transformer import SwinTransformer, SwinEncoderDecoderTransformer
from backbones.swin_transformer3d import SwinTransformer3D, SwinEncoderDecoderTransformer3D, SwinUNet3D, SwinUPerNet3D
from backbones.swin_transformer3d2 import SwinUNet3D as Swin2UNet3D
from backbones.twin_svt import TwinsSVT
from backbones.hypercomplex_transformer import HyperTransformerUNet
from backbones.sungbinchoi.models import Net as DenseUNet, Net3p as DenseUNet3p
from backbones.resnet import HyperUnet as ResNetUnet, Encoder, DecodeBlock
from layers import *
import torch_optimizer as extra_optim
from sam import SAM
from backbones import CoAtNet
from backbones.heads.fapn import FaPNHead
import torch.nn.functional as F


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
                  # 'adam': partial(DeepSpeedCPUAdam, adamw_mode=False),  # optim.Adam,
                  'adam': optim.Adam,
                  'adamw': optim.AdamW,
                  'swats': extra_optim.SWATS,
                  'lamb': extra_optim.Lamb,
                  'sparse_adam': optim.SparseAdam,
                  'adamax': optim.Adamax,
                  'asgd': optim.ASGD,
                  'sgd': optim.SGD,
                  'rprop': optim.Rprop,
                  'rmsprop': optim.RMSprop,
                  'sam': SAM,
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


def get_extras(depth, stages, activation):  # , n_divs=1):
    num_blocks = [depth] * stages
    act_layer = {'relu': F.relu,
                 'prelu': F.prelu,
                 'hardtanh': F.hardtanh,
                 'tanh': F.tanh,
                 'elu': F.elu,
                 'leakyrelu': F.leaky_relu,
                 'selu': F.selu,
                 'gelu': F.gelu,
                 'silu': F.silu,
                 'sigmoid': F.sigmoid,
                 }[activation]
    return num_blocks, act_layer


class HyperResNetUnet(ResNetUnet):
    def __init__(self, args):
         # Details here
        if hasattr(args, 'n_divs'):
            n_divs = args.n_divs
        else:
            n_divs = n_div_dict[args.net_type.lower()]
        in_channels = args.len_seq_in * args.in_depth
        out_channels = args.len_seq_out * args.out_depth

        in_planes = int(n_divs * np.ceil(args.sf / n_divs))
        if args.sf < in_planes:
            warnings.warn(f"start_filters = {args.sf} < in_planes used [{in_planes}]")

        num_blocks, act_layer = get_extras(args.nb_layers, args.stages, args.hidden_activation)  # , n_divs=n_divs)
        super().__init__(num_blocks=num_blocks,
                         in_channels=in_channels,
                         num_classes=out_channels,
                         in_planes=in_planes,
                         n_divs=n_divs,
                         act_layer=act_layer,
                         inplace_activation=args.inplace_activation,
                         constant_dim=args.constant_dim,
                         use_se=args.use_se,
                         with_sigmoid=False,
                         )


class HyperCoAtNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        if hasattr(args, 'n_divs'):
            n_divs = args.n_divs
        else:
            n_divs = n_div_dict[args.net_type.lower()]
        in_channels = args.len_seq_in*args.in_depth

        mid_channels = int(n_divs * np.ceil(in_channels / n_divs))
        if args.sf < mid_channels:
            warnings.warn(f"args.sf = {args.sf} < num_init_features used [{mid_channels}]")
        if args.sf > mid_channels:
            mid_channels = int(n_divs * np.ceil(args.sf / n_divs))

        assert args.coatnet_type <= 4, \
            f'Default configurations 0, 1, 2, 3, 4 or custom -1 applicable. Not {args.coatnet_type}'

        backbone = {0: CoAtNet.CoAtNet0,
                    1: CoAtNet.CoAtNet1,
                    2: CoAtNet.CoAtNet2,
                    3: CoAtNet.CoAtNet3,
                    4: CoAtNet.CoAtNet4}

        if args.coatnet_type == -1:
            planes = tuple([mid_channels * 2**i for i in range(args.stages)])
            if args.constant_dim:
                planes = tuple([mid_channels] * args.stages)
            self.backbone = CoAtNet.CoAtNet(num_channels=in_channels,
                                            num_init_features=mid_channels,  # in_planes=96,
                                            planes=planes,
                                            transformer_idx=2,  # None,
                                            num_heads=32,
                                            depths=tuple([args.nb_layers] * args.stages),  # (2, 3, 5, 2),
                                            image_size=(args.height, args.width),  # num_classes=1000,
                                            drop_rate=args.dropout,
                                            attn_drop_rate=0.,
                                            drop_path_rate=args.drop_path,  # 0.2,
                                            expansion=args.mlp_ratio,  # 4,
                                            )
        else:
            self.backbone = backbone[args.coatnet_type](num_channels=in_channels,
                                                        num_init_features=mid_channels,  # in_planes=96,
                                                        in_planes=mid_channels,
                                                        transformer_idx=2,  # None,
                                                        image_size=(args.height, args.width),  # num_classes=1000,
                                                        )

        out_channels = args.len_seq_out * args.out_depth
        if args.coatnet_head == 'fapn':
            self.head = FaPNHead([mid_channels, *self.backbone.planes], mid_channels, out_channels)
        else:
            pass

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out


class SedenionModel(nn.Module):
    def __init__(self, args):
        super(SedenionModel, self).__init__()
        self.args = args
        # Details here
        args.convArgs = {"padding": "same", "bias": False, "weight_init": 'hypercomplex'}
        if args.net_type.lower() == 'real':
            args.convArgs.pop('weight_init')
        l2Args = {'weight_decay': args.weight_decay}
        args.bnArgs = {"momentum": 0.9, "eps": 1e-04}
        args.actArgs = {"activation": args.hidden_activation}  # "elu"}
        # classifier_actArgs = {"activation": args.classifier_activation}


        args.n_divs = n_div_dict[args.net_type.lower()]
        self.n_divs = args.n_divs
        # print(args)
        h, w = args.height, args.width

        frame_shape = (h, w)

        ##########################
        self.net_type = args.net_type
        self.input_shape = (None, args.in_channels, *frame_shape)
        # self.dynamic_shape = (None, args.n_frame_in * args.n_channels, *frame_shape)
        # self.static_shape = (None, 8, *frame_shape) if args.use_time_slot else (None, 7, *frame_shape)
        n_divs = n_div_dict[args.net_type.lower()]
        assert n_divs in [16, 1]  # only real or sedanion implemented
        if n_divs == 16:
            n_multiples_in = int(16 * np.ceil(args.in_channels / 16))
            n_multiples_out = args.n_classes  # int(16 * np.ceil(args.n_classes/16))  # 16 * args.n_channels_out  # n_divs = 16 for sedanion
        else:
            n_multiples_out = args.n_classes  # args.n_frame_out * args.n_channels_out

        # Stage 1 - Vector learning and preparation
        if n_divs == 16:
            self.z0_shape = (None, n_multiples_in, *frame_shape)
            # TODO - change kernel size to 3 (May 11, 2021)
            self.encode_vector = LearnVectorBlock(args, self.input_shape, n_multiples_in, (3, 3), block_i=1)
        else:
            self.z0_shape = self.input_shape

        self.n_stages = np.floor(np.log2(np.array(frame_shape) / 2)).astype(int).max()
        if hasattr(args, 'stages'):
            self.n_stages = args.stages if args.stages > 0 else self.n_stages

        # print(n, frame_shape)
        for i in range(1, self.n_stages + 1):
            ii = (i - 1) // args.sf_grp
            sf_i = args.sf * 2 ** ii  # if args.blk_type.lower() == 'resnet' else args.growth_rate
            z_shape = self.z0_shape if i == 1 else eval(f'self.z_enc{i - 1}.output_shape')
            enc = EncodeBlock(args, z_shape, sf_i, layer_i=i)
            setattr(self, f'z_enc{i}', enc)

        # code
        self.z_code = CreateConvBnLayer(args, enc.output_shape,
                                        sf_i * 2,  # if args.blk_type.lower() == 'resnet' else enc.output_shape[1],
                                        layer_i=100)
        # self.z_code = CreateConvBnLayer(enc.output_shape, sf_i * 2, layer_i=100)

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            # ii = (i-1)//2
            ii = (i - 1) // args.sf_grp
            # print(f'dec{i}: {ii}')
            sf_i =args.sf * 2 ** ii  # (i-1)//2
            x_shape = self.z_code.output_shape if i == self.n_stages else eval(f'self.z_dec{i + 1}.output_shape')
            y_shape = eval(f'self.z_enc{i}.output_shape')
            dec = DecodeBlock(args, x_shape, y_shape, sf_i, layer_i=100+i)
            # dec = DecodeBlock(args, x_shape, y_shape, sf_i, layer_i=100 + i,  # TODO - modified on May 1, 2021
            #                   scale_factor=1 if i == self.n_stages else 2)
            setattr(self, f'z_dec{i}', dec)
        self.z_dec0 = DecodeBlock(args, self.z_dec1.output_shape, self.z0_shape, n_multiples_out, layer_i=200)

        # pad_type = args.convArgs['padding']
        # convArgs_ = args.convArgs.copy()
        # convArgs_['padding'] = (1, 1) if pad_type is 'same' else (0, 0)
        # ModelConv2D = conv_dict[args.net_type.lower()]
        # self.conv = ModelConv2D(in_channels=self.z_dec0.output_shape[CHANNEL_AXIS],
        #                         out_channels=n_multiples_out,  # n_divs = 16
        #                         kernel_size=(3, 3), stride=(1, 1), **convArgs_)
        # TODO - change kernel size to 3 (May 11, 2021)
        self.decode_vector = LearnVectorBlock(args, self.z_dec0.output_shape, n_multiples_out, (3, 3))
        self.classifier = Activation(args, args.classifier_activation)

    def forward(self, x):  # x is [dynamic, static]
        # if self.args.use_static:
        #     dynamic_x, static_x = x
        # else:
        #     dynamic_x = x

        if self.n_divs == 16:
            # self.static_input = self.static_learn(static_x)
            # s_device = self.static_input.device
            # s_shape = list(self.static_input.shape)
            # s_shape[CHANNEL_AXIS] *= 3
            # z0 = torch.cat([self.static_input, dynamic_x, torch.zeros(*s_shape, device=s_device)], dim=1)
            z0 = self.encode_vector(x)
        else:
            z0 = x  # torch.cat(x, dim=1)  # self.stacked_block(torch.cat(x, dim=1))

        for i in range(1, self.n_stages+1):
            exec(f'z{i} = self.z_enc{i}(z{i-1})')

        # code
        z100 = eval(f'self.z_code(z{self.n_stages})')

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            z_ = eval(f'z100 if i == self.n_stages else z10{i + 1}')
            exec(f'z10{i} = self.z_dec{i}([z_, z{i}])')
            # exec(f"del z_, z{i}")
            # torch.cuda.empty_cache()

        z200 = eval(f'self.z_dec0([z101, z0])')

        # model_output = self.classifier(self.conv(z200))
        model_output = self.classifier(self.decode_vector(z200))

        # if self.n_divs == 16:
        #     model_output = torch.cat([get_c(model_output, idx//5, 16) for idx in self.times_out], dim=1)
        return model_output


class SedenionDPN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Details here
        args.convArgs = {"padding": "same", "bias": False, "weight_init": 'hypercomplex'}
        l2Args = {'weight_decay': args.weight_decay}
        args.bnArgs = {"momentum": 0.9, "eps": 1e-04}
        args.actArgs = {"activation": args.hidden_activation}  # "elu"}
        # classifier_actArgs = {"activation": args.classifier_activation}


        args.n_divs = n_div_dict[args.net_type.lower()]
        self.n_divs = args.n_divs
        # print(args)
        h, w = args.height, args.width

        frame_shape = (h, w)

        ##########################
        self.net_type = args.net_type
        self.input_shape = (None, args.in_channels, *frame_shape)
        # self.dynamic_shape = (None, args.n_frame_in * args.n_channels, *frame_shape)
        # self.static_shape = (None, 8, *frame_shape) if args.use_time_slot else (None, 7, *frame_shape)
        n_divs = n_div_dict[args.net_type.lower()]
        assert n_divs in [16, 1]  # only real or sedanion implemented
        if n_divs == 16:
            n_multiples_in = int(16 * np.ceil(args.in_channels / 16))
            n_multiples_out = args.n_classes  # int(16 * np.ceil(args.n_classes/16))  # 16 * args.n_channels_out  # n_divs = 16 for sedanion
        else:
            n_multiples_out = args.n_classes  # args.n_frame_out * args.n_channels_out

        self.n_stages = args.stages

        net = DPN(num_init_features=n_multiples_in, num_input_channels=self.input_shape[1],
                  c0=n_multiples_out, c1=n_multiples_out * 2, c2=n_multiples_out * 4,
                  num_block=self.n_stages, n_divs=n_divs, activation=args.hidden_activation)
        # Stage 1 - Vector learning and preparation
        if n_divs == 16:
            # print(n_divs)
            self.z0_shape = (None, n_multiples_in, *frame_shape)
            # TODO - change kernel size to 3 (May 11, 2021)
            net.pre_blk = LearnVectorBlock(args, self.input_shape, n_multiples_in, (3, 3), block_i=1)
        else:
            self.z0_shape = self.input_shape

        self.features = net
        # TODO - change kernel size to 3 (May 11, 2021)
        self.decode_vector = LearnVectorBlock(args, [None, self.features.c0, None, None], n_multiples_out, (3, 3))
        self.classifier = Activation(args, args.classifier_activation)

    def forward(self, x):  # x is [dynamic, static]

        x = self.features(x)
        model_output = self.classifier(self.decode_vector(x))
        return model_output


class SedenionSwinTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Details here
        args.convArgs = {"padding": "same", "bias": False, "weight_init": 'hypercomplex'}
        l2Args = {'weight_decay': args.weight_decay}
        args.bnArgs = {"momentum": 0.9, "eps": 1e-04}
        args.actArgs = {"activation": args.hidden_activation}  # "elu"}
        # classifier_actArgs = {"activation": args.classifier_activation}


        args.n_divs = n_div_dict[args.net_type.lower()]
        self.n_divs = args.n_divs
        # print(args)
        h, w = args.height, args.width

        frame_shape = (h, w)

        ##########################
        self.net_type = args.net_type
        self.input_shape = (None, args.in_channels, *frame_shape)
        n_divs = n_div_dict[args.net_type.lower()]
        assert n_divs in [16, 1]  # only real or sedanion implemented
        if n_divs == 16:
            n_multiples_in = int(16 * np.ceil(args.in_channels / 16))
            n_multiples_out = args.n_classes  # int(16 * np.ceil(args.n_classes/16))  # 16 * args.n_channels_out  # n_divs = 16 for sedanion
        else:
            n_multiples_in = args.in_channels
            n_multiples_out = args.n_classes  # args.n_frame_out * args.n_channels_out

        self.n_stages = args.stages

        # Stage 1 - Vector learning and preparation
        if n_divs == 16:
            # print(n_divs)
            self.z0_shape = (None, n_multiples_in, *frame_shape)
            # TODO - change kernel size to 3 (May 11, 2021)
            encode_vector = LearnVectorBlock(args, self.input_shape, n_multiples_in, (3, 3), block_i=1)
        else:
            self.z0_shape = self.input_shape
            encode_vector = nn.Identity()

        patch_size = args.patch_size if hasattr(args, 'patch_size') else 4
        net = SwinTransformer(img_size=frame_shape,
                              use_checkpoint=args.memory_efficient if hasattr(args, 'memory_efficient')
                              else args.inplace_activation,
                              patch_size=patch_size,
                              depths=tuple([args.nb_layers] * self.n_stages),
                              num_heads=tuple([3 * (k + 1) for k in range(self.n_stages)]),
                              out_indices=tuple([k for k in range(self.n_stages)]),
                              embed_dim=int(n_divs * 3 * np.ceil(n_multiples_out / (n_divs * 3))),
                              in_chans=n_multiples_in,
                              n_divs=n_divs,
                              mlp_ratio=4)

        self.encode_vector = encode_vector
        self.features = net

        num_features = net.num_features

        # code
        self.z_code = CreateConvBnLayer(args, [None, num_features[-1], None, None], num_features[-1] * 2, layer_i=100)

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            sf_i = num_features[i-1]
            x_shape = self.z_code.output_shape if i == self.n_stages else eval(f'self.z_dec{i + 1}.output_shape')
            y_shape = [None, sf_i, None, None]  # eval(f'self.z_enc{i}.output_shape')
            dec = DecodeBlock(args, x_shape, y_shape, sf_i, layer_i=100 + i,
                              scale_factor=1 if i == self.n_stages else 2)
            setattr(self, f'z_dec{i}', dec)
        self.z_dec0 = DecodeBlock(args, self.z_dec1.output_shape, self.z0_shape, n_multiples_out, layer_i=200,
                                  scale_factor=patch_size)
        # TODO - change kernel size to 3 (May 11, 2021)
        self.decode_vector = LearnVectorBlock(args, self.z_dec0.output_shape, n_multiples_out, (3, 3))
        self.classifier = Activation(args, args.classifier_activation)

    def forward(self, x):  # x is [dynamic, static]

        z0 = self.encode_vector(x)

        features = self.features(z0)

        # code
        z100 = self.z_code(features[-1])

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            z_ = eval(f'z100 if i == self.n_stages else z10{i + 1}')
            exec(f'z10{i} = self.z_dec{i}([z_, features[{i-1}]])')

        z200 = eval(f'self.z_dec0([z101, z0])')
        # print(z200.shape)
        model_output = self.classifier(self.decode_vector(z200))

        return model_output


class HyperModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Details here
        args.convArgs = {"padding": "same", "bias": False, "weight_init": 'hypercomplex'}
        if args.net_type.lower() == 'real':
            args.convArgs.pop('weight_init')
        l2Args = {'weight_decay': args.weight_decay}
        args.bnArgs = {"momentum": 0.9, "eps": 1e-04}
        args.actArgs = {"activation": args.hidden_activation}  # "elu"}
        # classifier_actArgs = {"activation": args.classifier_activation}


        args.n_divs = n_div_dict[args.net_type.lower()]
        self.n_divs = args.n_divs
        # print(args)
        h, w = args.height, args.width

        frame_shape = (h, w)

        ##########################
        self.net_type = args.net_type
        self.input_shape = (None, args.in_channels, *frame_shape)

        if hasattr(args, 'n_divs'):
            n_divs = args.n_divs
        else:
            n_divs = n_div_dict[args.net_type.lower()]

        n_multiples_in = int(n_divs * np.ceil(args.in_channels / n_divs))
        n_multiples_out = args.n_classes  # args.n_frame_out * args.n_channels_out

        # Stage 1 - Vector learning and preparation
        if n_divs > 1:
            self.z0_shape = (None, n_multiples_in, *frame_shape)
            # TODO - change kernel size to 3 (May 11, 2021)
            self.encode_vector = LearnVectorBlock(args, self.input_shape, n_multiples_in, (3, 3), block_i=1)
        else:
            self.z0_shape = self.input_shape
            self.encode_vector = nn.Identity()

        self.n_stages = np.floor(np.log2(np.array(frame_shape) / 2)).astype(int).max()
        if hasattr(args, 'stages'):
            self.n_stages = args.stages if args.stages > 0 else self.n_stages

        # print(n, frame_shape)
        for i in range(1, self.n_stages + 1):
            ii = (i - 1) // args.sf_grp
            sf_i = args.sf * 2 ** ii  # if args.blk_type.lower() == 'resnet' else args.growth_rate
            z_shape = self.z0_shape if i == 1 else eval(f'self.z_enc{i - 1}.output_shape')
            enc = EncodeBlock(args, z_shape, sf_i, layer_i=i)
            setattr(self, f'z_enc{i}', enc)

        # code
        self.z_code = CreateConvBnLayer(args, enc.output_shape,
                                        sf_i * 2,  # if args.blk_type.lower() == 'resnet' else enc.output_shape[1],
                                        layer_i=100)
        # self.z_code = CreateConvBnLayer(enc.output_shape, sf_i * 2, layer_i=100)

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            # ii = (i-1)//2
            ii = (i - 1) // args.sf_grp
            # print(f'dec{i}: {ii}')
            sf_i =args.sf * 2 ** ii  # (i-1)//2
            x_shape = self.z_code.output_shape if i == self.n_stages else eval(f'self.z_dec{i + 1}.output_shape')
            y_shape = eval(f'self.z_enc{i}.output_shape')
            # dec = DecodeBlock(args, x_shape, y_shape, sf_i, layer_i=100+i)
            dec = DecodeBlock(args, x_shape, y_shape, sf_i, layer_i=100 + i,  # TODO - modified on May 1, 2021
                              scale_factor=1 if i == self.n_stages else 2)
            setattr(self, f'z_dec{i}', dec)
        self.z_dec0 = DecodeBlock(args, self.z_dec1.output_shape, self.z0_shape, n_multiples_out, layer_i=200)
        # TODO - change kernel size to 3 (May 11, 2021)
        self.decode_vector = LearnVectorBlock(args, self.z_dec0.output_shape, n_multiples_out, (3, 3))
        self.classifier = Activation(args, args.classifier_activation)

    def forward(self, x):  # x is [dynamic, static]
        z0 = self.encode_vector(x)

        for i in range(1, self.n_stages+1):
            exec(f'z{i} = self.z_enc{i}(z{i-1})')

        # code
        z100 = eval(f'self.z_code(z{self.n_stages})')

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            z_ = eval(f'z100 if i == self.n_stages else z10{i + 1}')
            exec(f'z10{i} = self.z_dec{i}([z_, z{i}])')
            # exec(f"del z_, z{i}")
            # torch.cuda.empty_cache()

        z200 = eval(f'self.z_dec0([z101, z0])')

        model_output = self.classifier(self.decode_vector(z200))

        return model_output


class HyperSwinTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Details here
        args.convArgs = {"padding": "same", "bias": False, "weight_init": 'hypercomplex'}
        if args.net_type.lower() == 'real':
            args.convArgs.pop('weight_init')
        l2Args = {'weight_decay': args.weight_decay}
        args.bnArgs = {"momentum": 0.9, "eps": 1e-04}
        args.actArgs = {"activation": args.hidden_activation}  # "elu"}
        # classifier_actArgs = {"activation": args.classifier_activation}


        args.n_divs = n_div_dict[args.net_type.lower()]
        self.n_divs = args.n_divs
        # print(args)
        h, w = args.height, args.width

        frame_shape = (h, w)

        ##########################
        self.net_type = args.net_type
        self.input_shape = (None, args.in_channels, *frame_shape)

        if hasattr(args, 'n_divs'):
            n_divs = args.n_divs
        else:
            n_divs = n_div_dict[args.net_type.lower()]

        n_multiples_in = int(n_divs * np.ceil(args.in_channels / n_divs))
        n_multiples_out = args.n_classes  # args.n_frame_out * args.n_channels_out

        self.n_stages = args.stages

        # Stage 1 - Vector learning and preparation
        if n_divs > 1:
            # print(n_divs)
            self.z0_shape = (None, n_multiples_in, *frame_shape)
            # TODO - change kernel size to 3 (May 11, 2021)
            encode_vector = LearnVectorBlock(args, self.input_shape, n_multiples_in, (3, 3), block_i=1)
        else:
            self.z0_shape = self.input_shape
            encode_vector = nn.Identity()

        patch_size = args.patch_size if hasattr(args, 'patch_size') else 4
        # num_heads = tuple([3 * (k + 1) for k in range(self.n_stages)])  # TODO - modified on May19,2021
        num_heads = tuple([8 for _ in range(self.n_stages)])  # Revisiting Vision Transformer
        heads_ = np.lcm.reduce(num_heads)
        embed_dim = int(n_divs * heads_ * np.ceil(n_multiples_in / (n_divs * heads_)))
        if args.sf > embed_dim:
            embed_dim = int(embed_dim * np.ceil(args.sf / embed_dim))
            net = SwinTransformer(img_size=frame_shape,
                                  use_checkpoint=args.memory_efficient if hasattr(args, 'memory_efficient')
                                  else False,
                                  patch_size=patch_size,
                                  depths=tuple([args.nb_layers] * self.n_stages),
                                  num_heads=num_heads,
                                  out_indices=tuple([k for k in range(self.n_stages)]),
                                  # embed_dim=int(n_divs * heads_ * np.ceil(n_multiples_out / (n_divs * heads_))),
                                  embed_dim=embed_dim,
                                  in_chans=n_multiples_in,
                                  n_divs=n_divs,
                                  mlp_ratio=4,
                                  ape=True)

        self.encode_vector = encode_vector
        self.features = net

        num_features = net.num_features

        # code
        self.z_code = CreateConvBnLayer(args, [None, num_features[-1], None, None], num_features[-1] * 2, layer_i=100)

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            sf_i = num_features[i-1]
            x_shape = self.z_code.output_shape if i == self.n_stages else eval(f'self.z_dec{i + 1}.output_shape')
            y_shape = [None, sf_i, None, None]  # eval(f'self.z_enc{i}.output_shape')
            dec = DecodeBlock(args, x_shape, y_shape, sf_i, layer_i=100 + i,
                              scale_factor=1 if i == self.n_stages else 2)
            setattr(self, f'z_dec{i}', dec)
        self.z_dec0 = DecodeBlock(args, self.z_dec1.output_shape, self.z0_shape, n_multiples_out, layer_i=200,
                                  scale_factor=patch_size)

        # TODO - change kernel size to 3 (May 11, 2021)
        self.decode_vector = LearnVectorBlock(args, self.z_dec0.output_shape, n_multiples_out, (3, 3))
        self.classifier = Activation(args, args.classifier_activation)

    def forward(self, x):  # x is [dynamic, static]

        z0 = self.encode_vector(x)

        features = self.features(z0)

        # code
        z100 = self.z_code(features[-1])

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            z_ = eval(f'z100 if i == self.n_stages else z10{i + 1}')
            exec(f'z10{i} = self.z_dec{i}([z_, features[{i-1}]])')

        z200 = eval(f'self.z_dec0([z101, z0])')
        # print(z200.shape)
        model_output = self.classifier(self.decode_vector(z200))
        return model_output


class HyperSwinEncoderDecoder(SwinEncoderDecoderTransformer):
    def __init__(self, args):
        if hasattr(args, 'n_divs'):
            n_divs = args.n_divs
        else:
            n_divs = n_div_dict[args.net_type.lower()]

        heads_ = 8
        n_multiples_in = int(n_divs * np.ceil(args.in_channels / n_divs))
        embed_dim = int(n_divs * heads_ * np.ceil(n_multiples_in / (n_divs * heads_)))
        if args.sf <embed_dim:
            warnings.warn(f"args.sf = {args.sf} < embed_dim used [{embed_dim}]")
        if args.sf > embed_dim:
            embed_dim = int(embed_dim * np.ceil(args.sf / embed_dim))

        super().__init__(depths=tuple([args.nb_layers] * args.stages),
                         num_heads=tuple([8] * args.stages),
                         out_chans=args.n_classes,
                         in_chans=args.in_channels,
                         embed_dim=embed_dim,  # args.sf,
                         img_size=(args.height, args.width),
                         # ape=True,
                         n_divs=n_divs,
                         patch_size=args.patch_size if hasattr(args, 'patch_size') else 4
                         )


class HyperUNet(HyperTransformerUNet):
    def __init__(self, args):

        super().__init__(depths=tuple([args.nb_layers] * args.stages), #2,2,6,2
                         out_channels=args.len_seq_out * args.out_depth,
                         in_channels=args.len_seq_in * args.in_depth,
                         embed_dim=args.sf,
                         img_size=(args.height, args.width),
                         patch_size=args.patch_size,
                         merge_type=args.merge_type,
                         drop_rate=args.dropout,
                         use_checkpoint=args.memory_efficient,
                         use_neck=args.use_neck,
                         with_sigmoid=False,
                         constant_dim=args.constant_dim,
                         )
                         

class HyperSwinEncoderDecoder3D(SwinEncoderDecoderTransformer3D):
    def __init__(self, args):
        if hasattr(args, 'n_divs'):
            n_divs = args.n_divs
        else:
            n_divs = n_div_dict[args.net_type.lower()]

        heads_ = 8# changed to 4 from 8
        n_multiples_in = int(n_divs * np.ceil(args.len_seq_in / n_divs))
        embed_dim = int(n_divs * heads_ * np.ceil(n_multiples_in / (n_divs * heads_)))
        if args.sf < embed_dim:
            warnings.warn(f"args.sf = {args.sf} < embed_dim used [{embed_dim}]")
        if args.sf > embed_dim:
            embed_dim = int(embed_dim * np.ceil(args.sf / embed_dim))
        patch1 = 1  # np.ceil(args.in_depth / args.out_depth).astype(int)
        # depths = [2] * args.stages
        # if args.stages >= 3:
        #     depths[3] = args.nb_layers

        super().__init__(depths=tuple([args.nb_layers] * args.stages), #2,2,6,2
                         num_heads=tuple([heads_] * args.stages), # 4,8,16,32
                         decode_depth=args.decode_depth,
                         out_chans=args.len_seq_out,
                         in_chans=args.len_seq_in,
                         embed_dim=embed_dim,  # args.sf,
                         img_size=(args.height, args.width),
                         mlp_ratio=args.mlp_ratio,
                         in_depth=np.ceil(args.in_depth / patch1).astype(int), out_depth=args.out_depth,
                         n_divs=n_divs,
                         patch_size=(patch1, *([args.patch_size] * 2)),
                         drop_rate=args.dropout,
                         use_checkpoint=args.memory_efficient,
                         use_neck=args.use_neck,
                         with_sigmoid=False,
                         init_filters=16,
                         )


class HyperSwinUNet3D(SwinUNet3D):
    def __init__(self, args):
        if hasattr(args, 'n_divs'):
            n_divs = args.n_divs
        else:
            n_divs = n_div_dict[args.net_type.lower()]

        heads_ = 8# changed to 4 from 8
        n_multiples_in = int(n_divs * np.ceil(args.len_seq_in / n_divs))
        embed_dim = int(n_divs * heads_ * np.ceil(n_multiples_in / (n_divs * heads_)))
        if args.sf < embed_dim:
            warnings.warn(f"args.sf = {args.sf} < embed_dim used [{embed_dim}]")
        if args.sf > embed_dim:
            embed_dim = int(embed_dim * np.ceil(args.sf / embed_dim))
        patch1 = 1  # np.ceil(args.in_depth / args.out_depth).astype(int)
        # depths = [2] * args.stages
        # if args.stages >= 3:
        #     depths[3] = args.nb_layers

        super().__init__(depths=tuple([args.nb_layers] * args.stages), #2,2,6,2
                         decode_depth=args.decode_depth,
                         num_heads=tuple([heads_] * args.stages), # 4,8,16,32
                         out_chans=args.len_seq_out,
                         in_chans=args.len_seq_in,
                         embed_dim=embed_dim,  # args.sf,
                         img_size=(args.height, args.width),
                         in_depth=np.ceil(args.in_depth / patch1).astype(int), out_depth=args.out_depth,
                         n_divs=n_divs,
                         patch_size=(patch1, *([args.patch_size] * 2)),
                         merge_type=args.merge_type,
                         mlp_ratio=args.mlp_ratio,
                         drop_rate=args.dropout,
                         use_checkpoint=args.memory_efficient,
                         use_neck=args.use_neck,
                         with_sigmoid=False,
                         init_filters=16,
                         constant_dim=args.constant_dim,
                         #window_size=(1, 8, 8),  # updated on sept 9 2021
                         window_size=(4, 4, 4),  # updated on oct 21 2021
                         )


class HyperSwin2UNet3D(Swin2UNet3D):
    def __init__(self, args):
        if hasattr(args, 'n_divs'):
            n_divs = args.n_divs
        else:
            n_divs = n_div_dict[args.net_type.lower()]

        heads_ = 8# changed to 4 from 8
        n_multiples_in = int(n_divs * np.ceil(args.len_seq_in / n_divs))
        embed_dim = int(n_divs * heads_ * np.ceil(n_multiples_in / (n_divs * heads_)))
        if args.sf < embed_dim:
            warnings.warn(f"args.sf = {args.sf} < embed_dim used [{embed_dim}]")
        if args.sf > embed_dim:
            embed_dim = int(embed_dim * np.ceil(args.sf / embed_dim))
        patch1 = 1  # np.ceil(args.in_depth / args.out_depth).astype(int)
        # depths = [2] * args.stages
        # if args.stages >= 3:
        #     depths[3] = args.nb_layers

        super().__init__(depths=tuple([args.nb_layers] * args.stages), #2,2,6,2
                         num_heads=tuple([heads_] * args.stages), # 4,8,16,32
                         out_chans=args.len_seq_out,
                         in_chans=args.in_depth,  #args.len_seq_in,
                         embed_dim=embed_dim,  # args.sf,
                         img_size=(args.height, args.width),
                         in_depth=args.len_seq_in, #np.ceil(args.in_depth / patch1).astype(int),
                         out_depth=args.out_depth,
                         n_divs=n_divs,
                         patch_size=(patch1, *([args.patch_size] * 2)),
                         merge_type=args.merge_type,
                         drop_rate=args.dropout,
                         use_checkpoint=args.memory_efficient,
                         with_sigmoid=False,
                         # init_filters=8, #16,  # 32,
                         use_neck=args.use_neck,
                         constant_dim=args.constant_dim,
                         decode_depth=args.decode_depth,
                         mlp_ratio=args.mlp_ratio,
                         # window_size=(1, 8, 8),
                         window_size=(4, 4, 4),  # modified on oct 21 2021
                         use_hgc=args.use_hgc,
                         )



class HyperSwinUPerNet3D(SwinUPerNet3D):
    def __init__(self, args):
        if hasattr(args, 'n_divs'):
            n_divs = args.n_divs
        else:
            n_divs = n_div_dict[args.net_type.lower()]

        heads_ = 8# changed to 4 from 8
        n_multiples_in = int(n_divs * np.ceil(args.len_seq_in / n_divs))
        embed_dim = int(n_divs * heads_ * np.ceil(n_multiples_in / (n_divs * heads_)))
        if args.sf < embed_dim:
            warnings.warn(f"args.sf = {args.sf} < embed_dim used [{embed_dim}]")
        if args.sf > embed_dim:
            embed_dim = int(embed_dim * np.ceil(args.sf / embed_dim))
        patch1 = 1  # np.ceil(args.in_depth / args.out_depth).astype(int)
        # depths = [2] * args.stages
        # if args.stages >= 3:
        #     depths[3] = args.nb_layers

        super().__init__(depths=tuple([args.nb_layers] * args.stages), #2,2,6,2
                         num_heads=tuple([heads_] * args.stages), # 4,8,16,32
                         out_chans=args.len_seq_out,
                         in_chans=args.len_seq_in,
                         embed_dim=embed_dim,  # args.sf,
                         img_size=(args.height, args.width),
                         in_depth=np.ceil(args.in_depth / patch1).astype(int), out_depth=args.out_depth,
                         n_divs=n_divs,
                         patch_size=(patch1, *([args.patch_size] * 2)),
                         drop_rate=args.dropout,
                         use_checkpoint=args.memory_efficient,
                         with_sigmoid=False,
                         init_filters=16,
                         )


class HyperTwinSVTTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Details here
        args.convArgs = {"padding": "same", "bias": False, "weight_init": 'hypercomplex'}
        if args.net_type.lower() == 'real':
            args.convArgs.pop('weight_init')
        args.bnArgs = {"momentum": 0.9, "eps": 1e-04}
        args.actArgs = {"activation": args.hidden_activation}  # "elu"}
        # classifier_actArgs = {"activation": args.classifier_activation}

        args.n_divs = n_div_dict[args.net_type.lower()]
        self.n_divs = args.n_divs
        # print(args)
        h, w = args.height, args.width

        frame_shape = (h, w)

        ##########################
        self.net_type = args.net_type
        self.input_shape = (None, args.in_channels, *frame_shape)

        if hasattr(args, 'n_divs'):
            n_divs = args.n_divs
        else:
            n_divs = n_div_dict[args.net_type.lower()]

        n_multiples_in = int(n_divs * np.ceil(args.in_channels / n_divs))
        n_multiples_out = args.n_classes  # args.n_frame_out * args.n_channels_out

        self.n_stages = args.stages

        # Stage 1 - Vector learning and preparation
        if n_divs > 1:
            # print(n_divs)
            self.z0_shape = (None, n_multiples_in, *frame_shape)
            # TODO - change kernel size to 3 (May 11, 2021)
            encode_vector = LearnVectorBlock(args, self.input_shape, n_multiples_in, (3, 3), block_i=1)
        else:
            self.z0_shape = self.input_shape
            encode_vector = nn.Identity()

        patch_size = (4,) + (2,) * (self.n_stages - 1)
        net = TwinsSVT(img_size=frame_shape,
                       in_channels=n_multiples_in,
                       dropout=args.dropout,
                       patch_size=patch_size,
                       local_patch_size=8,
                       global_k=8,
                       depths=tuple([args.nb_layers] * self.n_stages),
                       embed_dims=tuple([args.sf * 2 ** ii for ii in range(self.n_stages)]),
                       n_divs=n_divs
                       )

        self.encode_vector = encode_vector
        self.features = net

        num_features = net.num_features

        # code
        self.z_code = CreateConvBnLayer(args, [None, num_features[-1], None, None], num_features[-1] * 2, layer_i=100)

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            sf_i = num_features[i-1]
            x_shape = self.z_code.output_shape if i == self.n_stages else eval(f'self.z_dec{i + 1}.output_shape')
            y_shape = [None, sf_i, None, None]  # eval(f'self.z_enc{i}.output_shape')
            dec = DecodeBlock(args, x_shape, y_shape, sf_i, layer_i=100 + i,
                              scale_factor=1 if i == self.n_stages else 2)
            setattr(self, f'z_dec{i}', dec)
        self.z_dec0 = DecodeBlock(args, self.z_dec1.output_shape, self.z0_shape, n_multiples_out, layer_i=200,
                                  scale_factor=patch_size[0])

        # TODO - change kernel size to 3 (May 11, 2021)
        self.decode_vector = LearnVectorBlock(args, self.z_dec0.output_shape, n_multiples_out, (3, 3))
        self.classifier = Activation(args, args.classifier_activation)

    def forward(self, x):  # x is [dynamic, static]

        z0 = self.encode_vector(x)

        features = self.features(z0)

        # code
        z100 = self.z_code(features[-1])

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            z_ = eval(f'z100 if i == self.n_stages else z10{i + 1}')
            exec(f'z10{i} = self.z_dec{i}([z_, features[{i-1}]])')

        z200 = eval(f'self.z_dec0([z101, z0])')
        # print(z200.shape)
        model_output = self.classifier(self.decode_vector(z200))

        return model_output


class HyperDenseUNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Details here
        args.convArgs = {"padding": "same", "bias": False, "weight_init": 'hypercomplex'}
        if args.net_type.lower() == 'real':
            args.convArgs.pop('weight_init')
        l2Args = {'weight_decay': args.weight_decay}
        args.bnArgs = {"momentum": 0.9, "eps": 1e-04}
        args.actArgs = {"activation": args.hidden_activation}  # "elu"}
        # classifier_actArgs = {"activation": args.classifier_activation}


        args.n_divs = n_div_dict[args.net_type.lower()]
        self.n_divs = args.n_divs
        # print(args)
        h, w = args.height, args.width

        frame_shape = (h, w)

        ##########################
        self.net_type = args.net_type
        self.input_shape = (None, args.in_channels, *frame_shape)

        if hasattr(args, 'n_divs'):
            n_divs = args.n_divs
        else:
            n_divs = n_div_dict[args.net_type.lower()]

        n_multiples_in = int(n_divs * np.ceil(args.in_channels / n_divs))
        n_multiples_out = args.n_classes  # args.n_frame_out * args.n_channels_out

        self.n_stages = args.stages

        # Stage 1 - Vector learning and preparation
        if n_divs > 1:
            # print(n_divs)
            self.z0_shape = (None, n_multiples_in, *frame_shape)
            # TODO - change kernel size to 3 (May 11, 2021)
            encode_vector = LearnVectorBlock(args, self.input_shape, n_multiples_in, (3, 3), block_i=1)
        else:
            self.z0_shape = self.input_shape
            encode_vector = nn.Identity()

        net_func = {'denseunet': DenseUNet, 'denseunet3p': DenseUNet3p}[args.blk_type]
        net = net_func(input_channels=n_multiples_in, out_size=n_multiples_out,
                       nb_layers=tuple([args.nb_layers] * self.n_stages),
                       encode_dims=tuple([args.sf * 2 ** (i // args.sf_grp) for i in range(self.n_stages)]),
                       # encode_dims=tuple([args.sf] * self.n_stages),
                       hidden_size=int(n_divs * np.ceil(n_multiples_out / n_divs)),
                       dense_type=args.dense_type,
                       n_divs=n_divs,
                       drop_rate=args.dropout
                       )
        # if n_divs > 1:
        #     net.head = nn.Sequential(LearnVectorBlock(args, self.z_dec0.output_shape, n_multiples_out, (3, 3)),
        #                              Activation(args, args.classifier_activation))
        self.encode_vector = encode_vector
        self.net = net

    def forward(self, x):  # x is [dynamic, static]

        z0 = self.encode_vector(x)

        model_output = self.net(z0)

        return model_output


class HyperHybridSwinTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Details here
        args.convArgs = {"padding": "same", "bias": False, "weight_init": 'hypercomplex'}
        if args.net_type.lower() == 'real':
            args.convArgs.pop('weight_init')
        l2Args = {'weight_decay': args.weight_decay}
        args.bnArgs = {"momentum": 0.9, "eps": 1e-04}
        args.actArgs = {"activation": args.hidden_activation}  # "elu"}
        # classifier_actArgs = {"activation": args.classifier_activation}


        args.n_divs = n_div_dict[args.net_type.lower()]
        self.n_divs = args.n_divs
        # print(args)
        h, w = args.height, args.width

        frame_shape = (h, w)

        ##########################
        self.net_type = args.net_type
        self.input_shape = (None, args.in_channels, *frame_shape)

        if hasattr(args, 'n_divs'):
            n_divs = args.n_divs
        else:
            n_divs = n_div_dict[args.net_type.lower()]

        n_multiples_in = int(n_divs * np.ceil(args.in_channels / n_divs))
        n_multiples_out = args.n_classes  # args.n_frame_out * args.n_channels_out

        self.n_stages = args.stages

        # Stage 1 - Vector learning and preparation
        if n_divs > 1:
            # print(n_divs)
            self.z0_shape = (None, n_multiples_in, *frame_shape)
            # TODO - change kernel size to 3 (May 11, 2021)
            encode_vector = LearnVectorBlock(args, self.input_shape, n_multiples_in, (3, 3), block_i=1)
        else:
            self.z0_shape = self.input_shape
            encode_vector = nn.Identity()

        self.encode_vector = encode_vector
        args.blk_type = 'resnet'  # for the hybrid type
        for i in range(1, self.n_stages + 1):
            ii = (i - 1) // args.sf_grp
            sf_i = args.sf * 2 ** ii  # if args.blk_type.lower() == 'resnet' else args.growth_rate
            z_shape = self.z0_shape if i == 1 else eval(f'self.z_enc{i - 1}.output_shape')
            enc = EncodeBlock(args, z_shape, sf_i, layer_i=i)
            setattr(self, f'z_enc{i}', enc)

        # code
        patch_size = 1
        num_heads = (8, )  # tuple([3 * (k + 1) for k in range(self.n_stages)])
        net = SwinTransformer(img_size=frame_shape,
                              use_checkpoint=args.memory_efficient if hasattr(args, 'memory_efficient')
                              else False,
                              patch_size=patch_size,
                              depths=(args.nb_layers,),  # tuple([args.nb_layers] * self.n_stages),
                              num_heads=num_heads,
                              out_indices=(0, ),  # tuple([k for k in range(self.n_stages)]),
                              embed_dim=int(n_divs * max(num_heads) * np.ceil(sf_i * 2 /
                                                                              (n_divs * max(num_heads)))),
                              in_chans=sf_i * 2,
                              n_divs=n_divs,
                              mlp_ratio=4)

        # self.encode_vector = encode_vector
        self.features = net

        num_features = net.num_features
        self.z_code = CreateConvBnLayer(args, [None, num_features[-1], None, None], sf_i * 2, layer_i=100)

        code_dim = sf_i * 2
        # Stage 3 - Decoder
        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            # ii = (i-1)//2
            ii = (i - 1) // args.sf_grp
            # print(f'dec{i}: {ii}')
            sf_i = args.sf * 2 ** ii  # (i-1)//2
            x_shape = [None, code_dim, None, None] if i == self.n_stages else eval(f'self.z_dec{i + 1}.output_shape')
            y_shape = eval(f'self.z_enc{i}.output_shape')
            # dec = DecodeBlock(args, x_shape, y_shape, sf_i, layer_i=100+i)
            dec = DecodeBlock(args, x_shape, y_shape, sf_i, layer_i=100 + i,  # TODO - modified on May 1, 2021
                              scale_factor=patch_size if i == self.n_stages else 2)
            setattr(self, f'z_dec{i}', dec)
        self.z_dec0 = DecodeBlock(args, self.z_dec1.output_shape, self.z0_shape, n_multiples_out, layer_i=200)
        # TODO - change kernel size to 3 (May 11, 2021)
        self.decode_vector = LearnVectorBlock(args, self.z_dec0.output_shape, n_multiples_out, (3, 3))
        self.classifier = Activation(args, args.classifier_activation)

    def forward(self, x):  # x is [dynamic, static]
        z0 = self.encode_vector(x)

        for i in range(1, self.n_stages+1):
            exec(f'z{i} = self.z_enc{i}(z{i-1})')
            # print(eval(f"z{i}.shape"))

        # code
        # print(eval(f"[z.shape for z in z{self.n_stages}]"))
        features = eval(f'self.features(z{self.n_stages})')
        # z100 = eval(f'self.z_code(z{self.n_stages})')
        # print([t.shape for t in features])
        z100 = self.z_code(features[0])

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            z_ = eval(f'z100 if i == self.n_stages else z10{i + 1}')
            exec(f'z10{i} = self.z_dec{i}([z_, z{i}])')
            # exec(f"del z_, z{i}")
            # torch.cuda.empty_cache()

        z200 = eval(f'self.z_dec0([z101, z0])')

        model_output = self.classifier(self.decode_vector(z200))

        return model_output
