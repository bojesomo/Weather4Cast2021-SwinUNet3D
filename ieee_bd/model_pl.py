import numpy as np

from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torch
import os

import argparse
from argparse import ArgumentParser
from models import (optimizer_dict, SedenionModel, SedenionDPN, SedenionSwinTransformer, HyperModel,
                    HyperSwinTransformer, HyperTwinSVTTransformer, HyperDenseUNet,
                    HyperHybridSwinTransformer, HyperSwinEncoderDecoder, HyperSwinEncoderDecoder3D,
                    HyperUNet, HyperCoAtNet, HyperResNetUnet,
                    HyperSwinUNet3D, HyperSwinUPerNet3D, HyperSwin2UNet3D)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from dataset import W4CDataset
from einops import rearrange


def get_model(args):
    other_models = {'dpn': SedenionDPN, 'swin': HyperSwinTransformer, 'svt': HyperTwinSVTTransformer,
                    'hybridswin': HyperHybridSwinTransformer,
                    'swinencoder': HyperSwinEncoderDecoder,
                    'swinencoder3d': HyperSwinEncoderDecoder3D, 'swinunet3d': HyperSwinUNet3D,
                    'swinuper3d': HyperSwinUPerNet3D, 'swin2unet3d': HyperSwin2UNet3D,
                    'denseunet': HyperDenseUNet, 'denseunet3p': HyperDenseUNet,
                    #'hrnet': HyperHRNeT,
                    'coatnet': HyperCoAtNet,
                    'resunet': HyperResNetUnet,
                    # 'hyperunet': HyperUNet
                    }
    net = other_models[args.blk_type](args) if args.blk_type in other_models.keys() else HyperModel(args)
    return net


class Model(pl.LightningModule):
    def __init__(self, args):  # ,  # UNet_params: dict,
        # extra_data: str, depth: int, height: int,
        # width: int, len_seq_in: int, len_seq_out: int, bins_to_predict: int,
        # seq_mode: str, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        # args.log_dir = args.log_dir.replace('/l/proj/kuex0005', '/home/farhanakram')
        # args.versiondir = args.versiondir.replace('/l/proj/kuex0005', '/home/farhanakram')

        self.args = args  # argparse.Namespace(**kwargs)

        self.net = get_model(args)

        # self.loss_fn = F.mse_loss
        # if args.target_variable == 'cma':
        #     self.loss_fn = F.binary_cross_entropy_with_logits

        self.core_dir = ''  # os.path.join('result', 'core')
        self.transfer_dir = ''  # os.path.join('result', 'transfer')

        # target_vars = ['temperature', 'crr_intensity', 'asii_turb_trop_prob', 'cma']
        # variable_weights = {
        #     "temperature": 1.0 / 0.03163512,
        #     "crr_intensity": 1.0 / 0.00024158,
        #     "asii_turb_trop_prob": 1.0 / 0.00703378,
        #     "cma": 1.0 / 0.19160305
        # }
        # self.scale = torch.tensor([variable_weights[name] for name in target_vars], device=self.device)
        # self.scale /= self.scale.sum()
        # self.scale *= 40
        #
        self.EPS = 1e-12
        self.eps = 1e-6
        self.asii_logit_m = -torch.tensor(0.003).logit()

        # self.metrics = 100
        if self.args.optimizer == 'sam':
            self.automatic_optimization = False

    def norm_logit(self, y):
        y = (torch.logit(torch.clamp(y, min=0.003, max=0.997), eps=self.eps) + self.asii_logit_m) / (
                2 * self.asii_logit_m)
        return y

    def process_out(self, y):
        # y = torch.sigmoid(y)  # should be already done before here
        if self.args.target_variable == 'asii_turb_trop_prob':
            y = self.norm_logit(y)
        return y

    # def on_fit_start(self):
    #     # print(f"fit: {self.device}")
    #     self.scale = self.scale.to(self.device)

    # def on_epoch_start(self):
    #     # print(f"epoch: {self.device}")
    #     self.scale = self.scale.to(self.device)

    def forward(self, x, mask=None):
        y_hat = self.net(x)
        y_hat = torch.sigmoid(y_hat)  # applying sigmoid on raw output
        return y_hat

    def to_variable_first(self, y):
        if self.args.collapse_time:
            y = rearrange(y, 'b (c d) h w -> d b c h w', c=32)
        else:
            y = rearrange(y, 'b c d h w -> d b c h w')
        return y

    #@staticmethod
    def to_batch_first(self, y):
        #return rearrange(y, 'd b c h w -> b c d h w')
        if self.args.collapse_time:
            return rearrange(y, 'd b c h w -> b (c d) h w')
        else:
            return rearrange(y, 'd b c h w -> b c d h w')

    def calculate_logit(self, y_hat, mask=None, phase='val'):
        # use sigmoid before coming here

        # y_hat = y_hat * mask
        # let's rearrange to variables
        temp, crr, asii, cma = self.to_variable_first(y_hat)
        # y_hat0, y_hat1, y_hat2, y_hat3 = y_hat
        asii = self.norm_logit(asii)
        if phase == 'val':
            cma = torch.round(cma)

        if mask is not None:
            mask = self.to_variable_first(mask)
            temp = temp * (~mask[0])
        # logit = torch.stack([temp, crr, asii, cma])
        return temp, crr, asii, cma  # logit

    def calculate_loss(self, y_hat, y, mask=None, phase='train'):
        # use sigmoid before coming here
        temp, crr, asii, cma = self.calculate_logit(y_hat, mask, phase)
        y = self.to_variable_first(y)

        temp_mask = y[0] > 0
        temp_mse = F.mse_loss(torch.masked_select(temp, temp_mask),
                              torch.masked_select(y[0], temp_mask)) * (1.0 / 0.03163512)
        crr_mse = F.mse_loss(crr, y[1]) * (1.0 / 0.00024158)
        asii_mse = F.mse_loss(asii, y[2]) * (1.0 / 0.00703378)
        cma_mse = F.mse_loss(cma, y[3]) * (1.0 / 0.19160305)

        loss = (temp_mse + crr_mse + asii_mse + cma_mse) / 4
        return loss

    def _compute_loss(self, y_hat, y, mask=None):
        # use sigmoid before coming here
        # y_hat = torch.sigmoid(y_hat)

        y_hat = self.process_out(y_hat)
        y = self.process_out(y)

        if self.args.target_variable == 'cma':
            y_hat = torch.round(y_hat)  # to use rounded mse for categorical result (cma) during validation

        if mask is not None:
            y_hat = y_hat.flatten()[~mask.flatten()]
            y = y.flatten()[~mask.flatten()]

        loss = self.loss_fn(y_hat, y)
        return loss

    @staticmethod
    def process_batch(batch):
        # in_seq, out_seq, metadata = batch
        return batch

    def training_step(self, batch, batch_idx, phase='train'):
        optimizer = self.optimizers()

        x, y, metadata = self.process_batch(batch)
        y_hat = self.forward(x)
        # loss = self._compute_loss(y_hat, y, mask=metadata['mask'])
        loss = self.calculate_loss(y_hat, y, metadata['mask'], phase=phase)
        if self.args.optimizer == 'sam':
            # optimizer = self.optimizers()

            # first forward-backward pass
            self.manual_backward(loss)  # , optimizer)
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            y_hat1 = self.forward(x)
            # loss_2 = self._compute_loss(y_hat1, y, mask=metadata['mask'])
            loss_2 = self.calculate_loss(y_hat1, y, metadata['mask'], phase=phase)
            self.manual_backward(loss_2)  # , optimizer)
            optimizer.second_step(zero_grad=True)

        self.log(f'{phase}_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def create_inference_dirs(self):
        for region_id in ['R1', 'R2', 'R3', 'R7', 'R8']:
            os.makedirs(os.path.join(self.core_dir, region_id, 'test'), exist_ok=True)
        for region_id in ['R4', 'R5', 'R6', 'R9', 'R10', 'R11']:
            os.makedirs(os.path.join(self.transfer_dir, region_id, 'test'), exist_ok=True)

    def on_validation_epoch_start(self):
        if self.args.get_prediction:
            epoch_dir = os.path.join(self.args.versiondir, 'inference', f"epoch={self.current_epoch}")
            self.core_dir = os.path.join(epoch_dir, f'core_{self.current_epoch}')
            self.transfer_dir = os.path.join(epoch_dir, f'transfer_{self.current_epoch}')
            self.create_inference_dirs()

    def on_test_epoch_start(self):
        # folder_name = 'last' if self.args.initial_epoch == -1 else f"{self.args.initial_epoch}"
        ckpt_name = str(os.path.basename(self.trainer.resume_from_checkpoint))
        # epoch_number = f"{self.trainer.current_epoch - 1}"
        epoch_number = ckpt_name.split('.')[0].split('-')[0].split('=')[-1]
        epoch_dir = os.path.join(self.args.versiondir, 'test', f"epoch={epoch_number}")
        self.core_dir = os.path.join(epoch_dir, f'core_{epoch_number}')
        self.transfer_dir = os.path.join(epoch_dir, f'transfer_{epoch_number}')
        self.create_inference_dirs()

    def save_prediction(self, y_hat, metadata):  # , batch_idx, loader_idx):
        # use sigmoid before coming here
        logit = self.calculate_logit(y_hat, mask=metadata.get('mask'), phase='val')
        logit = torch.stack(logit)
        logit = self.to_batch_first(logit)

        for idx, (region_id, day_in_year) in enumerate(zip(metadata['region_id'],
                                                           metadata['day_in_year'])):
            if region_id in ['R1', 'R2', 'R3', 'R7', 'R8']:  # 'w4c-core-stage-1'
                save_path = os.path.join(self.core_dir, region_id, 'test', f"{day_in_year}.h5")
            else:  # 'w4c-transfer-learning-stage-1'
                save_path = os.path.join(self.transfer_dir, region_id, 'test', f"{day_in_year}.h5")
            data = W4CDataset.process_output(logit[idx], collapse_time=self.args.collapse_time)
            W4CDataset.write_data(data, save_path)

    def validation_step(self, batch, batch_idx, loader_idx=0, phase='val'):

        if not self.args.get_prediction:
            x, y, metadata = batch  # self.process_batch(batch)
            y_hat = self.forward(x)
            # loss = self._compute_loss(y_hat, y, mask=metadata['mask'])
            loss = self.calculate_loss(y_hat, y, metadata['mask'], phase='val')
            self.log(f'val_loss', loss, prog_bar=True, add_dataloader_idx=False)
        else:
            if loader_idx == 0:  # for validation loader only
                x, y, metadata = batch  # self.process_batch(batch)
                y_hat = self.forward(x)
                # loss = self._compute_loss(y_hat, y, mask=metadata['mask'])
                loss = self.calculate_loss(y_hat, y, metadata['mask'], phase='val')
                self.log(f'val_loss', loss, prog_bar=True, add_dataloader_idx=False)
            else:  # for prediction
                x, metadata = batch  # self.process_batch(batch)
                y_hat = self.forward(x)
                self.save_prediction(y_hat, metadata)  # , batch_idx, loader_idx)

    def test_step(self, batch, batch_idx):  # , phase='test'):

        x, metadata = batch  # self.process_batch(batch)
        y_hat = self.forward(x)
        self.save_prediction(y_hat, metadata)  # , batch_idx, loader_idx=0)

    def configure_optimizers(self):
        other_args = {}
        print(self.args)
        scheduler = None
        # create the optimizer
        no_decay = ['absolute_pos_embed', 'relative_position_bias_table', 'norm']
        params_decay = [p for n, p in self.net.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in self.net.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, 'weight_decay': self.args.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        if self.args.optimizer == 'sgd':
            other_args = {'lr': self.args.lr, 'momentum': self.args.momentum,
                          # 'weight_decay': self.args.weight_decay,
                          'nesterov': True}
        elif self.args.optimizer == 'sam':
            other_args = {'lr': self.args.lr,
                          # 'weight_decay': self.args.weight_decay,
                          'adaptive': False,
                          'base_optimizer': torch.optim.SGD,
                          'nesterov': True,
                          'momentum': 0.9}
        elif self.args.optimizer == 'adam':
            other_args = {'lr': self.args.lr, 'eps': self.args.epsilon,
                          'betas': (self.args.beta_1, self.args.beta_2),
                          # 'weight_decay': self.args.weight_decay
                          }
        elif self.args.optimizer == 'swats':
            other_args = {'lr': self.args.lr,
                          # 'weight_decay': self.args.weight_decay,
                          'nesterov': True
                          }
        elif self.args.optimizer == 'lamb':
            other_args = {'lr': self.args.lr, 'eps': self.args.epsilon,
                          'betas': (self.args.beta_1, self.args.beta_2),
                          'clamp_value': 10,
                          # 'weight_decay': self.args.weight_decay
                          }

        # optimizer = optimizer_dict[self.args.optimizer](self.net.parameters(), **other_args)
        optimizer = optimizer_dict[self.args.optimizer](optim_groups, **other_args)

        if not hasattr(self.args, 'scheduler'):
            self.args.scheduler = 'plateau'

        if self.args.scheduler == 'cosine':
            num_gpus = self.args.gpus if isinstance(self.args.gpus, int) else len(self.args.gpus.split(','))
            t_max = math.ceil(self.args.epochs * self.args.train_dims /
                              (self.args.batch_size * num_gpus * self.args.nodes))
            scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=0),
                         'interval': 'step',  # or 'epoch'
                         }
        elif self.args.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, cooldown=0, min_lr=1e-7)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--use_group_norm', action='store_true',  # type=bool, default=False,
                            help='are we using group normalization [True, False]')
        parser.add_argument('--net_type', default='real', help='type of network',
                            choices=['sedenion', 'real', 'complex', 'quaternion', 'octonion'])
        parser.add_argument('--blk_type', default='swinunet3d', help='type of block',
                            choices=['resnet', 'densenet', 'resunet', 'denseunet', 'denseunet3p', 'dpn', 'swin', 'svt',
                                     'hybridswin', 'swinencoder', 'swinencoder3d', 'swinunet3d', 'swinuper3d', 'resunet',
                                     'hrnet', 'coatnet', 'swin2unet3d'])
        # parser.add_argument('--architecture', default='unet', help='architecture type for decoder',
        #                     choices=['unet', 'unet3p', 'unet3ph'])
        parser.add_argument('--dense_type', default='D', help='type of dense block in denseunet',
                            choices=['A', 'B', 'C', 'D'])
        parser.add_argument('--coatnet_type', default=-1, type=int, help='type of default CoAtNet backbone',
                            # choices=['A', 'B', 'C', 'D']
                            )
        parser.add_argument('--coatnet_head', default='fapn', help='type of head block in CoAtNet',
                            choices=['fapn'])
        parser.add_argument('--merge_type', default='concat', help='type of decode block',
                            choices=['concat', 'add', 'both'])
        parser.add_argument('--use_neck', action='store_true',  # type=bool, default=False,
                            help='either use unet neck or not (default: False)')
        parser.add_argument('--use_se', action='store_true',  # type=bool, default=False,
                            help='either squeeze and excitation (default: False)')
        parser.add_argument('--use_hgc', action='store_true',  # type=bool, default=False,
                            help='hirarchical mixing of time dimension (default: False)')
        parser.add_argument('--constant_dim', action='store_true',  # type=bool, default=False,
                            help='are we using constant dim (default: False)')
        parser.add_argument('--mix_features', action='store_true',  # type=bool, default=False,
                            help='are we using mixing the features by default (default: False)')
        parser.add_argument('--patch_size', type=int, default=2, help='patch size to use in swin transfer')
        parser.add_argument('--nb_layers', type=int, default=4, help='depth of resnet blocks (default: 1)')
        parser.add_argument('--decode_depth', type=int, default=None, help='depth of encoder blocks (default: None)')
        parser.add_argument('--mlp_ratio', type=int, default=4, help='mlp ratio of transformer layer (default: 4)')
        parser.add_argument('--growth_rate', type=int, default=16 * 1, help='feature map per dense layer (default: 32)')
        # working on sf divisible by 16
        parser.add_argument('--sf', type=int, default=16 * 1,  # 16 * 8,
                            help='number of feature maps (default: 16*8)')
        parser.add_argument('--sf_grp', type=int, default=1,
                            help='number of feature groups before expansion (default: 2)')
        parser.add_argument('--stages', type=int, default=3,
                            help='number of encoder stages (<1 means infer) (default:0)')
        parser.add_argument('--hidden_activation', default='elu', help='hidden layer activation')  # try gelu
        parser.add_argument('--classifier_activation', default='sigmoid',
                            help='hidden layer activation (default: hardtanh)')  # sigmoid?
        parser.add_argument('--modify_activation', type=bool, default=True,
                            help='modify the range of hardtanh activation')
        parser.add_argument('--inplace_activation', type=bool, default=True, help='inplace activation')
        parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')
        parser.add_argument('--drop_path', type=float, default=0.0, help='drop path probability')

        parser.add_argument('--optimizer', default='adam', help='optimizer to train with',
                            choices=['sgd', 'adam', 'swats', 'lamb', 'sam'])
        parser.add_argument('--scheduler', default='plateau', help='optimizer to train with',
                            choices=['plateau', 'cosine'])
        parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
        parser.add_argument('--momentum', default=0.9, type=float, help='momentum term for sgd')
        parser.add_argument('--beta_1', default=0.9, type=float, help='beta_1 term for adam')
        parser.add_argument('--beta_2', default=0.999, type=float, help='beta_2 term for adam')
        parser.add_argument('--epsilon', default=1e-8, type=float, help='epsilon term for adam')
        parser.add_argument('--weight_decay', default=1e-6, type=float,
                            help='weight decay for regularization (default: 1e-6)')

        return parser

