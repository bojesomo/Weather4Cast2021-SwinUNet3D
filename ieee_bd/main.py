import pathlib
import sys
import os

module_dir = str(pathlib.Path(os.getcwd()))
sys.path.append(module_dir)

import re
import argparse
import warnings

import numpy as np
import pandas as pd
import time
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
import torch

from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin
from pytorch_lightning.loggers import CSVLogger

from model_pl import Model, get_model
from utils import model_summary
from dataset import W4CDataset


def get_held_out_params(params):
    held_out_params = params
    # print(params.keys())
    old_path = held_out_params['data_path']
    paths = re.split('/|\\\\', old_path)
    paths[-2] += '-heldout'
    new_path = f'{os.sep}'.join(paths)
    held_out_params['data_path'] = new_path
    return held_out_params


class DataModule(pl.LightningDataModule):
    """ Class to handle training/validation splits in a single object
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_dims = None
        # self.target_variables_properties = None
        self.in_depth = None
        self.precision = args.precision
        self.augment_data = args.augment_data

        self.train = self.val = self.predict = self.held_out = None

    def setup(self):
        start_time = time.time()
        if self.args.use_all_region:
            train_datasets = []
            val_datasets = []
            predict_datasets = []
            held_out_datasets = []
            core_regions = ['R1', 'R2', 'R3']
            all_regions = [f"R{i + 1}" for i in range(6)]
            if self.args.competition == 'ieee-bd':
                core_regions.extend(['R7', 'R8'])
                all_regions.extend([f"R{i + 1}" for i in range(6, 11)])

            for region_id in core_regions:
                train_datasets.append(W4CDataset(region_id=region_id, use_all_products=self.args.use_all_products,
                                                 data_root=self.args.data_root, use_static=self.args.use_static,augment_data=self.args.augment_data,
                                                 collapse_time=self.args.collapse_time, stage='training'))
                val_datasets.append(W4CDataset(region_id=region_id, use_all_products=self.args.use_all_products,
                                               data_root=self.args.data_root, use_static=self.args.use_static,
                                               collapse_time=self.args.collapse_time, stage='validation'))

            for region_id in all_regions:
                predict_datasets.append(W4CDataset(region_id=region_id, use_static=self.args.use_static,
                                                   data_root=self.args.data_root,
                                                   use_all_products=self.args.use_all_products,
                                                   collapse_time=self.args.collapse_time,
                                                   stage='heldout' if self.args.held_out else 'test',
                                                   ))
                #predict_datasets.append(W4CDataset(region_id=region_id, use_all_products=self.args.use_all_products,
                #                                   data_root=self.args.data_root, use_static=self.args.use_static,
                #                                   collapse_time=self.args.collapse_time, stage='test'))
                #if self.args.held_out:  # using held-out data
                #    held_out_datasets.append(W4CDataset(region_id=region_id, use_all_products=self.args.use_all_products,
                #                                        data_root=self.args.data_root, use_static=self.args.use_static,
                #                                        collapse_time=self.args.collapse_time, stage='test'))

            self.train = ConcatDataset(train_datasets)
            self.val = ConcatDataset(val_datasets)
            self.predict = ConcatDataset(predict_datasets)
            #if self.args.held_out:  # using held-out data
            #    self.held_out = ConcatDataset(held_out_datasets)
            # self.target_variables_properties = train_datasets[0].target_variable_properties
            self.in_depth = train_datasets[0].in_depth  # len(train_datasets[0].variable_properties)
        else:
            self.train = W4CDataset(region_id=self.args.region, use_static=self.args.use_static,augment_data=self.args.augment_data,
                                    data_root=self.args.data_root, use_all_products=self.args.use_all_products,
                                    collapse_time=self.args.collapse_time, stage='training')
            self.val = W4CDataset(region_id=self.args.region, use_static=self.args.use_static,
                                  data_root=self.args.data_root, use_all_products=self.args.use_all_products,
                                  collapse_time=self.args.collapse_time, stage='validation')
            #self.predict = W4CDataset(region_id=self.args.region, use_static=self.args.use_static,
            #                          data_root=self.args.data_root, use_all_products=self.args.use_all_products,
            #                          collapse_time=self.args.collapse_time, stage='test')
            #if self.args.held_out:  # using held-out data
            #    self.held_out = W4CDataset(region_id=self.args.region, use_static=self.args.use_static,
            #                               data_root=self.args.data_root, use_all_products=self.args.use_all_products,
            #                               collapse_time=self.args.collapse_time, stage='test')
            self.predict = W4CDataset(region_id=self.args.region, use_static=self.args.use_static,
                                      data_root=self.args.data_root, use_all_products=self.args.use_all_products,
                                      collapse_time=self.args.collapse_time,
                                      stage='heldout' if self.args.held_out else 'test',
                                      )
            self.in_depth = self.train.in_depth  # len(self.train.variable_properties)

        self.train_dims = self.train.__len__()
        print(f"Datasets loaded in {time.time() - start_time} seconds")

    def __load_dataloader(self, dataset, shuffle=True, pin=True):
        dl = DataLoader(dataset,
                        batch_size=self.args.batch_size, num_workers=self.args.workers,
                        shuffle=shuffle, pin_memory=pin)
        return dl

    def train_dataloader(self):
        ds = self.train  # create_dataset('training', self.params)
        return self.__load_dataloader(ds, shuffle=True, pin=True)

    def val_dataloader(self):
        val_loader = self.__load_dataloader(self.val, shuffle=False, pin=True)
        if not self.args.get_prediction:
            return val_loader  # [val_loader]
        else:
            predict_loader = self.__load_dataloader(self.predict, shuffle=False, pin=True)
            return [val_loader, predict_loader]

    def test_dataloader(self):
        #if self.args.held_out:  # using held-out data
        #    predict_loader = self.__load_dataloader(self.held_out, shuffle=False, pin=True)
        #else:
        #    predict_loader = self.__load_dataloader(self.predict, shuffle=False, pin=True)
        predict_loader = self.__load_dataloader(self.predict, shuffle=False, pin=True)
        return predict_loader

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_root', type=str, default='/l/proj/kuex0005/Alabi/Datasets/Weather4Cast2021/',
                            help='root directory of the data')
        parser.add_argument("-r", "--region", type=str, required=False, default='R1',
                            help="region_id to load data from. R1 as default")
        parser.add_argument("-a", "--use_all_region", action='store_true',  # type=bool, required=False, default=False,
                            help="use all region")
        # parser.add_argument("-v", "--target_variable", type=str, required=False, default=None,
        #                     help="which variable to trainthe network for",
        #                     choices={'temp', 'crr', 'asii', 'cma'})
        parser.add_argument("-ho", "--held-out", action='store_true',  # type=bool, required=False, default=False,
                            help="are we using held-out dataset for the 'test'")
        parser.add_argument('--use_all_products', action='store_true', help='use all variable (Default: False)')
        parser.add_argument('--use_static', action='store_true', help='use static variable (Default: Fasle)')
        # parser.add_argument('--use_time_slot', type=bool, default=False, help='use time slots (Default: True)')
        parser.add_argument('--augment_data', action='store_true',  # type=bool, default=True,
                            help='use data augmentation to reduce over-fitting')

        return parser


def print_training(params):
    """ print pre-training info """

    print(f'Extra variables: {params["extra_data"]} | spatial_dim: {params["spatial_dim"]} ',
          f'| collapse_time: {params["collapse_time"]} | in channels depth: {params["depth"]} | len_seq_in: {params["len_seq_in"]}')


def load_model(Model, params, options,
               checkpoint_path=''):
    """ loads a model from a checkpoint or from scratch if checkpoint_path='' """
    if checkpoint_path == '':
        print('-> model from scratch!')
        # model = Model(params['model_params'], **params['data_params'])
        model = Model(options, **params['data_params'])
    else:
        print(f'-> Loading model checkpoint: {checkpoint_path}')
        model = Model.load_from_checkpoint(checkpoint_path)
    return model


def modify_options(options, n_params):
    filename = '_'.join(
        [f"{item}" for item in ('ALL' if options.use_all_region else options.region, options.net_type, options.blk_type,
                                int(n_params))])
    options.filename = options.name or filename  # to account for resuming from a previous state

    options.versiondir = os.path.join(options.log_dir, options.filename, options.time_code)
    os.makedirs(options.versiondir, exist_ok=True)
    readme_file = os.path.join(options.versiondir, 'options.csv')
    args_dict = vars(argparse.Namespace(**{'modelname': options.filename, 'num_params': n_params}, **vars(options)))
    args_df = pd.DataFrame([args_dict])
    if os.path.exists(readme_file):
        args_df.to_csv(readme_file, mode='a', index=False, header=False)
    else:
        args_df.to_csv(readme_file, mode='a', index=False)

    return options


def save_options(options, n_params):
    options.versiondir = os.path.join(options.log_dir, options.filename, options.time_code)
    os.makedirs(options.versiondir, exist_ok=True)
    readme_file = os.path.join(options.versiondir, 'options.csv')
    args_dict = vars(argparse.Namespace(**{'modelname': options.filename, 'num_params': n_params}, **vars(options)))
    args_df = pd.DataFrame([args_dict])
    if os.path.exists(readme_file):
        args_df.to_csv(readme_file, mode='a', index=False, header=False)
    else:
        args_df.to_csv(readme_file, mode='a', index=False)
    return options


def get_trainer(options):  # gpus, max_epochs=20):
    """ get the trainer, modify here it's options:
        - save_top_k
        - max_epochs
     """
    lr_monitor = LearningRateMonitor(logging_interval='step')

    early_stop_callback = EarlyStopping(
        monitor='val_loss',  # should be found in logs
        patience=20,  # 3,
        strict=False,  # will act as disabled if monitor not found
        verbose=False,
        mode='min'
    )

    logger = CSVLogger(save_dir=options.log_dir,
                       name=options.filename,
                       version=options.time_code,
                       )  # time_code)

    resume_from_checkpoint = None
    if options.name and options.time_code:
        # resume_from_checkpoint = os.path.join(options.versiondir, 'checkpoints', 'last.ckpt')
        checkpoint_dir = os.path.join(options.versiondir, 'checkpoints')
        if options.initial_epoch == -1:
            checkpoint_name = 'last.ckpt'
        else:
            format_str = f"epoch={options.initial_epoch:03g}"
            checkpoint_names = os.listdir(checkpoint_dir)
            checkpoint_name = checkpoint_names[[t.startswith(format_str) for t in checkpoint_names].index(True)]
        resume_from_checkpoint = os.path.join(checkpoint_dir, checkpoint_name)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=10,
                                          save_last=True, verbose=False,
                                          # filename='{epoch:02d}-{val_loss:.6f}',
                                          filename='{epoch:02d}-{val_loss:.6f}'
                                          )

    callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]

    trainer = pl.Trainer(gpus=options.gpus,
                         max_epochs=options.epochs,
                         progress_bar_refresh_rate=100,  # 80,
                         deterministic=True,
                         # accumulate_grad_batches=5,  # 10,  # to stylishly increase the batch size
                         gradient_clip_val=1 if options.optimizer != 'sam' else 0.0,
                         # to clip gradient value and prevent exploding gradient
                         # gradient_clip_algorithm='value',
                         # stochastic_weight_avg=True,  # smooth loss to prevent local minimal
                         default_root_dir=os.path.dirname(options.log_dir),
                         #limit_train_batches=20, limit_val_batches=10, #limit_test_batches=10,
                         # fast_dev_run=True,
                         callbacks=callbacks,
                         profiler='simple',
                         sync_batchnorm=True,
                         num_sanity_val_steps=0,
                         # accelerator='ddp_sharded_spawn',
                         logger=logger,
                         resume_from_checkpoint=resume_from_checkpoint,
                         num_nodes=options.nodes,
                         plugins=DDPPlugin(num_nodes=options.nodes, find_unused_parameters=False),
                         # plugins=DeepSpeedPlugin(),
                         # plugins=DeepSpeedPlugin(stage=3, cpu_offload=True, partition_activations=True),
                         precision=options.precision,
                         # move_metrics_to_cpu=True,  # to avoid metric related GPU  memory bottleneck
                         # distributed_backend='ddp',
                         )

    return trainer


def do_test(trainer, model, test_data):
    print("-----------------")
    print("--- TEST MODE ---")
    print("-----------------")
    scores = trainer.test(model, test_dataloaders=test_data)


def train(options):  # gpus, region_id, mode, checkpoint_path, options=None):
    """ main training/evaluation method
    """

    # some needed stuffs
    warnings.filterwarnings("ignore")

    pl.seed_everything(options.manual_seed, workers=True)
    torch.manual_seed(options.manual_seed)
    torch.cuda.manual_seed_all(options.manual_seed)

    # ------------
    # Data and model params
    # ------------
    options.collapse_time = not options.blk_type.lower().endswith('3d')
    # options.target_variable = {'temp': 'temperature', 'crr': 'crr_intensity',
    #                            'asii': 'asii_turb_trop_prob', 'cma': 'cma'}[options.target_variable]
    data = DataModule(options)
    data.setup()

    # add other depending args
    options.train_dims = data.train_dims
    # options.target_variables_properties = data.target_variables_properties
    # print(type(options.target_variables_properties))
    options.height = options.width = 256
    options.in_depth = data.in_depth  # 25
    options.out_depth = 4
    options.len_seq_in = 4
    options.len_seq_out = 32
    options.in_channels = options.len_seq_in * options.in_depth
    options.n_classes = options.len_seq_out * options.out_depth
    # options.in_depth += 2 * options.use_static + options.use_time_slot

    # let's load model for printing structure
    # print(options.constant_dim)
    model = get_model(options)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x_all = torch.rand(1, options.len_seq_in * options.in_depth, 256, 256) if options.collapse_time else \
        torch.rand(1, options.len_seq_in, options.in_depth, 256, 256)

    # print(f"n_params={n_params}, x_shape={x_all.shape}")
    # print(options)
    # del model
    # model.eval()
    # # print(model)
    _ = model_summary(model.eval(), x_all, print_summary=True, max_depth=1)  #2 if options.blk_type.startswith('coat') else 1)
    # s = summary(model, input_size=input_size, col_names=("input_size", "output_size", "num_params", "mult_adds"),
    #             depth=1, device='cpu')
    # print(s)
    # # _ = model_summary(model, x_all, print_summary=True, max_depth=0)
    # # print(model.training)
    del model, x_all
    #raise ValueError()  # stop here for now
    '''
    python ieee_bd/main1.py --use_all_region  --log-dir ieee_bd/separate --precision 32 --net_type real --blk_type coatnet --sf 128 --stages 4  --coatnet_type 0 --workers 12 --optimizer adam --scheduler plateau  --lr 1e-4 --augment_data --constant_dim --target_variable cma --batch-size 4

    '''
    # ------------
    # trainer
    # ------------
    options = modify_options(options, n_params)
    # options = save_options(options, n_params)
    trainer = get_trainer(options)
    print(options)

    # ------
    # Model
    # -----
    # # model = load_model(Model, params, checkpoint_path)
    # model = Model(options, **params['data_params'])  # load_model(Model, params, options, checkpoint_path)
    # print(model)
    checkpoint_path = trainer.resume_from_checkpoint
    if checkpoint_path is not None:
        model = Model.load_from_checkpoint(checkpoint_path)
    else:
        model = Model(options)

    print(options)
    # ------------
    # train & final validation
    # ------------
    if options.mode == 'train':
        print("-----------------")
        print("-- TRAIN MODE ---")
        print("-----------------")
        trainer.fit(model, datamodule=data)
    else:
        print("-----------------")
        print("--- TEST MODE ---")
        print("-----------------")
        trainer.test(model, datamodule=data)

    # # validate
    # do_test(trainer, model, data.val_dataloader())


def add_main_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("-g", "--gpus", type=int,
                        required=False, default=4,
                        help="specify number of gpus per node")
    parser.add_argument("-m", "--mode", type=str, required=False, default='train',
                        help="choose mode: train (default)  / val")
    parser.add_argument("-n", "--nodes", type=int, required=False, default=1,
                        help="number of nodes used for training (hpc)")
    parser.add_argument("-c", "--checkpoint", type=str, required=False, default='',
                        help="init a model from a checkpoint path. '' as default (random weights)")
    parser.add_argument("-j", "--workers", type=int, required=False, default=8,
                        help="number of workers")
    parser.add_argument("--get_prediction", action='store_true',
                        help='are we interested in prediction during training')
    parser.add_argument('--competition', default='ieee-bd', help='competition name', choices=['stage-1', 'ieee-bd'])

    parser.add_argument('--precision', type=int, default=16, help='precision to use for training', choices=[16, 32])
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=1, help='batch- size')

    parser.add_argument('--manual-seed', default=0, type=int, help='manual global seed')
    parser.add_argument('--log-dir', default='/l/proj/kuex0005/Alabi/logs/IEEE-BD', help='base directory to save logs')
    # parser.add_argument('--model-dir', default='', help='base directory to save logs')
    parser.add_argument('--name', default='',  # 'R1_real_swinencoder3d_776228',
                        help='identifier for model if already exist')
    parser.add_argument('--time-code', default='',  # '20210903T224723',
                        help='identifier for model if already exist')
    parser.add_argument('--initial-epoch', type=int, default=-1, help='number of epochs done (-1 == last)')
    parser.add_argument('--memory_efficient', action='store_true',  # type=bool, default=True,
                        help='memory_efficient')
    return parser

'''
python ieee_bd/main2.py --nodes 1 --gpus 4 --blk_type swinunet3d --stages 4 --patch_size 2 --sf 128 --nb_layers 4  --use_neck --use_all_region --lr 1e-4 --optimizer adam --scheduler plateau --merge_type both  --mlp_ratio 2 --decode_depth 2 --precision 32 --epoch 100 --batch-size 4 --augment_data  --constant_dim --workers 12 --get_prediction

'''
def get_time_code():
    time_now = [f"{'0' if len(x) < 2 else ''}{x}" for x in np.array(time.localtime(), dtype=str)][:6]
    if os.path.exists('t.npy'):
        time_before = np.load('t.npy')  # .astype(np.int)
        if abs(int(''.join(time_before)) - int(''.join(time_now))) < 90:
            time_now = time_before
        else:
            np.save('t.npy', time_now)
    else:
        np.save('t.npy', time_now)
    time_now = ''.join(time_now[:3]) + 'T' + ''.join(time_now[3:])
    return time_now


def main():
    parser = argparse.ArgumentParser(description="Weather4Cast Arguments")
    parser = add_main_args(parser)
    parser = Model.add_model_specific_args(parser)
    parser = DataModule.add_data_specific_args(parser)
    options = parser.parse_args()

    options.region = options.region.upper()

    time_code = get_time_code()
    options.time_code = options.time_code or time_code  # to account for resuming from a previous state

    # train(options.gpu_ids, options.region, options.mode, options.checkpoint)
    train(options)  # .gpus, options.region, options.mode, options.checkpoint, options)


if __name__ == "__main__":
    main()