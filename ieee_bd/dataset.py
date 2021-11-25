import numpy as np
from torch.utils.data import Dataset
import os
import glob
import netCDF4
from collections import OrderedDict
from einops import rearrange
from pathlib import Path
import pandas as pd
import datetime
import h5py
import json
from tqdm import tqdm
from data_analysis import extract_info, extract_test
from static import StaticData


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.axis = -2
        # self.fn = vflip

    def __call__(self, sample):
        in_seq1, out_seq1, metadata = sample
        if np.random.rand() < self.p:
            in_seq = np.flip(in_seq1, axis=self.axis).copy()
            out_seq = np.flip(out_seq1, axis=self.axis).copy()
            del in_seq1, out_seq1
            # print(in_seq.shape, out_seq.shape)
            if 'mask' in metadata:
                mask = metadata.pop('mask')
                metadata['mask'] = np.flip(mask, axis=self.axis).copy()
                del mask

            return in_seq, out_seq, metadata
        else:
            return in_seq1, out_seq1, metadata

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomHorizontalFlip(RandomVerticalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)
        self.axis = -1


class RandomRotate(object):
    def __init__(self):
        super().__init__()
        self.axes = (-2, -1)

    def __call__(self, sample):
        in_seq1, out_seq1, metadata = sample

        rotation_count = np.random.randint(4)
        in_seq = np.rot90(in_seq1, k=rotation_count, axes=self.axes).copy()
        out_seq = np.rot90(out_seq1, k=rotation_count, axes=self.axes).copy()
        del in_seq1, out_seq1
        # print(in_seq.shape, out_seq.shape)
        if 'mask' in metadata:
            mask = metadata.pop('mask')
            metadata['mask'] = np.rot90(mask, k=rotation_count, axes=self.axes).copy()
            del mask
        return in_seq, out_seq, metadata



class W4CDataset(Dataset):
    def __init__(self,
                 region_id='R1',
                 data_root='/home/farhanakram/Alabi/Datasets/Weather4cast2021/',
                 # sampling_step=1,
                 stage='training',
                 collapse_time=False,  # are we using 2D model (by collapsing the time axis ?
                 sanity_check=False,  # ensure if all valid data point exist at initialization
                 target_variable=None,
                 use_all_products=True,
                 use_static=False,
                 augment_data=False,
                 ):
        super().__init__()
        assert stage in ['training', 'validation', 'test', 'heldout']
        self.n_frame_in = 4
        self.n_frame_out = 32
        self.height = self.width = 256
        # self.sampling_step = sampling_step
        self.stage = stage if stage != 'heldout' else 'test'
        self.region_id = region_id
        self.data_root = data_root
        self.collapse_time = collapse_time

        self.augmentation = None
        if stage == 'training' and augment_data:
            self.augmentation = [RandomVerticalFlip(), RandomHorizontalFlip(), RandomRotate()]
        
        # ------------
        # 1. Files to load
        # ------------
        if region_id in ['R1', 'R2', 'R3', 'R7', 'R8']:
            track = 'ieee-bd-core'
        else:
            track = 'ieee-bd-transfer-learning'

        self.weather_products = OrderedDict()
        if use_all_products:
            self.weather_products['CTTH'] = ['temperature', 'ctth_tempe', 'ctth_pres', 'ctth_alti', 'ctth_effectiv',
                                              'ishai_skt']
            self.weather_products['CRR'] = ['crr', 'crr_intensity', 'crr_accum']
            self.weather_products['ASII'] = ['asii_turb_trop_prob']  # perhaps TF means true/false
            self.weather_products['CMA'] = ['cma_cloudsnow', 'cma', 'cma_dust', 'cma_volcanic', 'cma_smoke']
            self.weather_products['CT'] = ['ct', 'ct_cumuliform', 'ct_multilayer']
        else:
            self.weather_products['CTTH'] = ['temperature']
            self.weather_products['CRR'] = ['crr_intensity']
            self.weather_products['ASII'] = ['asii_turb_trop_prob']  # perhaps TF means true/false
            self.weather_products['CMA'] = ['cma']

        self.in_depth = sum(len(t) for t in self.weather_products.values())
        
        self.static_data = None
        if use_static:
            self.static_products = ["elevation", "longitude", "latitude"]
            self.in_depth += 3
            self.static_data = StaticData(data_dir=os.path.join(data_root,
                                                                'statics')).get_data(region=region_id,
                                                                                     variables=self.static_products)
            self.static_mask = np.zeros((3, 256, 256)).astype(bool)
        self.target_variables = ['temperature', 'crr_intensity', 'asii_turb_trop_prob', 'cma']

        if target_variable is not None:
            # print(target_variable)
            self.target_variables = [target_variable]  # if provided

        self.variable_properties = {
            # CMA
            'cma_cloudsnow': {'fill_value': 0, 'max_value': 3, 'add_offset': 0, 'scale_factor': 1},
            'cma': {'fill_value': 0, 'max_value': 1, 'add_offset': 0, 'scale_factor': 1},
            'cma_dust': {'fill_value': 0, 'max_value': 2, 'add_offset': 0, 'scale_factor': 1},
            'cma_volcanic': {'fill_value': 0, 'max_value': 2, 'add_offset': 0, 'scale_factor': 1},
            'cma_smoke': {'fill_value': 0, 'max_value': 2, 'add_offset': 0, 'scale_factor': 1},
            # CTTH
            'temperature': {'fill_value': 0, 'max_value': 35000, 'add_offset': 130, 'scale_factor': np.float32(0.01)},
            'ctth_tempe': {'fill_value': 0, 'max_value': 35000, 'add_offset': 130, 'scale_factor': np.float32(0.01)},
            'ctth_pres': {'fill_value': 0, 'max_value': 11000, 'add_offset': 0, 'scale_factor': np.float32(10)},
            'ctth_alti': {'fill_value': 0, 'max_value': 25000, 'add_offset': -2000, 'scale_factor': np.float32(1)},
            'ctth_effectiv': {'fill_value': 0, 'max_value': 100, 'add_offset': 0, 'scale_factor': np.float32(0.01)},
            'ishai_skt': {'fill_value': 0, 'max_value': 34300, 'add_offset': 0, 'scale_factor': np.float32(0.01)},
            # CRR
            'crr_intensity': {'fill_value': 0, 'max_value': 500, 'add_offset': 0, 'scale_factor': np.float32(0.1)},
            'crr': {'fill_value': 0, 'max_value': 11, 'add_offset': 0, 'scale_factor': np.float32(1)},
            'crr_accum': {'fill_value': 0, 'max_value': 500, 'add_offset': 0, 'scale_factor': np.float32(0.1)},
            # ASII
            'asii_turb_trop_prob': {'fill_value': 0, 'max_value': 100, 'add_offset': 0, 'scale_factor': 1},
            # CT
            'ct': {'fill_value': 0, 'max_value': 15, 'add_offset': 0, 'scale_factor': 1},
            'ct_cumuliform': {'fill_value': 0, 'max_value': 5, 'add_offset': 0, 'scale_factor': 1},
            'ct_multilayer': {'fill_value': 0, 'max_value': 3, 'add_offset': 0, 'scale_factor': 1},
        }
        # self.target_variables_properties = {variable: {} for variable in self.target_variables}
        # self.variables = OrderedDict()

        # update to accomodate heldout
        if stage == 'heldout':
            track = 'heldout'
            
        self.data_path = os.path.join(self.data_root, track, self.region_id, self.stage)

        save_path = os.path.join(self.data_root, track, self.region_id)
        info_file = os.path.join(save_path, f"{region_id}_{stage}_info.csv")
        # blacklist_path = os.path.join(save_path, f"{region_id}_{stage}_blacklist.csv")

        if os.path.exists(info_file):
            self.df = pd.read_csv(info_file, dtype={'date': str, 'time': str, 'day': str,
                                                    'number_stamp': int, 'used': bool})
        else:
            # print(f"info_file saved to {info_file}")
            if self.stage == 'test':
                self.df = extract_test(region=self.region_id, root=self.data_root, stage=stage)
            else:
                self.df, _ = extract_info(region=self.region_id, root=self.data_root, stage=self.stage)
        self.indices = self.df[self.df['used']].index.to_list()

        if sanity_check:
            for idx in tqdm(range(len(self.indices))):
                self.check_sample(idx)
            print(f"{region_id}|{stage} successfully checked")

    def __len__(self):
        return len(self.indices)

    def all_paths(self, product, day, date, time_code):
        product_code = product if product != 'ASII' else 'ASII-TF'
        filename = f'S_NWC_{product_code}_MSG*_Europe-VISIR_{date}T{time_code}Z.nc'  # could be MSG2 or MSG4
        file_path = glob.glob(f"{self.data_path}/{day}/{product}/{filename}")
        return file_path

    # @staticmethod
    def _filepath(self, product, idx):
        row = self.df.iloc[idx]
        # print(row)
        day = row['day']
        date = row['date']
        product_code = product if product != 'ASII' else 'ASII-TF'
        time_code = row['time']
        filename = f'S_NWC_{product_code}_MSG*_Europe-VISIR_{date}T{time_code}Z.nc'  # could be MSG2 or MSG4
        # print(filename)
        file_path = glob.glob(f"{self.data_path}/{day}/{product}/{filename}")
        try:
            assert len(file_path) == 1, f" Error with file in {file_path} ----> all these files were found: {file_path}"
        except:
            AssertionError(print(f"time_idx={self.time_idx}  :  {self.data_path}/{day}/{product}/{filename}"))
        file_path = file_path[0]
        return file_path

    @staticmethod
    def postprocess(data, fill_value, max_value, add_offset, scale_factor):
        """ scales 'v' to the original scale ready to save into disk
        """

        # 1. scale data to the original range
        data = data * (max_value * scale_factor - add_offset) + add_offset

        # 2. pack the variable into an uint16 valid range (as netCDF does)
        # shttps://unidata.github.io/netcdf4-python/#Variable.set_auto_maskandscale
        data = (data - add_offset) / scale_factor

        # 3. Cast the data to integer
        # this step must be used to get back the original integer saved in the input files
        data = np.uint16(np.round(data, 0))
        assert data.max() <= max_value, f"Error, postprocess of the variables is wrong"

        return data

    @staticmethod
    def postprocess_fn(data):
        target_variables = ['temperature', 'crr_intensity', 'asii_turb_trop_prob', 'cma']
        target_variables_properties = {
            # CMA
            'cma': {'fill_value': 0, 'max_value': 1, 'add_offset': 0, 'scale_factor': 1},
            # CTTH
            'temperature': {'fill_value': 0, 'max_value': 35000, 'add_offset': 130, 'scale_factor': np.float32(0.01)},
            # CRR
            'crr_intensity': {'fill_value': 0, 'max_value': 500, 'add_offset': 0, 'scale_factor': np.float32(0.1)},
            # ASII
            'asii_turb_trop_prob': {'fill_value': 0, 'max_value': 100, 'add_offset': 0, 'scale_factor': 1},
        }
        """ post process each variable separately """
        for i, tgt_var in enumerate(target_variables):
            data[:, i] = W4CDataset.postprocess(data[:, i], **target_variables_properties[tgt_var])
            if tgt_var == 'cma':
                data[:, i][data[:, i] < 0.5] = 0
                data[:, i][data[:, i] >= 0.5] = 1

        # data = data.astype(np.uint16)
        return data

    @staticmethod
    def preprocess_fn(data, fill_value, max_value, add_offset, scale_factor):
        """ returns a processed numpy array
            from a given numpy masked array 'v'
        """
        data = np.float32(data)

        # scale it into [0, 1]
        data = (data - add_offset) / (max_value * scale_factor - add_offset)

        # fill NaNs with 'fill_value'
        data = data.filled(fill_value)

        assert 0 <= np.nanmin(data) and np.nanmax(data) <= 1, f"Error, the scale of the variables is wrong"

        return data

    def format_input(self, ds, variable):
        data_arr = ds.variables[variable][...]
        if isinstance(data_arr.mask, np.bool_):
            data_arr.mask = np.zeros(data_arr.shape)
        mask_arr = data_arr.mask
        # changed the mask from 0 active to 1 active

        data_arr = self.preprocess_fn(data_arr, **self.variable_properties[variable])

        return data_arr, mask_arr

    def extract_variables(self, ds, all_variables, variables=None):  # =None):

        variables = variables or all_variables
        data = []
        mask = []
        for variable in all_variables:
            if variable in variables:
                data_arr, mask_arr = self.format_input(ds, variable)
                data.append(data_arr)
                mask.append(mask_arr)
        if len(data) != 0:
            data, mask = np.stack(data), np.stack(mask)
        return data, mask

    def check_frame(self, time_idx, category='input'):
        time_idx += 0 if category == 'input' else self.n_frame_in
        time_extended = self.n_frame_in if category == 'input' else self.n_frame_out
        for idx in range(time_idx, time_idx + time_extended):
            row = self.df.iloc[idx]
            all_paths = self.all_paths('*', row['day'], row['date'], row['time'])
            blacklist = len(all_paths) < 5
            if blacklist:
                ValueError(f"error for time_idx={time_idx}: idx={idx}: category={category}")

    def get_frame(self, time_idx, category='input'):
        frame_data = []
        frame_mask = []
        time_idx += 0 if category == 'input' else self.n_frame_in
        time_extended = self.n_frame_in if category == 'input' else self.n_frame_out
        variables = None if category == 'input' else self.target_variables
        # if category != 'input':
        #     print(variables)
        for idx in range(time_idx, time_idx + time_extended):
            products_data = []
            products_mask = []
            # present_day = self.df.iloc[idx]['day']
            present_time_idx = idx
            # present_time_idx, present_day = self.df.iloc[idx]  # self.get_terms(idx, day)
            for product in self.weather_products:
                filepath = self._filepath(product, present_time_idx)
                ds = netCDF4.Dataset(filepath, 'r')
                all_variables = self.weather_products[product]
                data, mask = self.extract_variables(ds, all_variables=all_variables, variables=variables)
                if len(data) != 0:
                    products_data.append(data)
                    products_mask.append(mask)
            # print([t.shape for t in products_data], category)
            # if len(products_data) != 0:
            if self.static_data is not None and category != 'label':
                products_data.append(self.static_data)
                products_mask.append(self.static_mask)
            products_data = np.concatenate(products_data, axis=0)
            products_mask = np.concatenate(products_mask, axis=0)
            # print(products_data.shape, category)
            frame_data.append(products_data)
            frame_mask.append(products_mask)
        frame_data = np.stack(frame_data)
        frame_mask = np.stack(frame_mask)

        frame_data = W4CDataset.rearrange_input(frame_data, collapse_time=self.collapse_time)
        frame_mask = W4CDataset.rearrange_input(frame_mask, collapse_time=self.collapse_time)
        return frame_data, frame_mask

    @staticmethod
    def rearrange_input(data, collapse_time=True):
        if collapse_time:
            return rearrange(data, 'f d h w -> (f d) h w')
        return data

    @staticmethod
    def project_output(data, target_variables, target_variables_properties):
        for idx, variable in enumerate(target_variables):
            properties = target_variables_properties[variable]
            # print(variable, properties, data.max())
            if variable == 'cma':  # the only discrete output
                data[idx][data[idx] < 0.5] = 0
                data[idx][data[idx] >= 0.5] = 1
            else:
                scale_factor = properties['scale_factor']
                add_offset = properties['add_offset']
                fill_value = properties['fill_value']
                max_value = properties['max_value']

                # 1. scale data to the original range
                data[idx] = data[idx] * (max_value * scale_factor - add_offset) + add_offset

                # 2. pack the variable into an uint16 valid range (as netCDF does)
                # shttps://unidata.github.io/netcdf4-python/#Variable.set_auto_maskandscale
                data[idx] = (data[idx] - add_offset) / scale_factor

                # 3. Cast the data to integer
                # this step must be used to get back the original integer saved in the input files
                data[idx] = np.uint16(np.round(data[idx], 0))
                assert data[idx].max() <= max_value, f"Error, postprocess of the variables is wrong"

        return data

    @staticmethod
    def process_output(data, n_frame_out=32, collapse_time=True):
        target_variables = ['temperature', 'crr_intensity', 'asii_turb_trop_prob', 'cma']
        target_variables_properties = {
            # CMA
            'cma': {'fill_value': 0, 'max_value': 1, 'add_offset': 0, 'scale_factor': 1},
            # CTTH
            'temperature': {'fill_value': 0, 'max_value': 35000, 'add_offset': 130, 'scale_factor': np.float32(0.01)},
            # CRR
            'crr_intensity': {'fill_value': 0, 'max_value': 500, 'add_offset': 0, 'scale_factor': np.float32(0.1)},
            # ASII
            'asii_turb_trop_prob': {'fill_value': 0, 'max_value': 100, 'add_offset': 0, 'scale_factor': 1},
        }
        data = data.cpu().numpy()  # first turn to numpy
        if collapse_time:
            data = rearrange(data, '(f d) h w -> f d h w', f=n_frame_out)

        data = rearrange(data, 'f d h w -> d f h w')
        data = W4CDataset.project_output(data, target_variables,  target_variables_properties)
        data = rearrange(data, 'd f h w -> f d h w')
        return data

    def do_augmentation(self, sample):
        if self.augmentation is not None:
            for augment_fn in self.augmentation:
                sample = augment_fn(sample)
        return sample

    @staticmethod
    def write_data(data, filename):
        """ write data in gzipped h5 format with type uint16 """
        # f = h5py.File(filename, 'w', libver='latest')
        # dset = f.create_dataset('array', shape=(data.shape), data=data, dtype=np.uint16, compression='gzip',
        #                         compression_opts=9)
        # f.close()
        with h5py.File(filename, 'w', libver='latest') as f:
            _ = f.create_dataset('array', shape=data.shape, data=data, dtype=np.uint16,
                                 compression='gzip', compression_opts=9)

    @staticmethod
    def write_dict(data_dict, file_path):
        """
        write data in gzipped h5 format.
        """
        with h5py.File(file_path, 'w', libver='latest') as f:
            for name, data_in in data_dict.items():
                f.create_dataset(name,
                                 shape=data_in.shape,
                                 data=data_in,
                                 # chunks=(1, *data_in.shape[1:]),
                                 compression='gzip',
                                 dtype=data_in.dtype,
                                 compression_opts=9)

    def check_data(self, time_idx):
        # let's load the input
        self.check_frame(time_idx, category='input')
        if self.stage in ['training', 'validation']:
            # let's load the label
            self.check_frame(time_idx, category='label')

    def get_data(self, time_idx, check=False):
        # let's load the input
        input_data, input_mask = self.get_frame(time_idx, category='input')
        label_data, label_mask = (None, None)
        if self.stage in ['training', 'validation']:
            # let's load the label
            label_data, label_mask = self.get_frame(time_idx, category='label')

        return (input_data, input_mask), (label_data, label_mask)

    def __getitem__(self, index):
        if index < 0:
            index += self.__len__()
        time_idx = self.indices[index]
        self.time_idx = time_idx
        row = self.df.iloc[time_idx]
        (input_data, input_mask), (label_data, label_mask) = self.get_data(time_idx)
        # input_data *= input_mask
        # if label_data is not None:
        #     label_data *= label_mask
        # metadata = {'region_id': self.region_id, 'day_in_year': row['day'], 'mask': label_mask}
        if label_data is not None:
            metadata = {'region_id': self.region_id, 'day_in_year': row['day'], 'mask': label_mask}
            sample = input_data, label_data, metadata
            return self.do_augmentation(sample)
            #return input_data, label_data, metadata
        else:
            # label_mask = input_mask[:, self.target_idx, ...]
            metadata = {'region_id': self.region_id, 'day_in_year': row['day']}  # , 'mask': label_mask}
            return input_data, metadata

    def check_sample(self, index):
        if index < 0:
            index += self.__len__()
        time_idx = self.indices[index]
        # self.time_idx = time_idx
        row = self.df.iloc[time_idx]
        self.check_data(time_idx)

