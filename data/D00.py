import torch
import os
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from numpy import random
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler as TorchSampler
from torch.utils.data import DataLoader as TorchLoader


def load_image_file(base, name):
    filename = os.path.join(base, name)

    with open(filename, 'rb') as file:
        img = Image.open(file)
        tsr = torch.tensor(np.array(img)[None] / 255, dtype=torch.float32)

    return tsr


def load_image_weight(base, name):
    filename = os.path.join(base, name)
    tsr = torch.tensor(np.load(filename), dtype=torch.float32)

    return tsr


class DataSampler(TorchSampler):
    def __init__(self, data_source, num_samples):
        self.max_n = len(data_source)
        self.n_samples = num_samples

        assert self.n_samples <= self.max_n

    def __iter__(self):
        yield from np.random.randint(0, self.max_n, self.n_samples)

    def __len__(self):
        return self.n_samples


class Dataset(TorchDataset):
    def __init__(self, data):
        self.data = data
        assert len(self.data) > 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        base = self.data[index]

        ctx_manifest = pd.read_csv(os.path.join(base, 'ctx-data.csv'))
        qry_manifest = pd.read_csv(os.path.join(base, 'qry-data.csv'))
        ctx_header = ['pos_x', 'pos_y', 'eul_z', 'file_dep', 'file_jlos', 'file_weight_jlos']
        qry_header = ['pos_x', 'pos_y', 'eul_z', 'file_los', 'file_weight_los']

        N = len(ctx_manifest)

        ctx_x = []
        ctx_v = []
        qry_jlos = []
        weight_jlos = []

        for pos_x, pos_y, eul_z, file_dep, file_los, file_wlos in ctx_manifest[ctx_header].values:
            ctx_v.append(torch.tensor([pos_x, pos_y, np.cos(eul_z), np.sin(eul_z)],
                                      dtype=torch.float32))
            ctx_x.append(load_image_file(base, file_dep))
            qry_jlos.append(load_image_file(base, file_los))
            weight_jlos.append(load_image_weight(base, file_wlos))

        qry_v = []
        qry_dlos = []
        weight_dlos = []

        for pos_x, pos_y, eul_z, file_los, file_wlos in qry_manifest[qry_header].values:
            qry_v.append(torch.tensor([pos_x, pos_y, np.cos(eul_z), np.sin(eul_z)],
                                      dtype=torch.float32))
            qry_dlos.append(load_image_file(base, file_los))
            weight_dlos.append(load_image_weight(base, file_wlos))

        ctx_x = torch.stack(ctx_x, dim=0)
        ctx_v = torch.stack(ctx_v, dim=0).view(N, 4, 1, 1)
        qry_jlos = torch.stack(qry_jlos, dim=0)
        qry_dlos = torch.stack(qry_dlos, dim=0)
        wgt_jlos = torch.stack(weight_jlos, dim=0)
        wgt_dlos = torch.stack(weight_dlos, dim=0)
        qry_v = torch.stack(qry_v, dim=0).view(N, 4, 1, 1)

        return ctx_x, ctx_v, qry_jlos, qry_dlos, qry_v, wgt_jlos, wgt_dlos


class DataLoader(TorchLoader):

    def __init__(self, dataset, subset_size, batch_size, drop_last, **kwargs):
        sampler = DataSampler(dataset, subset_size)
        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=drop_last)

        super().__init__(dataset, batch_sampler=batch_sampler, **kwargs)


def Data(xprm):

    class _Data(object):

        def __init__(self):
            self.data = self.get_data()
            self.dataset = {
                'train': Dataset(self.data['train']),
                'test': Dataset(self.data['test'])}

        @xprm.capture
        def get_data(self, paths):
            data = {
                'train': sorted(glob(os.path.join(paths['data']['train'], '**'))),
                'test': sorted(glob(os.path.join(paths['data']['test'], '**')))
            }

            return data

        @xprm.capture
        def get_loader(self, phase, shuffle, drop_last, 
                       batch_size, subset_size, num_workers):

            if phase == 'train':
                loader = TorchLoader(
                    self.dataset['train'], shuffle=shuffle, 
                    drop_last=drop_last, batch_size=batch_size,
                    num_workers=num_workers)

            elif phase == 'test':
                loader = DataLoader(
                    self.dataset['test'], shuffle=False, 
                    subset_size=subset_size, batch_size=batch_size, 
                    drop_last=drop_last, num_workers=num_workers)

            return loader

        @xprm.capture
        def unpack(self, package, batch_split):
            ctx_x, ctx_v, qry_jlos, qry_dlos, qry_v, wgt_jlos, wgt_dlos = package
            k = random.randint(1, 6)
            ctx_x = ctx_x[:, :k].chunk(batch_split, dim=0)
            ctx_v = ctx_v[:, :k].chunk(batch_split, dim=0)
            qry_jlos = qry_jlos[:, k - 1].chunk(batch_split, dim=0)
            qry_dlos = qry_dlos[:, k - 1].chunk(batch_split, dim=0)
            qry_v = qry_v[:, k - 1].chunk(batch_split, dim=0)
            wgt_jlos = wgt_jlos[:, k - 1:k].chunk(batch_split, dim=0)
            wgt_dlos = wgt_dlos[:, k - 1:k].chunk(batch_split, dim=0)

            return zip(ctx_x, ctx_v, qry_jlos, qry_dlos, qry_v, wgt_jlos, wgt_dlos)

    return _Data
