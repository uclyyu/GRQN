import torch
import torchvision
import os
import json
import cv2
import numpy as np
import pandas as pd
from functools import reduce


class GernDataset(torch.utils.data.Dataset):
    def __init__(self, rootdir):
        self.rootdir = os.path.abspath(rootdir)
        assert os.path.exists(self.rootdir)
        self.data = os.listdir(self.rootdir)
        assert len(self.data) > 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        serial = self.data[index]  # 8-digit folder name

        ctx_vector = pd.read_csv(os.path.join(
            self.rootdir, serial, 'ctx-data.csv'))
        qry_vector = pd.read_csv(os.path.join(
            self.rootdir, serial, 'qry-data.csv'))

        N = len(ctx_vector)

        ctx_x = []
        ctx_v = []
        qry_jlos = []
        for pos_x, pos_y, eul_z, file_dep, file_los in ctx_vector[['pos_x', 'pos_y', 'eul_z', 'file_dep', 'file_jlos']].values:
            ctx_v.append(torch.tensor([pos_x, pos_y, np.cos(
                eul_z), np.sin(eul_z)], dtype=torch.float32))

            file_dep = os.path.join(self.rootdir, serial, file_dep)
            dep = cv2.imread(file_dep, cv2.IMREAD_GRAYSCALE)[None] / 255.
            ctx_x.append(torch.tensor(dep, dtype=torch.float32))

            file_los = os.path.join(self.rootdir, serial, file_los)
            los = cv2.imread(file_los, cv2.IMREAD_GRAYSCALE)[None] / 255.
            qry_jlos.append(torch.tensor(los, dtype=torch.float32))

        qry_v = []
        qry_dlos = []
        for pos_x, pos_y, eul_z, file_los in qry_vector[['pos_x', 'pos_y', 'eul_z', 'file_los']].values:
            qry_v.append(torch.tensor([pos_x, pos_y, np.cos(
                eul_z), np.sin(eul_z)], dtype=torch.float32))

            file_los = os.path.join(self.rootdir, serial, file_los)
            los = cv2.imread(file_los, cv2.IMREAD_GRAYSCALE)[None] / 255.
            qry_dlos.append(torch.tensor(los, dtype=torch.float32))

        ctx_x = torch.stack(ctx_x, dim=0)
        ctx_v = torch.stack(ctx_v, dim=0).view(N, 4, 1, 1)
        qry_jlos = torch.stack(qry_jlos, dim=0)
        qry_dlos = torch.stack(qry_dlos, dim=0)
        qry_v = torch.stack(qry_v, dim=0).view(N, 4, 1, 1)

        return ctx_x, ctx_v, qry_jlos, qry_dlos, qry_v


class GernSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, num_samples):
        self.max_n = len(data_source)
        self.n_samples = num_samples

        assert self.n_samples <= self.max_n

    def __iter__(self):
        yield from np.random.randint(0, self.max_n, self.n_samples)

    def __len__(self):
        return self.n_samples


class GernDataLoader(torch.utils.data.DataLoader):
    def __init__(self, rootdir, subset_size=64, batch_size=8, drop_last=True, **kwargs):
        dataset = GernDataset(rootdir)
        sampler = GernSampler(dataset, subset_size)
        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=drop_last)

        super(GernDataLoader, self).__init__(
            dataset, batch_sampler=batch_sampler, **kwargs)
