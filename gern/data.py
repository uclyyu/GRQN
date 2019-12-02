import torch
import os
import numpy as np
import pandas as pd
from PIL import Image


def load_image_file(root, serial, name):
    filename = os.path.join(root, serial, name)
    with open(filename, 'rb') as file:
        img = Image.open(file)
        tsr = torch.tensor(np.array(img)[None] / 255, dtype=torch.float32)
    return tsr


def load_image_weight(root, serial, name):
    filename = os.path.join(root, serial, name)
    tsr = torch.tensor(np.load(filename), dtype=torch.float32)
    return tsr


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
        weight_jlos = []

        for pos_x, pos_y, eul_z, file_dep, file_los, file_wlos in ctx_vector[['pos_x', 'pos_y', 'eul_z', 'file_dep', 'file_jlos', 'file_weight_jlos']].values:
            ctx_v.append(torch.tensor([pos_x, pos_y, np.cos(
                eul_z), np.sin(eul_z)], dtype=torch.float32))
            ctx_x.append(load_image_file(self.rootdir, serial, file_dep))
            qry_jlos.append(load_image_file(self.rootdir, serial, file_los))
            weight_jlos.append(load_image_weight(
                self.rootdir, serial, file_wlos))

        qry_v = []
        qry_dlos = []
        weight_dlos = []

        for pos_x, pos_y, eul_z, file_los, file_wlos in qry_vector[['pos_x', 'pos_y', 'eul_z', 'file_los', 'file_weight_los']].values:
            qry_v.append(torch.tensor([pos_x, pos_y, np.cos(
                eul_z), np.sin(eul_z)], dtype=torch.float32))

            qry_dlos.append(load_image_file(self.rootdir, serial, file_los))
            weight_dlos.append(load_image_weight(
                self.rootdir, serial, file_wlos))

        ctx_x = torch.stack(ctx_x, dim=0)
        ctx_v = torch.stack(ctx_v, dim=0).view(N, 4, 1, 1)
        qry_jlos = torch.stack(qry_jlos, dim=0)
        qry_dlos = torch.stack(qry_dlos, dim=0)
        wgt_jlos = torch.stack(weight_jlos, dim=0)
        wgt_dlos = torch.stack(weight_dlos, dim=0)
        qry_v = torch.stack(qry_v, dim=0).view(N, 4, 1, 1)

        return ctx_x, ctx_v, qry_jlos, qry_dlos, qry_v, wgt_jlos, wgt_dlos


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
