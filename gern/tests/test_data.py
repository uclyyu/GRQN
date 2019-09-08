from .. import data
import unittest
import torch
import torch.nn as nn


class TestGernDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = data.GernDataset('/home/yen/data/gern/phase/test')

    def test_indexing(self):
        ctx_x, ctx_v, qry_jlos, qry_dlos, qry_v = self.dataset[-1]

        self.assertEqual(ctx_x.size(), torch.Size([5, 1, 240, 320]))
        self.assertEqual(ctx_v.size(), torch.Size([5, 4, 1, 1]))
        self.assertEqual(qry_jlos.size(), torch.Size([5, 1, 256, 256]))
        self.assertEqual(qry_dlos.size(), torch.Size([5, 1, 256, 256]))
        self.assertEqual(qry_v.size(), torch.Size([5, 4, 1, 1]))


class TestGernDataLoader(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.subset_size = 32
        self.loader_cpu = data.GernDataLoader(
            '/home/yen/data/gern/phase/test',
            subset_size=self.subset_size,
            drop_last=True, batch_size=self.batch_size)
        self.num_batch = self.subset_size // self.batch_size

    def test_draw_cpu(self):
        for i, (ctx_x, ctx_v, qry_jlos, qry_dlos, qry_v) in enumerate(self.loader_cpu):
            pass
        self.assertEqual(i + 1, self.num_batch)
        self.assertEqual(ctx_x.size(), torch.Size(
            [self.batch_size, 5, 1, 240, 320]))


if __name__ == '__main__':
    unittest.main()
