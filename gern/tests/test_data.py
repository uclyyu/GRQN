from .. import data
import unittest, torch
import torch.nn as nn


class TestGernDataset(unittest.TestCase):
	def setUp(self):
		self.dataset_cpu = data.GernDataset('resources/examples/dataset')

	def test_indexing_cpu(self):
		cnd, qry, lab = self.dataset_cpu[0]
		Xc, Mc, Kc, Vc = cnd
		Xq, Mq, Kq, Vq = qry

		self.assertEqual(Xc[0].size(), torch.Size([3, 256, 256]))

		for cnd, qry, lab in self.dataset_cpu:
			pass


class TestGernDataLoader(unittest.TestCase):
	def setUp(self):
		self.batch_size = 3
		self.subset_size = 4
		self.loader_cpu = data.GernDataLoader('resources/examples/dataset', subset_size=self.subset_size, drop_last=True, batch_size=self.batch_size)
		self.num_batch = self.subset_size // self.batch_size

	def test_draw_cpu(self):
		for i, (C, Q, L) in enumerate(self.loader_cpu):
			pass
		self.assertEqual(i + 1, self.num_batch)
		self.assertEqual(C[0].size(0), self.batch_size)


if __name__ == '__main__':
	unittest.main()