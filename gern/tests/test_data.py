from .. import data
import unittest, torch
import torch.nn as nn


class TestGernDataset(unittest.TestCase):
	def setUp(self):
		self.dataset_cpu = data.GernDataset('resources/examples/dataset')

	def test_indexing_cpu(self):
		print(' --- ', self.__class__.__name__)
		self.dataset_cpu.renew_dataset_state()
		kx, kk, kv, qx, qk, qv, label = self.dataset_cpu[0]
		print('kX: ', kx.size())
		print('kK: ', kk.size())
		print('kV: ', kv.size())
		print('qX: ', qx.size())
		print('qK: ', qk.size())
		print('qV: ', qv.size())
		print('Label: ', label)


class TestGernDataLoader(unittest.TestCase):
	def setUp(self):
		self.batch_size = 2
		self.subset_size = 2
		self.loader_cpu = data.GernDataLoader('resources/examples/dataset', subset_size=self.subset_size, drop_last=True, batch_size=self.batch_size)

	def test_draw_cpu(self):
		print(' --- ', self.__class__.__name__)
		self.loader_cpu.dataset.renew_dataset_state()
		for i, (kx, kk, kv, qx, qk, qv, label) in enumerate(self.loader_cpu):
			print(i, ' kX: ', kx.size())
			print(i, ' kK: ', kk.size())
			print(i, ' kV: ', kv.size())
			print(i, ' qX: ', qx.size())
			print(i, ' qK: ', qk.size())
			print(i, ' qV: ', qv.size())
			print(i, ' Label: ', label)



if __name__ == '__main__':
	unittest.main()