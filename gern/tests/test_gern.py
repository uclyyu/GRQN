from .. import gern
import unittest, torch
import torch.nn as nn


class TestGern(unittest.TestCase):
	def setUp(self):
		self.net = gern.GeRN()

	def test_forward(self):
		cnd_x = torch.randn(2, 5, 3, 256, 256)
		cnd_m = torch.randn(2, 5, 1, 256, 256)
		cnd_k = torch.randn(2, 5, 3, 256, 256)
		cnd_v = torch.randn(2, 5, 7,   1,   1)
		qry_x = torch.randn(2, 3, 3, 256, 256)
		qry_m = torch.randn(2, 3, 1, 256, 256)
		qry_k = torch.randn(2, 3, 3, 256, 256)
		qry_v = torch.randn(2, 3, 7,   1,   1) 

		out = self.net(
			cnd_x, cnd_m, cnd_k, cnd_v, 
			qry_x, qry_m, qry_k, qry_v)

	def test_optimiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-4)


	# def test_compute_packed_representation(self):
	# 	B = 1
	# 	T = 3
	# 	x = torch.randn(B, T, 3, 256, 256)
	# 	m = torch.randn(B, T, 1, 256, 256)
	# 	k = torch.randn(B, T, 3, 256, 256)
	# 	v = torch.randn(B, T, 7,   1,   1)

	# 	reps, aggr = self.net._compute_packed_representation(x, m, k, v)
	# 	self.assertEqual(reps.size(), torch.Size([B, T, 256]))
	# 	self.assertEqual(aggr.size(), torch.Size([B, T, 256]))


if __name__ == '__main__':
	unittest.main()