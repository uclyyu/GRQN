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
			qry_x, qry_m, qry_k, qry_v,
			gamma=.99, asteps=32, rsteps=4)

	def test_predict(self):
		cnd_x = torch.randn(2, 5, 3, 256, 256)
		cnd_m = torch.randn(2, 5, 1, 256, 256)
		cnd_k = torch.randn(2, 5, 3, 256, 256)
		cnd_v = torch.randn(2, 5, 7,   1,   1)
		qry_v = torch.randn(2, 3, 7,   1,   1) 

		out = self.net.predict(
			cnd_x, cnd_m, cnd_k, cnd_v, qry_v,
			gamma=.99, asteps=32, rsteps=4)

	def test_optimiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-4)


if __name__ == '__main__':
	unittest.main()