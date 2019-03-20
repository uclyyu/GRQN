from .. import criteria
import unittest, torch
import torch.nn as nn


class TestPerceptualLoss(unittest.TestCase):
	def setUp(self):
		self.ploss = criteria.PerceptualLoss()

	def test_forward(self):
		pred = torch.randn(10, 3, 256, 256)
		targ = torch.randn(10, 3, 256, 256)
		loss = self.ploss(pred, targ)
		self.assertTrue(loss.item() > 0.)


if __name__ == '__main__':
	unittest.main()