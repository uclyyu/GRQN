from .. import criteria
from ..model import GernOutput, GernTarget
from ..data import GernDataLoader
from ..gern import GeRN
from loguru import logger
import unittest, torch
import torch.nn as nn


class TestGernCriteria(unittest.TestCase):
	def setUp(self):
		self.net = GeRN()
		self.criteria = criteria.GernCriterion()
		self.loader = GernDataLoader('resources/examples/dataset', subset_size=5, batch_size=5)

	def test_criterion(self):
		(C, Q, L), = self.loader
		weights = [1., 1., 1., 1., 0.1]

		logger.info('Input batch size = {size}', size=C[0].size(0))
		logger.info('Input sequence length = {length}', length=C[0].size(1))
		logger.info('Rewind sequence length = {length}', length=Q[0].size(1))

		gern_output = self.net(*C, *Q)
		gern_target = self.net.make_target(*Q, L)

		sum_loss = self.criteria(gern_output, gern_target, weights)
		logger.info('Total weighted loss = {loss:.4f}', loss=sum_loss.item())

		loss = self.criteria.item()
		logger.info('Perceptual L2 Loss = {loss:.4f}', loss=loss[0])
		logger.info('Heatmap BCE Loss = {loss:.4f}', loss=loss[1])
		logger.info('Classifier CE Loss = {loss:.4f}', loss=loss[2])
		logger.info('Aggregator L2 Loss = {loss:.4f}', loss=loss[3])
		logger.info('Autoencoder KL Loss = {loss:.4f}', loss=loss[4])


if __name__ == '__main__':
	unittest.main()