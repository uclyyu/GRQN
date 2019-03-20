from .. import scheduler
from loguru import logger
import unittest, torch
import torch.nn as nn


class TestLearningRateScheduler(unittest.TestCase):
	def setUp(self):
		self.net = nn.Linear(3, 4)
		self.optimiser = torch.optim.Adam(self.net.parameters(), 0.1)
		self.scheduler = scheduler.LearningRateScheduler(self.optimiser, 0.1, 1.0, 10)

	def test_step(self):
		print()
		for epoch in range(15):
			self.scheduler.step(epoch)
			lr = self.optimiser.param_groups[0]['lr']
			logger.info('Epoch {epoch:02d}, learning rate = {lr:.3f}', epoch=epoch, lr=lr)


class TestPixelStdDevScheduler(unittest.TestCase):
	def setUp(self):
		self.criterion_weights = [0.1, 0.2, 0.3, 0.4, 0.5]
		self.index = 4
		self.scheduler = scheduler.PixelStdDevScheduler(self.criterion_weights, self.index, 0.7, 2.0, 10)

	def test_step(self):
		print()
		for epoch in range(15):
			self.scheduler.step(epoch)
			sd = self.criterion_weights[self.index]
			logger.info('Epoch {epoch:02d}, stddev = {sd:.3f}', epoch=epoch, sd=sd)

if __name__ == '__main__':
	unittest.main()