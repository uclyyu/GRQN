from .. import model
import unittest, torch
import torch.nn as nn


class TestRepresentationEncoderState(unittest.TestCase):
	def setUp(self):
		self.batch_size = 7
		self.sequence_length = 5
		self.input_size = 256
		self.hidden_size = 128
		self.zoneout_prob = 0.3
		self.net = model.RepresentationEncoderState(self.input_size, self.hidden_size, self.zoneout_prob)
		# self.net_init = model.RepresentationEncoderState(input_size=256, hidden_size=128, zoneout=.15, learn_init=True)

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)

	def test_forward(self):
		for dev in ['cpu', 'cuda']:
			self.net.to(dev)

			inp = torch.randn(self.batch_size, self.sequence_length, self.input_size, device=dev)
			hid = torch.randn(self.batch_size, self.hidden_size, device=dev)
			cel = torch.randn(hid.size(), device=dev)
			pog = torch.randn(hid.size(), device=dev)

			# with tensors
			self.net(inp, hid, cel, pog)

			# with nones
			self.net(inp, None, None, None)


class TestAggregateRewind(unittest.TestCase):
	def setUp(self):
		self.batch_size = 7
		self.input_size = 256
		self.sequence_length = 5
		self.hidden_size = 512
		self.zoneout_prob = 0.3
		self.net = model.AggregateRewind(self.input_size, self.hidden_size, self.zoneout_prob)

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)

	def test_forward(self):
		for dev in ['cpu', 'cuda']:
			self.net.to(dev)

			inp = torch.randn(self.batch_size, self.input_size, device=dev)
			hid = torch.randn(self.batch_size, self.hidden_size, device=dev)
			cel = torch.randn(hid.size(), device=dev)
			pog = torch.randn(hid.size(), device=dev)

			# with tensors
			self.net(inp, hid, cel, pog, self.sequence_length)

			# with Nones
			self.net(inp, None, None, None, self.sequence_length)


if __name__ == '__main__':
	unittest.main()