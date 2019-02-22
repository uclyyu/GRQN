from .. import gern
import unittest, torch
import torch.nn as nn


class TestRepresentationEncoderPrimitive(unittest.TestCase):
	def setUp(self):
		self.net = gern.RepresentationEncoderPrimitive()

	def test_network(self):
		x = torch.randn(1, 3, 256, 256)
		k = torch.randn(1, 1, 256, 256)
		m = torch.randn(1, 3, 256, 256)
		q = torch.randn(1, 7,   1,   1)

		self.assertEqual(self.net(x, k, m, q).size(), torch.Size([1, 256, 1, 1]))


class TestRepresentationEncoderState(unittest.TestCase):
	def setUp(self):
		self.net = gern.RepresentationEncoderState(input_size=256, hidden_size=128, zoneout=.15)
		self.net_init = gern.RepresentationEncoderState(input_size=256, hidden_size=128, zoneout=.15, init=True)

	def test_input(self):
		x = torch.randn(1, 7, 256)
		try:
			h, c, o = self.net(x)
		except:
			self.fail('RepresentationEncoderState failed!')
		self.assertEqual(h.size(), torch.Size([1, 7, 128]))
		self.assertEqual(c.size(), torch.Size([1, 7, 128]))
		self.assertEqual(o.size(), torch.Size([1, 7, 128]))

	def test_input_hco(self):
		x = torch.randn(1, 7, 256)
		h = torch.randn(1, 128)
		c = torch.randn(1, 128)
		o = torch.randn(1, 128)
		try:
			h, c, o = self.net(x, h, c, o)
		except:
			self.fail('RepresentationEncoderState failed!')
		self.assertEqual(h.size(), torch.Size([1, 7, 128]))
		self.assertEqual(c.size(), torch.Size([1, 7, 128]))
		self.assertEqual(o.size(), torch.Size([1, 7, 128]))

	def test_input_init(self):
		x = torch.randn(1, 7, 256)
		try:
			h, c, o = self.net_init(x)
		except:
			self.fail('RepresentationEncoderState failed!')
		self.assertEqual(h.size(), torch.Size([1, 7, 128]))
		self.assertEqual(c.size(), torch.Size([1, 7, 128]))
		self.assertEqual(o.size(), torch.Size([1, 7, 128]))

	def test_input_init_hco(self):
		x = torch.randn(1, 7, 256)
		h = torch.randn(1, 128)
		c = torch.randn(1, 128)
		o = torch.randn(1, 128)
		try:
			h, c, o = self.net_init(x, h, c, o)
		except:
			self.fail('RepresentationEncoderState failed!')
		self.assertEqual(h.size(), torch.Size([1, 7, 128]))
		self.assertEqual(c.size(), torch.Size([1, 7, 128]))
		self.assertEqual(o.size(), torch.Size([1, 7, 128]))


class TestRepresentationEncoder(unittest.TestCase):
	def setUp(self):
		self.net = gern.RepresentationEncoder(256, 128, 256)
	def test_network(self):
		prim = torch.randn(1, 256)
		state = torch.randn(1, 128)

		self.assertEqual(self.net(prim, state).size(), torch.Size([1, 256]))


class TestRepresentationAggregator(unittest.TestCase):
	def setUp(self):
		self.net = gern.RepresentationAggregator(128, 256)

	def test_network(self):
		x = torch.randn(1, 128)

		self.assertEqual(self.net(x).size(), torch.Size([1, 256]))


class TestAggregateRewind(unittest.TestCase):
	def setUp(self):
		self.net = gern.AggregateRewind(256, 128)
		self.net_init = gern.AggregateRewind(256, 128, learn_init=True)

	def test_input(self):
		N = 3
		x = torch.randn(1, 256)
		y, o = self.net(x, rewind_steps=N)
		self.assertEqual(y.size(), torch.Size([1, N + 1, 256]))
		self.assertEqual(o.size(), torch.Size([1, 128]))

	def test_input_init(self):
		N = 3
		x = torch.randn(1, 256)
		y, o = self.net_init(x, rewind_steps=N)
		self.assertEqual(y.size(), torch.Size([1, N + 1, 256]))
		self.assertEqual(o.size(), torch.Size([1, 128]))

	def test_input_hco(self):
		N = 3
		x = torch.randn(1, 256)
		h = torch.randn(1, 128)
		c = torch.randn(1, 128)
		o = torch.randn(1, 128)
		y, o_hat = self.net(x, h, c, o, rewind_steps=N)
		self.assertEqual(y.size(), torch.Size([1, N + 1, 256]))
		self.assertEqual(o.size(), torch.Size([1, 128]))

	def test_input_init_hco(self):
		N = 3
		x = torch.randn(1, 256)
		h = torch.randn(1, 128)
		c = torch.randn(1, 128)
		o = torch.randn(1, 128)
		y, o_hat = self.net_init(x, h, c, o, rewind_steps=N)
		self.assertEqual(y.size(), torch.Size([1, N + 1, 256]))
		self.assertEqual(o.size(), torch.Size([1, 128]))


if __name__ == '__main__':
	unittest.main()