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


class TestRepresentationEncoderPrimitive(unittest.TestCase):
	def setUp(self):
		self.net = model.RepresentationEncoderPrimitive()

	def test_network(self):
		B = 1
		T = 3
		x = torch.randn(B, T, 3, 256, 256)
		m = torch.randn(B, T, 1, 256, 256)
		k = torch.randn(B, T, 3, 256, 256)
		v = torch.randn(B, T, 7,   1,   1)

		self.assertEqual(self.net(x, m, k, v).size(), torch.Size([B, T, 256, 1, 1]))

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)


class TestRepresentationEncoder(unittest.TestCase):
	def setUp(self):
		self.net = model.RepresentationEncoder(256, 128, 256)

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)

	def test_network(self):
		prim = torch.randn(1, 256)
		state = torch.randn(1, 128)

		self.assertEqual(self.net(prim, state).size(), torch.Size([1, 256]))


class TestRepresentationAggregator(unittest.TestCase):
	def setUp(self):
		self.net = model.RepresentationAggregator(128, 256)

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)

	def test_network(self):
		x = torch.randn(1, 128)

		self.assertEqual(self.net(x).size(), torch.Size([1, 256]))


class TestRecurrentCell(unittest.TestCase):
	def setUp(self):
		self.net = model.RecurrentCell(256, 128)
		self.net_init = model.RecurrentCell(256, 128, feature_size=(16, 16), kernel_size=(5, 5), learn_init=True)

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)
		optimiser = torch.optim.Adam(self.net_init.parameters(), 1e-3)

	def test_input(self):
		x = torch.randn(2, 256, 16, 16)
		hid, cel, pog = self.net(x)
		self.assertEqual(hid.size(), torch.Size([2, 128, 16, 16]))
		self.assertEqual(cel.size(), torch.Size([2, 128, 16, 16]))
		self.assertEqual(pog.size(), torch.Size([2, 128, 16, 16]))
		hid, cel, pog = self.net(x, hid, cel, pog)
		self.assertEqual(hid.size(), torch.Size([2, 128, 16, 16]))
		self.assertEqual(cel.size(), torch.Size([2, 128, 16, 16]))
		self.assertEqual(pog.size(), torch.Size([2, 128, 16, 16]))

	def test_input_init(self):
		x = torch.randn(2, 256, 16, 16)
		hid, cel, pog = self.net(x)
		self.assertEqual(hid.size(), torch.Size([2, 128, 16, 16]))
		self.assertEqual(cel.size(), torch.Size([2, 128, 16, 16]))
		self.assertEqual(pog.size(), torch.Size([2, 128, 16, 16]))
		hid, cel, pog = self.net(x, hid, cel, pog)
		self.assertEqual(hid.size(), torch.Size([2, 128, 16, 16]))
		self.assertEqual(cel.size(), torch.Size([2, 128, 16, 16]))
		self.assertEqual(pog.size(), torch.Size([2, 128, 16, 16]))


class TestGaussianFactor(unittest.TestCase):
	def setUp(self):
		self.net = model.GaussianFactor()

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)

	def test_input(self):
		x = torch.randn(3, 256, 16, 16)
		dist, mean, logv = self.net(x)
		self.assertEqual(mean.size(), torch.Size([3, 256, 16, 16]))
		self.assertEqual(logv.size(), torch.Size([3, 256, 16, 16]))
		self.assertEqual(dist.rsample().size(), torch.Size([3, 256, 16, 16]))


class TestGeneratorDelta(unittest.TestCase):
	def setUp(self):
		self.net = model.GeneratorDelta()

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)

	def test_input(self):
		u = torch.randn(3, 256, 16, 16)
		h = torch.randn(3, 256, 16, 16)
		self.assertEqual(self.net(u, h).size(), torch.Size([3, 256, 16, 16]))


class TestAuxiliaryClassifier(unittest.TestCase):
	def setUp(self):
		self.net = model.AuxiliaryClassifier()

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)

	def test_input(self):
		x = torch.randn(3, 256, 16, 16)
		self.assertEqual(self.net(x).size(), torch.Size([3, 13]))


class TestDecoders(unittest.TestCase):
	def setUp(self):
		self.netb = model.DecoderBase()
		self.neth = model.DecoderHeatmap()
		self.netv = model.DecoderRGBVision()

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.netb.parameters(), 1e-3)
		optimiser = torch.optim.Adam(self.neth.parameters(), 1e-3)
		optimiser = torch.optim.Adam(self.netv.parameters(), 1e-3)

	def test_input(self):
		x = torch.randn(3, 256, 16, 16)
		b = self.netb(x)
		h = self.neth(b)
		v = self.netv(b, h)

		self.assertEqual(b.size(), torch.Size([3, 128, 130, 130]))
		self.assertEqual(h.size(), torch.Size([3,   1, 256, 256]))
		self.assertEqual(v.size(), torch.Size([3,   3, 256, 256]))

if __name__ == '__main__':
	unittest.main()