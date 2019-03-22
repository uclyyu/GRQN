from .. import gern
import unittest, torch
import torch.nn as nn


class TestRepresentationEncoderPrimitive(unittest.TestCase):
	def setUp(self):
		self.net = gern.RepresentationEncoderPrimitive()

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


class TestRepresentationEncoderState(unittest.TestCase):
	def setUp(self):
		self.net = gern.RepresentationEncoderState(input_size=256, hidden_size=128, zoneout=.15)
		self.net_init = gern.RepresentationEncoderState(input_size=256, hidden_size=128, zoneout=.15, learn_init=True)

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)
		optimiser = torch.optim.Adam(self.net_init.parameters(), 1e-3)

	def test_input(self):
		x = torch.randn(1, 7, 256)
		try:
			h, c, o = self.net(x)
		except:
			self.fail('RepresentationEncoderState failed!')
		self.assertEqual(h.size(), torch.Size([1, 7, 128]))
		self.assertEqual(c.size(), torch.Size([1, 128]))
		self.assertEqual(o.size(), torch.Size([1, 128]))

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
		self.assertEqual(c.size(), torch.Size([1, 128]))
		self.assertEqual(o.size(), torch.Size([1, 128]))

	def test_input_init(self):
		x = torch.randn(1, 7, 256)
		try:
			h, c, o = self.net_init(x)
		except:
			self.fail('RepresentationEncoderState failed!')
		self.assertEqual(h.size(), torch.Size([1, 7, 128]))
		self.assertEqual(c.size(), torch.Size([1, 128]))
		self.assertEqual(o.size(), torch.Size([1, 128]))

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
		self.assertEqual(c.size(), torch.Size([1, 128]))
		self.assertEqual(o.size(), torch.Size([1, 128]))


class TestRepresentationEncoder(unittest.TestCase):
	def setUp(self):
		self.net = gern.RepresentationEncoder(256, 128, 256)

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)

	def test_network(self):
		prim = torch.randn(1, 256)
		state = torch.randn(1, 128)

		self.assertEqual(self.net(prim, state).size(), torch.Size([1, 256]))


class TestRepresentationAggregator(unittest.TestCase):
	def setUp(self):
		self.net = gern.RepresentationAggregator(128, 256)

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)

	def test_network(self):
		x = torch.randn(1, 128)

		self.assertEqual(self.net(x).size(), torch.Size([1, 256]))


class TestAggregateRewind(unittest.TestCase):
	def setUp(self):
		self.net = gern.AggregateRewind(256, 128)
		self.net_init = gern.AggregateRewind(256, 128, learn_init=True)

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)
		optimiser = torch.optim.Adam(self.net_init.parameters(), 1e-3)

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

class TestRecurrentCell(unittest.TestCase):
	def setUp(self):
		self.net = gern.RecurrentCell(256, 128)
		self.net_init = gern.RecurrentCell(256, 128, feature_size=(16, 16), kernel_size=(5, 5), learn_init=True)

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
		self.net = gern.GaussianFactor()

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
		self.net = gern.GeneratorDelta()

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)

	def test_input(self):
		u = torch.randn(3, 256, 16, 16)
		h = torch.randn(3, 256, 16, 16)
		self.assertEqual(self.net(u, h).size(), torch.Size([3, 256, 16, 16]))


class TestAuxiliaryClassifier(unittest.TestCase):
	def setUp(self):
		self.net = gern.AuxiliaryClassifier()

	def test_optmiser(self):
		optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)

	def test_input(self):
		x = torch.randn(3, 256, 16, 16)
		self.assertEqual(self.net(x).size(), torch.Size([3, 13]))


class TestDecoders(unittest.TestCase):
	def setUp(self):
		self.netb = gern.DecoderBase()
		self.neth = gern.DecoderHeatmap()
		self.netv = gern.DecoderRGBVision()

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