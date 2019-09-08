from .. import model
import unittest
import torch
import torch.nn as nn


class RepresentationEncoder(unittest.TestCase):
    def setUp(self):
        self.net = model.RepresentationEncoder()

    def test_network(self):
        num_batch = 7
        num_channel = 1
        height = 320
        width = 240
        vector_dim = 4
        x = torch.randn(num_batch, num_channel, height, width)
        v = torch.randn(num_batch, vector_dim, 1, 1)

        self.assertEqual(self.net(x, v, index=1).size(),
                         torch.Size([num_batch, 256, 1, 1]))

        self.assertEqual(self.net(x, index=2).size(),
                         torch.Size([num_batch, 256, 1, 1]))

    def test_optmiser(self):
        optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)


class TestGaussianFactor(unittest.TestCase):
    def setUp(self):
        self.net = model.GaussianFactor()

    def test_optmiser(self):
        optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)

    def test_input(self):
        x = torch.randn(3, 256, 16, 16)
        z, mean, logv = self.net(x)
        self.assertEqual(mean.size(), torch.Size([3, 256, 16, 16]))
        self.assertEqual(logv.size(), torch.Size([3, 256, 16, 16]))
        self.assertEqual(z.size(), torch.Size([3, 256, 16, 16]))


class TestRecurrentCell(unittest.TestCase):
    def setUp(self):
        repr_size = 256
        hidden_size = 128
        vector_size = 4
        self.net_jinf = model.RecurrentCell(
            repr_size + hidden_size, None, hidden_size)
        self.net_dgen = model.RecurrentCell(
            repr_size + hidden_size + vector_size, repr_size + hidden_size, hidden_size)

    def test_input(self):
        x = torch.randn(1, 384, 16, 16)
        y = torch.randn(1, 388, 16, 16)
        hid, cel, pog = self.net_jinf(x)
        self.assertEqual(hid.size(), torch.Size([1, 128, 16, 16]))
        self.assertEqual(cel.size(), torch.Size([1, 128, 16, 16]))
        self.assertEqual(pog.size(), torch.Size([1, 128, 16, 16]))
        hid, cel, pog = self.net_dgen(y, bypass=False)
        self.assertEqual(hid.size(), torch.Size([1, 128, 16, 16]))
        self.assertEqual(cel.size(), torch.Size([1, 128, 16, 16]))
        self.assertEqual(pog.size(), torch.Size([1, 128, 16, 16]))
        hid, cel, pog = self.net_dgen(x, bypass=True)
        self.assertEqual(hid.size(), torch.Size([1, 128, 16, 16]))
        self.assertEqual(cel.size(), torch.Size([1, 128, 16, 16]))
        self.assertEqual(pog.size(), torch.Size([1, 128, 16, 16]))


class TestGeneratorDelta(unittest.TestCase):
    def setUp(self):
        self.net = model.GeneratorDelta()

    def test_optmiser(self):
        optimiser = torch.optim.Adam(self.net.parameters(), 1e-3)

    def test_input(self):
        u = torch.randn(3, 128, 16, 16)
        h = torch.randn(3, 128, 16, 16)
        self.assertEqual(self.net(u, h).size(), torch.Size([3, 128, 16, 16]))


class TestDecoders(unittest.TestCase):
    def setUp(self):
        self.net = model.Decoder()

    def test_input(self):
        x = torch.randn(1, 128, 16, 16)
        b = self.net(x)

        self.assertEqual(b.size(), torch.Size([1, 1, 320, 240]))


if __name__ == '__main__':
    unittest.main()
