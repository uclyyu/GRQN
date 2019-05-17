from .. import utils 
import unittest, torch
import torch.nn as nn


class TestUtils(unittest.TestCase):
	def setUp(self):
		self.batch_size = 7
		self.input_size = 32
		self.hidden_size = 128
		self.kernel_size = (7, 7)
		self.zoneout_prob = 0.4
		self.tensor2d = torch.randn(self.batch_size, self.hidden_size * 4)
		self.convlstmcell = utils.ConvLSTMCell(self.input_size, self.hidden_size, self.kernel_size, self.zoneout_prob)
		self.lstmcell = utils.LSTMCell(self.input_size, self.hidden_size, self.zoneout_prob)

	def test_lstm(self):
		for dev in ['cpu', 'cuda']:
			gf, gi, gs, go = torch.chunk(self.tensor2d.to(dev), 4, dim=1)
			hid = torch.randn(go.size(), device=dev)
			cel = torch.randn(go.size(), device=dev)
			mask = torch.zeros_like(go).bernoulli_(0.4)
			go_prev = torch.randn(go.size(), device=dev)
			# with zoneout
			h_next, c_next, go_next = utils._lstm_zoneout(gf, gi, gs, go, hid, cel, mask, go_prev)
			# without zoneout
			h_next, c_next = utils._lstm(gf, gi, gs, go, hid, cel)


	def test_ConvLSTMCell(self):
		for dev in ['cpu', 'cuda']:
			self.convlstmcell.to(dev)
			inp = torch.randn(self.batch_size, self.input_size,  self.kernel_size[0] * 2, self.kernel_size[0] * 2, device=dev)
			hid = torch.randn(self.batch_size, self.hidden_size, self.kernel_size[0] * 2, self.kernel_size[0] * 2, device=dev)
			cel = torch.randn(self.batch_size, self.hidden_size, self.kernel_size[0] * 2, self.kernel_size[0] * 2, device=dev)
			pog = torch.randn(self.batch_size, self.hidden_size, self.kernel_size[0] * 2, self.kernel_size[0] * 2, device=dev)

			self.convlstmcell.train()
			h1, c1, o1 = self.convlstmcell(inp, hid, cel, pog)

			self.convlstmcell.eval()
			h2, c2, o2 = self.convlstmcell(inp, hid, cel, pog)

	def test_LSTMCell(self):
		for dev in ['cpu', 'cuda']:
			self.lstmcell.to(dev)
			inp = torch.randn(self.batch_size, self.input_size, device=dev)
			hid = torch.randn(self.batch_size, self.hidden_size, device=dev)
			cel = torch.randn(self.batch_size, self.hidden_size, device=dev)
			pog = torch.randn(self.batch_size, self.hidden_size, device=dev)




if __name__ == '__main__':
	unittest.main()
