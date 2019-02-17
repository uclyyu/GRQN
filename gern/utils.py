import torch
import torch.nn as nn



class ConvLSTMCell(nn.Module):
	def __init__(self, input_size, hidden_size, kernel_size, zoneout=0, bias=False):
		super(ConvLSTMCell, self).__init__()

		# !TODO:
		#	Zoneout, https://arxiv.org/pdf/1606.01305.pdf
		# 	Recurrent Dropout, https://arxiv.org/pdf/1603.05118.pdf
		# 	Layer Normalisation
		# 	Recurrent Batch Normalisation, https://arxiv.org/pdf/1603.09025.pdf

		kH, kW = kernel_size

		self.kernel_size = kernel_size
		self.padding = kH // 2
		self._weight_ih = nn.Parameter(torch.empty(hidden_size * 4, hidden_size, kH, kW))
		self._weight_hh = nn.Parameter(torch.empty(hidden_size * 4, input_size, kH, kW))
		self.bias = None
		self.zoneout = 0

		if bias:
			self.bias = nn.Parameter(torch.zeros(hidden_size * 4))

		if zoneout:
			self.zoneout = torch.distributions.Bernoulli(probs=zoneout)

		# --- weight initialisation
		# Xavier uniform for input-to-hidden,
		nn.init.xavier_uniform_(self._weight_ih)
		# Orthogonal for hidden-to-hidden,
		nn.init.ortogonal_(self._weight_hh)
		# From "Learning to Forget: Continual Prediction with LSTM"
		if bias:
			self._bias[:hidden_size].fill_(1.)

	def _convlstm(gf, gi, gs, go, hid, cel, zo, go_prev):
		gf = torch.sigmoid(gf)
		gi = torch.sigmoid(gi)
		gs = torch.tanh(gs)
		go = torch.sigmoid(go)

		if isinstance(zo, torch.distributions.Bernoulli):
			# Use shared zoneout mask
			z = zo.sample()
			c_next = gf * cel + z * gi * gs
			h_next = ((1 - z) * go + z * go_prev) * torch.tanh(c_next)
		else:
			c_next = gf * cel + gi * gs
			h_next = go * torch.tanh(c_next)

		return h_next, c_next, go

	def forward(self, x, h, c, o):
		inp = torch.cat([x, h], dim=1)
		weight = torch.cat([self._weight_ih, self._weight_hh], dim=1)

		out = nn.functional.conv2d(inp, weight, self.bias, stride=1, padding=self.padding)
		gf, gi, gs, go = torch.chunk(out, 4, dim=1)
		h, c, o = self._convlstm(gf, gi, gs, go, h, c, self.zoneout, o)






