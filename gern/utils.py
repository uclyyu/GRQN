import torch
import torch.nn as nn


def count_parameters(cls, trainable_only=True):

	if trainable_only:
		filt = filter(lambda p: p.requires_grad, cls.parameters())
	else:
		filt = cls.parameters()

	count = sum(map(lambda p: p.numel(), filt))

	return count

# Helper class for skip connection
class SkipConnect(nn.Module):
	def __init__(self, main, skip=None):
		super(SkipConnect, self).__init__()
		self.main = main
		self.skip = skip
		
	def forward(self, inp):
		if self.skip is None:
			return self.main(inp) + inp
		else:
			return self.main(inp) + self.skip(inp)
		

class BilinearInterpolate(nn.Module):
    def __init__(self, scale):
        super(BilinearInterpolate, self).__init__()
        
        self.scale = scale
        
    def forward(self, inp):
        return nn.functional.interpolate(inp, scale_factor=self.scale, mode='bilinear', align_corners=True)


class GroupNorm1d(nn.Module):
	def __init__(self, features, groups, eps=1e-5):
		super(GroupNorm1d, self).__init__()

		self.gamma = nn.Parameter(torch.ones(1, features))
		self.beta = nn.Parameter(torch.zeros(1, features))
		self.num_groups = groups
		self.eps = eps

	def forward(self, x):
		N, C = x.size()
		G = self.num_groups

		x = x.view(N, G, -1)
		mean = x.mean(dim=2, keepdim=True)
		var = (x - mean).pow(2).sum(2, keepdim=True) / x.size(2)

		x = (x - mean) / (var + self.eps).sqrt()
		x = x.view(N, C)

		return x * self.gamma + self.beta

class GroupNorm2d(nn.Module):
	def __init__(self, channels, groups, eps=1e-5):
		super(GroupNorm2d, self).__init__()

		self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
		self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
		self.num_groups = groups
		self.eps = eps

	def forward(self, x):
		N, C, H, W = x.size()
		G = self.num_groups

		x = x.view(N, G, -1)
		mean = x.mean(dim=2, keepdim=True)
		var = (x - mean).pow(2).sum(2, keepdim=True) / x.size(2)

		x = (x - mean) / (var + self.eps).sqrt()
		x = x.view(N, C, H, W)

		return x * self.gamma + self.beta


def _lstm_zoneout(gf, gi, gs, go, hid, cel, zo, go_prev):
	gf = torch.sigmoid(gf)
	gi = torch.sigmoid(gi)
	gs = torch.tanh(gs)
	go = torch.sigmoid(go)

	if isinstance(zo, torch.distributions.Bernoulli):
		# Use shared zoneout mask
		z = zo.sample(go.size())
		c_next = gf * cel + z * gi * gs
		h_next = ((1 - z) * go + z * go_prev) * torch.tanh(c_next)
	else:
		c_next = gf * cel + gi * gs
		h_next = go * torch.tanh(c_next)

	return h_next, c_next, go


class ConvLSTMCell(nn.Module):
	def __init__(self, input_size, hidden_size, kernel_size, zoneout=0, bias=True):
		super(ConvLSTMCell, self).__init__()

		# !TODO:
		#	Zoneout, https://arxiv.org/pdf/1606.01305.pdf
		# 	Recurrent Dropout, https://arxiv.org/pdf/1603.05118.pdf
		# 	Layer Normalisation
		# 	Recurrent Batch Normalisation, https://arxiv.org/pdf/1603.09025.pdf

		kH, kW = kernel_size

		self.kernel_size = kernel_size
		self.padding = kH // 2
		self._weight_ih = nn.Parameter(torch.empty(hidden_size * 4, input_size, kH, kW))
		self._weight_hh = nn.Parameter(torch.empty(hidden_size * 4, hidden_size, kH, kW))
		self.bias = None
		self.zoneout = 0

		if bias:
			self.bias = nn.Parameter(torch.zeros(hidden_size * 4))

		if zoneout > 0:
			self.zoneout = torch.distributions.Bernoulli(probs=zoneout)

		# --- weight initialisation
		# Xavier uniform for input-to-hidden,
		nn.init.xavier_uniform_(self._weight_ih.data)
		# Orthogonal for hidden-to-hidden,
		nn.init.orthogonal_(self._weight_hh.data)
		# From "Learning to Forget: Continual Prediction with LSTM"
		if bias:
			self.bias[:hidden_size].data.fill_(1.)


	def forward(self, x, h, c, o):
		inp = torch.cat([x, h], dim=1)
		weight = torch.cat([self._weight_ih, self._weight_hh], dim=1)

		out = nn.functional.conv2d(inp, weight, self.bias, stride=1, padding=self.padding)
		gf, gi, gs, go = torch.chunk(out, 4, dim=1)
		h, c, o = _lstm_zoneout(gf, gi, gs, go, h, c, self.zoneout, o)

		return h, c, o


class LSTMCell(nn.Module):
	def __init__(self, input_size, hidden_size, zoneout=0, bias=True):
		super(LSTMCell, self).__init__()

		self._weight_ih = nn.Parameter(torch.empty(hidden_size * 4, input_size))
		self._weight_hh = nn.Parameter(torch.empty(hidden_size * 4, hidden_size))
		self.bias = None
		self.zoneout = 0

		if bias:
			self.bias = nn.Parameter(torch.zeros(hidden_size * 4))

		if zoneout > 0:
			self.zoneout = torch.distributions.Bernoulli(probs=zoneout)

		nn.init.xavier_uniform_(self._weight_ih.data)
		nn.init.orthogonal_(self._weight_hh.data)

		if bias:
			self.bias[:hidden_size].data.fill_(1.)

	def forward(self, x, h, c, o):
		inp = torch.cat([x, h], dim=1)
		weight = torch.cat([self._weight_ih, self._weight_hh], dim=1)

		out = nn.functional.linear(inp, weight, self.bias)
		gf, gi, gs, go = torch.chunk(out, 4, dim=1)
		if self.training:
			h, c, o = _lstm_zoneout(gf, gi, gs, go, h, c, self.zoneout, o)
		else:
			h, c, o = _lstm_zoneout(gf, gi, gs, go, h, c, None, o)

		return h, c, o






