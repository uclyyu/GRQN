import torch, random, typing
import torch.nn as nn
from numpy.random import randint
from torch.distributions import Normal
from collections import namedtuple
from .utils import ConvLSTMCell, LSTMCell, GroupNorm2d, GroupNorm1d, SkipConnect, BilinearInterpolate, count_parameters
from torch.nn import functional as F


def init_parameters(module):
	if type(module) in (nn.Linear, nn.Conv2d):
		nn.init.xavier_uniform_(module.weight.data)
		if module.bias is not None:
			module.bias.data.fill_(0.)

def l1_regularizer(r=[]):
	def _l1_regularizer(module):
		if type(module) in (nn.Linear, nn.Conv2d, nn.ConvTranspose2d):
			reg = F.l1_loss(module.weight, torch.zeros_like(module.weight))
			num = module.weight.numel()
			r.append((reg, num))
	return _l1_regularizer


class RepresentationEncoderPrimitive(nn.Module):
	"""The `Pool' Representation Network as documented in
	`Neural scene representation and rendering'."""
	def __init__(self, input_channels=3, output_channels=256, p=0.2):
		super(RepresentationEncoderPrimitive, self).__init__()
		cinp = input_channels
		cout = output_channels
		self.output_channels = cout
		self.features = self.features = nn.ModuleList([
			# Modifications:
			# Dropout at bottleneck
			# LeakyReLU instead of ReLU
			nn.Sequential(
				nn.Conv2d(cinp, 256, (2, 2), (2, 2), padding=0), 
				nn.ReLU(True),
				GroupNorm2d(256, 32), 
				SkipConnect(
					nn.Conv2d(256, 128, (3, 3), (1, 1), padding=1),
					nn.Conv2d(256, 128, (1, 1), (1, 1), padding=0)), 
				nn.LeakyReLU(0.1),
				GroupNorm2d(128, 32), 
				nn.Dropout(p=p),
				nn.Conv2d(128, 256, (2, 2), (2, 2), padding=0),
				nn.LeakyReLU(0.1),
				GroupNorm2d(256, 32)),
			nn.Sequential(
				SkipConnect(
					nn.Conv2d(263, 128, (3, 3), (1, 1), padding=1),
					nn.Conv2d(263, 128, (1, 1), (1, 1), padding=0)),
				nn.LeakyReLU(0.1),
				GroupNorm2d(128, 32),
				nn.Dropout(p=p),
				nn.Conv2d(128, 256, (3, 3), (1, 1), padding=1),
				nn.LeakyReLU(0.1),
				GroupNorm2d(256, 32),
				nn.Conv2d(256, 256, (1, 1), (1, 1), padding=0),
				nn.AvgPool2d((16, 16))
				)
		])

		self.features.apply(init_parameters)
		# print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))

	def forward(self, x, v):
		B, T, C, H, W = x.size()
		x = x.view(B * T, C, H, W)
		v = v.view(B * T, 7, 1, 1)

		h = self.features[0](x)
		h = torch.cat([h, v.expand(-1, -1, h.size(2), h.size(3))], dim=1)
		out = self.features[1](h)

		return out.view(B, T, self.output_channels)

	def regularizer(self):
		r = []
		self.apply(l1_regularizer(r))
		reg, num = zip(*r)
		return sum(reg), sum(num)


class RepresentationEncoderState(torch.jit.ScriptModule):
	__constants__ = ['hidden_size']
	def __init__(self, input_size=256, hidden_size=128, zoneout=0.4):
		super(RepresentationEncoderState, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.network = LSTMCell(input_size, hidden_size, zoneout=zoneout, bias=True)

		# print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))
		
	@torch.jit.script_method
	def forward(self, x, hid=None, cel=None, pog=None):
		# type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]
		"""Summary
		
		Args:
		    x (Tensor, 3D): scene repressentation primitives
		    hid (Tensor, 2D, optional): LSTM hidden neurons at t=T
		    cel (Tensor, 2D, optional): LSTM cell neurons at t=T
		    pog (Tensor, 2D, optional): LSTM hidden neurons at t=T-1
		
		Returns:
		    Tuple[Tensor, Tensor, Tensor]: Description
		"""

		# batch_size = x.size(0)
		# num_steps = x.size(1)
		# dev = x.device
		
		default_shape = torch.Size([x.size(0), self.hidden_size])
		if pog is None:
			pog = torch.zeros(1, dtype=torch.float32, device=x.device).expand(default_shape)
				
		if hid is None:
			hid = torch.zeros(1, dtype=torch.float32, device=x.device).expand(default_shape)
				
		if cel is None:
			cel = torch.zeros(1, dtype=torch.float32, device=x.device).expand(default_shape)
				
		hids = []
		for x_ in torch.unbind(x, dim=1):
			hid, cel, pog = self.network(x_, hid, cel, pog)
			hids.append(hid)

		hids = torch.stack(hids, dim=1)
		
		return hids, cel, pog


class RepresentationEncoder(torch.jit.ScriptModule):
	__constants__ = ['zoneout_prob', 'hidden_size']
	def __init__(self, primitive_size=256, state_size=128, hidden_size=256, zoneout=0.4):
		super(RepresentationEncoder, self).__init__()

		input_size = primitive_size + state_size
		self.hidden_size = hidden_size
		self.zoneout_prob = zoneout
		self.layer = nn.Linear(input_size + hidden_size, hidden_size * 3)

		self.apply(init_parameters)

		# print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))

	@torch.jit.script_method
	def forward(self, prim, state, hid=None, cel=None, pog=None):
		# type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]
		inp = torch.cat([prim, state], dim=2)
		T = inp.size(1)
		device = prim.device 
		default_shape = torch.Size([inp.size(0), self.hidden_size])

		if pog is None:
			pog = torch.zeros(1, dtype=torch.float32, device=device).expand(default_shape)
		if hid is None:
			hid = torch.zeros(1, dtype=torch.float32, device=device).expand(default_shape)
		if cel is None:
			cel = torch.zeros(1, dtype=torch.float32, device=device).expand(default_shape)

		for t in range(T):
			prim_t = prim[:, t]
			rnn_inp = torch.cat([inp[:, t], hid], dim=1)
			rnn_out = self.layer(rnn_inp)
			gf, gi, go = torch.chunk(rnn_out, 3, dim=1)

			gf = torch.sigmoid(gf)
			gi = torch.sigmoid(gi)
			go = torch.sigmoid(go)

			if self.training:
				mask = torch.zeros_like(go).bernoulli(self.zoneout_prob)
				cel = gf * cel + prim_t * gi
				hid = ((1 - mask) * go + mask * pog) * torch.tanh(cel)
				pog = go
			else:
				cel = gf * cel + prim_t * gi
				hid = go * torch.tanh(cel)

		return hid, cel, pog


class GaussianFactor(nn.Module):
	def __init__(self):
		super(GaussianFactor, self).__init__()

		self.layer = nn.Sequential(
			nn.Conv2d(256, 512, (3, 3), (1, 1), padding=1),
			nn.LeakyReLU(0.1),
			SkipConnect(
				nn.Conv2d(512, 256, (3, 3), (1, 1), padding=1),
				nn.Conv2d(512, 256, (1, 1), (1, 1), bias=False)),
			nn.LeakyReLU(0.1),
			nn.Conv2d(256, 512, (1, 1), (1, 1))  # mean, log-variance
		)
		
		self.apply(init_parameters)

	def forward(self, inp):
		mean, logv = torch.chunk(self.layer(inp), 2, dim=1)
		scale = (0.5 * logv).exp()
		z = torch.randn(inp.size(), device=inp.device)
		sample = z * scale + mean
		# dist = torch.distributions.Normal(mean, scale)
		return sample, mean, logv


class RecurrentCell(nn.Module):
	def __init__(self, input_channels, hidden_channels, kernel_size=(3, 3), feature_size=None, zoneout=0.15, learn_init=False):
		super(RecurrentCell, self).__init__()
		self.input_channels = input_channels
		self.hidden_channels = hidden_channels

		self.features = ConvLSTMCell(input_channels, hidden_channels, kernel_size, zoneout=zoneout)
		self.init_hid = None
		self.init_cel = None
		if learn_init:
			if feature_size is None:
				raise 
			self.init_hid = nn.Parameter(torch.zeros(1, hidden_channels, feature_size[0], feature_size[1]))
			self.init_cel = nn.Parameter(torch.zeros(1, hidden_channels, feature_size[0], feature_size[1]))

	def forward(self, x, hid=None, cel=None, pog=None):
		B, C, H, W = x.size()
		Ch = self.hidden_channels
		device = x.device

		if hid is None:
			if self.init_hid is None:
				hid = torch.zeros(1, device=device).expand(B, Ch, H, W)
			else:
				hid = self.init_hid.expand(B, -1, -1, -1)
		if cel is None:
			if self.init_cel is None:
				cel = torch.zeros(1, device=device).expand(B, Ch, H, W)
			else:
				cel = self.init_cel.expand(B, -1, -1, -1)
		if pog is None:
			pog = torch.zeros(1, device=device).expand(B, Ch, H, W)

		hid, cel, pog = self.features(x, hid, cel, pog)

		return hid, cel, pog


class GeneratorDelta(nn.Module):
	def __init__(self):
		super(GeneratorDelta, self).__init__()

		self.layers = nn.Sequential(
			nn.Conv2d(512, 256, (3, 3), (1, 1), padding=1),
			nn.LeakyReLU(0.1),
			GroupNorm2d(256, 8),
			SkipConnect(nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1)),
			nn.LeakyReLU(0.1),
			GroupNorm2d(256, 8),
			nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1)
			)

		self.apply(init_parameters)

	def forward(self, u, h):
		inp = torch.cat([u, h], dim=1)
		return self.layers(inp)


class DecoderRGBV(nn.Module):
	def __init__(self):
		super(DecoderRGBV, self).__init__()

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(256, 256, (2, 2), (2, 2), bias=False, padding=0),
			nn.LeakyReLU(0.1),
			GroupNorm2d(256, 32),
			SkipConnect(
				nn.Conv2d(256, 128, (1, 1), (1, 1), bias=False, padding=0),
				nn.Conv2d(256, 128, (3, 3), (1, 1), bias=False, padding=1)),
			nn.LeakyReLU(0.1),
			GroupNorm2d(128, 32),
			SkipConnect(
				nn.Conv2d(128, 128, (1, 1), (1, 1), bias=False, padding=0),
				nn.Conv2d(128, 128, (3, 3), (1, 1), bias=False, padding=1)),
			nn.LeakyReLU(0.1),
			GroupNorm2d(128, 32),
			nn.ConvTranspose2d(128, 64, (2, 2), (2, 2), bias=False, padding=0),
			nn.LeakyReLU(0.1),
			GroupNorm2d(64, 32),
			SkipConnect(
				nn.Conv2d(64, 16, (1, 1), (1, 1), bias=False, padding=0),
				nn.Conv2d(64, 16, (3, 3), (1, 1), bias=True,  padding=1)),
			nn.LeakyReLU(0.1),
			nn.Conv2d(16, 3, (3, 3), (1, 1), padding=1)
			)

		self.apply(init_parameters)

	def forward(self, x):
		return self.decoder(x)

	def regularizer(self):
		r = []
		self.apply(l1_regularizer(r))
		reg, num = zip(*r)
		return sum(reg), sum(num)


class LatentClassifier(nn.Module):
	def __init__(self, nclass=13, p=0.2):
		super(LatentClassifier, self).__init__()

		self.features = nn.Sequential(
			nn.Conv2d(259, 128, (4, 4), (4, 4), bias=False, padding=0),
			nn.LeakyReLU(0.1),
			GroupNorm2d(128, 32),
			SkipConnect(
				nn.Conv2d(128, 128, (3, 3), (1, 1), bias=False, padding=1)),
			nn.LeakyReLU(0.1),
			nn.MaxPool2d((2, 2)),
			GroupNorm2d(128, 32),
			nn.Conv2d(128, 256, (4, 4), (4, 4), bias=False, padding=0),
			nn.LeakyReLU(0.1),
			GroupNorm2d(256, 32),
			SkipConnect(
				nn.Conv2d(256, 256, (3, 3), (1, 1), bias=False, padding=1)),
			nn.LeakyReLU(0.1),
			GroupNorm2d(256, 32),
			nn.Conv2d(256, 1024, (2, 2), (2, 2), bias=False, padding=0),
			nn.LeakyReLU(0.1),
			GroupNorm2d(1024, 32),
			nn.Conv2d(1024, nclass, (1, 1), (1, 1))
			)
		self.nclass = nclass
		self.apply(init_parameters)

	def forward(self, x, r):
		inp = torch.cat([x, r], dim=1)
		return self.features(inp).view(-1, self.nclass)

	def regularizer(self):
		r = []
		self.apply(l1_regularizer(r))
		reg, num = zip(*r)
		return sum(reg), sum(num)
