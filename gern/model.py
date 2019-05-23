import torch, random, typing
import torch.nn as nn
from numpy.random import randint
from torch.distributions import Normal
from collections import namedtuple
from .utils import ConvLSTMCell, LSTMCell, GroupNorm2d, GroupNorm1d, SkipConnect, BilinearInterpolate, count_parameters


def init_parameters(module):
	if type(module) in (nn.Linear, nn.Conv2d):
		nn.init.xavier_uniform_(module.weight.data)
		if module.bias is not None:
			module.bias.data.fill_(0.)


class RepresentationEncoderPrimitive(nn.Module):
	def __init__(self):
		super(RepresentationEncoderPrimitive, self).__init__()

		self.features = self.features = nn.ModuleList([
			# 0 ---
			nn.Sequential(
				nn.Conv2d(7, 32, (3, 3), (2, 2), padding=1), 
				GroupNorm2d(32, 8), 
				nn.ReLU(True),
				SkipConnect(
					nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1)), 
				GroupNorm2d(32, 8), 
				nn.ReLU(True),
				nn.MaxPool2d((2, 2), stride=(2, 2)),
				nn.Conv2d(32, 64, (3, 3), (1, 1), padding=1), 
				GroupNorm2d(64, 8), 
				nn.ReLU(True),
				SkipConnect(
					nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1)), 
				GroupNorm2d(64, 8), 
				nn.ReLU(True),
				nn.MaxPool2d((2, 2), stride=(2, 2)),
				nn.Conv2d( 64, 128, (3, 3), (1, 1), padding=1), 
				GroupNorm2d(128, 8), 
				nn.ReLU(True),
				SkipConnect(
					nn.Sequential(
						nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1), 
						GroupNorm2d(128, 8), 
						nn.ReLU(True),
						nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1), 
						)), 
				GroupNorm2d(128, 8), 
				nn.ReLU(True),
				nn.MaxPool2d((2, 2), stride=(2, 2))
			),
			# 1 ---
			nn.Sequential(
				nn.Conv2d(135, 128, (3, 3), (2, 2), padding=1), 
				GroupNorm2d(128, 8),
				nn.ReLU(True),
				nn.MaxPool2d((2, 2), stride=(2, 2)),
				nn.Conv2d(128, 256, (3, 3), (1, 1), padding=1), 
				GroupNorm2d(256, 8),
				nn.ReLU(True),
				SkipConnect(
					nn.Conv2d(256, 256, (1, 1), (1, 1), bias=False),
					nn.Sequential(
						nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1),
						GroupNorm2d(256, 8),
						nn.ReLU(True),
						nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1))
					), 
				GroupNorm2d(256, 8),
				nn.ReLU(True),
				nn.MaxPool2d((2, 2), stride=(2, 2)),
				nn.Conv2d(256, 256, (3, 3), (2, 2), padding=1)
			)
		])

		self.features.apply(init_parameters)
		# print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))

	def forward(self, x, m, k, v):
		# --- Inputs (size)
		# x: RGB input sequence (B, T, 3, 256, 256)
		# m: Openpose 'background' heatmap (B, T, 1, 256, 256)
		# k: Openpose rendered keypoints (B, T, 3, 256, 256)
		# v: Camera orientation (B, T, 7, 1, 1)
		# --- Output (size)
		# Representation primitive (B, T, 256, 1, 1)
		B, T, _, H, W = x.size()
		dev = x.device

		x = x.view(B * T, -1, H, W)
		m = m.view(B * T, -1, H, W)
		k = k.view(B * T, -1, H, W)
		v = v.view(B * T, -1, 1, 1)

		inp = torch.cat([x, m, k], dim=1)
		inp = self.features[0](inp)
		inp = torch.cat([inp, v.expand(-1, -1, inp.size(2), inp.size(3))], dim=1)
		out = self.features[1](inp)

		return out.view(B, T, -1, out.size(2), out.size(3))


class RepresentationEncoderState(torch.jit.ScriptModule):
	__constants__ = ['hidden_size']
	def __init__(self, input_size=256, hidden_size=128, zoneout=.15):
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


class RepresentationEncoder(nn.Module):
	def __init__(self, primitive_size=256, state_size=128, hidden_size=256):
		super(RepresentationEncoder, self).__init__()

		input_size = primitive_size + state_size

		self.op_hidden = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			GroupNorm1d(hidden_size, 8),
			nn.ReLU(True),
			SkipConnect(nn.Sequential(
				nn.Linear(hidden_size, hidden_size),
				GroupNorm1d(hidden_size, 8),
				nn.ReLU(),
				nn.Linear(hidden_size, hidden_size, bias=False))),
			GroupNorm1d(hidden_size, 8),
			nn.ReLU(True),
			SkipConnect(nn.Sequential(
				nn.Linear(hidden_size, hidden_size),
				GroupNorm1d(hidden_size, 8),
				nn.ReLU(),
				nn.Linear(hidden_size, hidden_size, bias=False))),
			GroupNorm1d(hidden_size, 8),
			nn.ReLU(True),
			)
		self.op_key = nn.Linear(hidden_size, primitive_size)
		self.op_query = nn.Linear(hidden_size, primitive_size)

		self.apply(init_parameters)

		# print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))

	def forward(self, prim, state):
		inp = torch.cat([prim, state], dim=1)

		hid = self.op_hidden(inp)
		key = self.op_key(hid)
		qry = self.op_query(hid)
		att = nn.functional.softmax(key * qry, dim=1)

		return att * prim


class RepresentationAggregator(nn.Module):
	def __init__(self, input_size=128, output_size=256):
		super(RepresentationAggregator, self).__init__()

		self.features = nn.Sequential(
			nn.Linear(input_size, output_size),
			GroupNorm1d(output_size, 8),
			nn.ReLU(True),
			SkipConnect(nn.Sequential(
				nn.Linear(output_size, output_size),
				GroupNorm1d(output_size, 8),
				nn.ReLU(),
				nn.Linear(output_size, output_size, bias=False))),
			GroupNorm1d(output_size, 8),
			nn.ReLU(True),
			SkipConnect(nn.Sequential(
				nn.Linear(output_size, output_size),
				GroupNorm1d(output_size, 8),
				nn.ReLU(),
				nn.Linear(output_size, output_size, bias=False))),
			GroupNorm1d(output_size, 8),
			nn.ReLU(True),
			nn.Linear(output_size, output_size)
			)

		self.apply(init_parameters)

		# print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))

	def forward(self, x):
		return self.features(x)


class AggregateRewind(torch.jit.ScriptModule):

	__constants__ = ['hidden_size']

	def __init__(self, input_size=256, hidden_size=128, zoneout=.15):
		super(AggregateRewind, self).__init__()

		self.hidden_size = hidden_size
		self.op_progrm = LSTMCell(input_size, hidden_size, zoneout=zoneout)
		self.op_rewind = nn.Sequential(
			nn.Linear(input_size + hidden_size, hidden_size),
			GroupNorm1d(hidden_size, 8),
			nn.ReLU(True),
			SkipConnect(nn.Sequential(
				nn.Linear(hidden_size, hidden_size),
				GroupNorm1d(hidden_size, 8),
				nn.ReLU(True),
				nn.Linear(hidden_size, hidden_size, bias=False))),
			GroupNorm1d(hidden_size, 8),
			nn.ReLU(True),
			SkipConnect(nn.Sequential(
				nn.Linear(hidden_size, hidden_size),
				GroupNorm1d(hidden_size, 8),
				nn.ReLU(True),
				nn.Linear(hidden_size, hidden_size, bias=False))),
			GroupNorm1d(hidden_size, 8),
			nn.ReLU(True),
			nn.Linear(hidden_size, input_size)
			)

		self.op_rewind.apply(init_parameters)

		# print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))

	@torch.jit.script_method
	def forward(self, x, hid=None, cel=None, pog=None, steps=0):
		# type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], int) -> Tuple[Tensor, Tensor]
		"""Summary
		
		Args:
		    x (Tensor, 2D): aggregated scene representation
		    hid (Tensor, 2D, optional): lstm hidden neurons at t=T
		    pog (Tensor, 2D, optional): lstm hidden neurons at t=T-1
		    steps (int, optional): number of step to look back in time
		
		Returns:
		    Tuple[Tensor, Tensor]: Description
		"""
		

		# x: input aggregate
		# hid: lstm hidden state
		# cel: lstm cell state
		# pog: previous lstm output gate

		hidden_shape = torch.Size([x.size(0), self.hidden_size])
		if hid is None:
			hid = torch.zeros(1, device=x.device).expand(hidden_shape)

		if cel is None:
			cel = torch.zeros(1, device=x.device).expand(hidden_shape)

		if pog is None:
			pog = torch.zeros(1, device=x.device).expand(hidden_shape)

		rewind = [x]
		for _ in range(steps):
			x_ = rewind[-1]
			hid, cel, pog = self.op_progrm(x_, hid, cel, pog)

			r_input = torch.cat([x_, hid], dim=1)
			r = self.op_rewind(r_input) + x_
			rewind.append(r)

		return torch.stack(rewind, dim=1), pog


class GaussianFactor(nn.Module):
	def __init__(self):
		super(GaussianFactor, self).__init__()

		self.layer = nn.Sequential(
			nn.Conv2d(256, 512, (3, 3), (1, 1), padding=1),
			nn.ReLU(),
			SkipConnect(
				nn.Conv2d(512, 256, (3, 3), (1, 1), padding=1),
				nn.Conv2d(512, 256, (1, 1), (1, 1), bias=False)),
			nn.ReLU(),
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
			GroupNorm2d(256, 8),
			nn.ReLU(True),
			SkipConnect(nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1)),
			GroupNorm2d(256, 8),
			nn.ReLU(True),
			nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1)
			)

		self.apply(init_parameters)

	def forward(self, u, h):
		inp = torch.cat([u, h], dim=1)
		return self.layers(inp)


class DecoderBase(nn.Module):
	def __init__(self):
		super(DecoderBase, self).__init__()

		self.decoder_base = nn.Sequential(
			nn.ConvTranspose2d(256, 256, (3, 3), (2, 2), padding=0),
			GroupNorm2d(256, 8),
			nn.ReLU(True),
			SkipConnect(
				nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1),
				nn.Conv2d(256, 256, (1, 1), (1, 1), bias=False)),
			GroupNorm2d(256, 8),
			nn.ReLU(True),
			nn.ConvTranspose2d(256, 128, (3, 3), (2, 2), padding=0),
			GroupNorm2d(128, 8),
			nn.ReLU(True),
			SkipConnect(
				nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1),
				nn.Conv2d(128, 128, (1, 1), (1, 1), bias=False)),
			GroupNorm2d(128, 8),
			nn.ReLU(True),
			nn.ConvTranspose2d(128, 128, (3, 3), (2, 2), padding=0),
			GroupNorm2d(128, 8),
			nn.ReLU(True),
			SkipConnect(
				nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1),
				nn.Conv2d(128, 128, (1, 1), (1, 1), bias=False)),
			GroupNorm2d(128, 8),
			nn.ReLU(True)
			)

		self.apply(init_parameters)

	def forward(self, x):
		y = self.decoder_base(x)
		return y[:, :, 2:-3, 2:-3]


class DecoderHeatmap(nn.Module):
	def __init__(self):
		super(DecoderHeatmap, self).__init__()

		self.decoder_hm = nn.Sequential(
			nn.ConvTranspose2d(128, 64, (3, 3), (1, 1), padding=1),
			GroupNorm2d(64, 8),
			nn.ReLU(True),
			SkipConnect(
				nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1),
				nn.Conv2d(64, 64, (1, 1), (1, 1), bias=False)),
			GroupNorm2d(64, 8),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 16, (3, 3), (2, 2), padding=1),
			GroupNorm2d(16, 8),
			nn.ReLU(True),
			SkipConnect(
				nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1),
				nn.Conv2d(16, 16, (1, 1), (1, 1), bias=False)),
			GroupNorm2d(16, 4),
			nn.ReLU(True),
			nn.Conv2d(16, 1, (3, 3), (1, 1), padding=1)
			)

		self.apply(init_parameters)

	def forward(self, x):
		return self.decoder_hm(x)[:, :, 2:-1, 2:-1]


class DecoderRGBVision(nn.Module):
	def __init__(self):
		super(DecoderRGBVision, self).__init__()

		self.decoder_pre = nn.Sequential(
			nn.ConvTranspose2d(128, 64, (3, 3), (1, 1), padding=1),
			GroupNorm2d(64, 8),
			nn.ReLU(True),
			SkipConnect(
				nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1),
				nn.Conv2d(64, 64, (1, 1), (1, 1), bias=False)),
			GroupNorm2d(64, 8),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 16, (3, 3), (2, 2), padding=1),
			GroupNorm2d(16, 4),
			nn.ReLU(True),
			SkipConnect(
				nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1),
				nn.Conv2d(16, 16, (1, 1), (1, 1), bias=False)),
			GroupNorm2d(16, 4),
			nn.ReLU(True),
			nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1)
			)

		self.decoder_post = nn.Sequential(
			nn.Conv2d(17, 16, (3, 3), (1, 1), padding=1), 
			GroupNorm2d(16, 4), 
			nn.ReLU(True),
			SkipConnect(
				nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1),
				nn.Conv2d(16, 16, (1, 1), (1, 1), bias=False)),
			GroupNorm2d(16, 4), 
			nn.ReLU(True),
			nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1),
			GroupNorm2d(16, 4), 
			nn.ReLU(True),
			SkipConnect(
				nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1),
				nn.Conv2d(16, 16, (1, 1), (1, 1), bias=False)),
			GroupNorm2d(16, 4), 
			nn.ReLU(True),
			nn.Conv2d(16, 3, (3, 3), (1, 1), padding=1)
			)

		self.apply(init_parameters)

	def forward(self, b, h):
		pre = self.decoder_pre(b)[:, :, 2:-1, 2:-1]
		inp = torch.cat([pre, h], dim=1)
		post = self.decoder_post(inp)
		return post


class AuxiliaryClassifier(nn.Module):
	def __init__(self, nclass=13):
		super(AuxiliaryClassifier, self).__init__()

		self.features = nn.Sequential(
			SkipConnect(
				nn.Sequential(
					nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1),
					GroupNorm2d(256, 8),
					nn.ReLU(True),
					nn.Conv2d(256, 256, (3, 3), (2, 2), padding=1)),
				nn.Conv2d(256, 256, (1, 1), (2, 2), bias=False)),
			GroupNorm2d(256, 8),
			nn.ReLU(True),
			nn.MaxPool2d((2, 2)),
			nn.Conv2d(256, 128, (3, 3), (1, 1), padding=1),
			SkipConnect(
				nn.Sequential(
					nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1),
					GroupNorm2d(128, 8),
					nn.ReLU(True),
					nn.Conv2d(128, 128, (3, 3), (2, 2), padding=1)),
				nn.Conv2d(128, 128, (1, 1), (2, 2), bias=False)),
			GroupNorm2d(128, 8),
			nn.ReLU(True),
			nn.MaxPool2d((2, 2)),
			)
		self.classifier = nn.Sequential(
			SkipConnect(nn.Linear(128, 128)),
			GroupNorm1d(128, 8),
			nn.ReLU(True),
			nn.Linear(128, nclass))

		self.apply(init_parameters)

	def forward(self, x):
		B = x.size(0)
		h = self.features(x)
		return self.classifier(h.view(B, -1))


GernOutput = namedtuple('GernOutput', 
	['rgbv', 'heat', 'label', 'gamma',
	 'prior_mean', 'prior_logv', 'posterior_mean', 'posterior_logv',
	 'cnd_repr', 'cnd_aggr'])

GernTarget = namedtuple('GernTarget',
	['rgbv', 'heat', 'label'])
