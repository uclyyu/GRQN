import torch, imageio, os, random
import torch.nn as nn
from numpy.random import randint
from torch.distributions import Normal
from tqdm import tqdm
from collections import namedtuple
from .utils import ConvLSTMCell, LSTMCell, GroupNorm2d, GroupNorm1d, SkipConnect, BilinearInterpolate, count_parameters


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

		print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))

	def forward(self, x, m, k, v):
		# x: RGB input sequence (BxTx3xHxW)
		# m: Openpose 'background' heatmap (BxTx1xHxW)
		# k: Openpose rendered keypoints (BxTx3xHxW)
		# v: Camera orientation (BxTx7x1x1)
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


class RepresentationEncoderState(nn.Module):
	def __init__(self, input_size=256, hidden_size=128, zoneout=.15, init=False):
		super(RepresentationEncoderState, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.network = LSTMCell(input_size, hidden_size, zoneout=zoneout, bias=True)

		self._init_hid = None
		self._init_cel = None
		self._init_pog = None

		if init:
			self._init_hid = nn.Parameter(torch.zeros(1, hidden_size))
			self._init_cel = nn.Parameter(torch.zeros(1, hidden_size))
			self._init_pog = nn.Parameter(torch.zeros(1, hidden_size))

		print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))
		
	def forward(self, x, hid=None, cel=None, pog=None):
		# x: Representation encoder primitive
		# hid: LSTM hidden state
		# cel: LSTM cell state
		# pog: LSTM output gate from previous time step
		batch_size = x.size(0)
		num_steps = x.size(1)
		dev = x.device
		
		default_size = torch.Size([batch_size, self.hidden_size])
		if pog is None and self._init_pog is None:
			pog = torch.zeros(1, dtype=torch.float32, device=dev).expand(default_size)
		elif pog is None:
			pog = self._init_pog.expand(batch_size, -1)
				
		if hid is None and self._init_hid is None:
			hid = torch.zeros(1, dtype=torch.float32, device=dev).expand(default_size)
		elif hid is None:
			hid = self._init_hid.expand(batch_size, -1)
				
		if cel is None and self._init_cel is None:
			cel = torch.zeros(1, dtype=torch.float32, device=dev).expand(default_size)
		elif cel is None:
			cel = self._init_cel.expand(batch_size, -1)
				
		
		hids, cels, pogs = [], [], []
		for x_ in torch.unbind(x, dim=1):
			hid, cel, pog = self.network(x_, hid, cel, pog)
			hids.append(hid)
			cels.append(cel)
			pogs.append(pog)

		hids = torch.stack(hids, dim=1)
		cels = torch.stack(cels, dim=1)
		pogs = torch.stack(pogs, dim=1)
		
		return hids, cels, pogs


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

		print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))

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

		print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))

	def forward(self, x):
		return self.features(x)


class AggregateRewind(nn.Module):
	def __init__(self, input_size=256, hidden_size=128, zoneout=.15, learn_init=False):
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

		self.init_hid = None
		self.init_cel = None
		if learn_init:
			self.init_hid = nn.Parameter(torch.zeros(1, hidden_size))
			self.init_cel = nn.Parameter(torch.zeros(1, hidden_size))

		print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))

	def forward(self, x, hid=None, cel=None, pog=None, rewind_steps=0):
		# x: input aggregate
		# hid: lstm hidden state
		# cel: lstm cell state
		# pog: previous lstm output gate
		B = x.size(0)
		C = self.hidden_size

		if hid is None:
			if self.init_hid is None:
				hid = torch.zeros(1, device=x.device).expand(B, C)
			else:
				hid = self.init_hid.expand(B, -1)
				
		if cel is None:
			if self.init_cel is None:
				cel = torch.zeros(1, device=x.device).expand(B, C)
			else:
				cel = self.init_cel.expand(B, -1)
				
		if pog is None:
			pog = torch.zeros(1, device=x.device).expand(B, C)

		rewind = [x]
		for _ in range(rewind_steps):
			x_ = rewind[-1]
			hid, cel, pog = self.op_progrm(x_, hid, cel, pog)

			r_input = torch.cat([x_, hid], dim=1)
			r = self.op_rewind(r_input) + x_
			rewind.append(r)

		return torch.stack(rewind, dim=1), pog


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

	def forward(self, b, h):
		pre = self.decoder_pre(b)[:, :, 2:-1, 2:-1]
		inp = torch.cat([pre, h], dim=1)
		post = self.decoder_post(inp)
		return post


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
		
	def forward(self, inp):
		mean, logv = torch.chunk(self.layer(inp), 2, dim=1)
		scale = (0.5 * logv).exp()
		dist = torch.distributions.Normal(mean, scale)
		return dist, mean, logv


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

	def forward(self, u, h):
		inp = torch.cat([u, h], dim=1)
		return self.layers(inp)


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
			nn.Linear(128, nclass),
			nn.Softmax(dim=1))

	def forward(self, x):
		B = x.size(0)
		h = self.features(x)
		return self.classifier(h.view(B, -1))


GernOutput = namedtuple('GernOutput', 
	['rgbv', 'heat', 'label', 
	 'prior_mean', 'prior_logv', 'posterior_mean', 'posterior_logv',
	 'cnd_repr', 'cnd_aggr'])


class GeRN(nn.Module):
	def __init__(self):
		super(GeRN, self).__init__()

		# default sizes/dimensionality
		Nr = 256  # aggregated representation
		Nh = 256  # recurrent cell hidden
		Nv =   7  # query vector

		# representaiton operators
		self.rop_primitive = RepresentationEncoderPrimitive()
		self.rop_state = RepresentationEncoderState()
		self.rop_representation = RepresentationEncoder()
		self.rop_aggregator = RepresentationAggregator()
		self.rop_rewind = AggregateRewind()

		# inference operators
		self.iop_posterior = GaussianFactor()
		self.iop_state = RecurrentCell(Nr * 2 + Nh, Nh)

		# generation operators
		self.gop_prior = GaussianFactor()
		self.gop_state = RecurrentCell(Nr + Nh + Nv, Nh)
		self.gop_delta = GeneratorDelta()

		# classifier
		self.aux_class = AuxiliaryClassifier()

		# decoding operators
		self.dop_base = DecoderBase()
		self.dop_heat = DecoderHeatmap()
		self.dop_rgbv = DecoderRGBVision()

		# self.rnn_representation = RecurrentRepresentationAggregator(512, 512, 256)  # inp, hid, out channels
		# self.rnn_inference = RecurrentRepresentationAggregator(512, 128, 128)
		# self.rnn_encoder = RecurrentCell(1152, 256)
		# self.rnn_generator = RecurrentCell(775, 256)
		# self.rnn_decoder = RecurrentCell(512, 256)
		# self.image_decoder = Decoder()
		# self.prior_factor = GaussianFactor()
		# self.posterior_factor = GaussianFactor()
		# self.generator_delta = nn.Sequential(
		# 	nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1),
		# 	nn.ReLU(True),
		# 	nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1))

		# self.net_status = dict()
		# self.reset_aggregator()
		# self.reset_rnns()


		print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))

	def _on_rep(self, x, m ,k, v):
		prim = self.rop_primitive(x, m, k, v).squeeze(4).squeeze(2)
		return prim

	# def _compute_packed_representation(self, x, m, k, v):
	# 	B, T, _, _, _ = x.size()

	# 	primitives = self.rop_primitive(x, m, k, v).squeeze(4).squeeze(3)
	# 	states, _cel, _pog = self.rop_state(primitives)

	# 	primitives = primitives.view(B * T, -1)
	# 	states = states.view(B * T, -1)

	# 	reps = self.rop_representation(primitives, states).view(B, T, -1)
	# 	aggr = self.rop_aggregator(states).view(B, T, -1)

	# 	return reps, aggr

	def pack_time(self, x):
		size = x.size()
		T = size[1]
		new_size = torch.Size([-1]) + size[2:]
		return x.view(new_size), T

	def unpack_time(self, x, t):
		size = x.size()
		new_size = torch.Size([-1, t]) + size[1:]
		return x.view(new_size)

	def _igloop(self, ):
		pass

	# @torch.jit.script_method
	def forward(self, 
				cnd_x, cnd_m, cnd_k, cnd_v, 
				qry_x, qry_m, qry_k, qry_v, 
				gamma=.95, 
				asteps=7, rsteps=None):

		# --- Conditional (cnd_*) and query (qry_*) inputs
		# cnd/qry_x: RGB image (B, T, 3, 256, 256)
		# cnd/qry_m: 'Background' heatmap (B, T, 1, 256, 256)
		# cnd/qry_k: Rendered skeleton (B, T, 3, 256, 256)
		# cnd/qry_v: Orientation vector (B, T, 7, 1, 1)

		# Containers to hold outputs.
		prior_means, prior_logvs = [], []
		posterior_means, posterior_logvs = [], []
		output_heat = []
		output_rgb = []

		# Size information.
		Bc, Tc, _, Hc, Wc = cnd_x.size()
		Tq = qry_v.size(1)
		dev = cnd_x.device

		# Number of steps to rewind representation.
		if rsteps is None:
			rsteps = Tq - 1
		else:
			rsteps = min(Tq - 1, rsteps)

		# --- Conditional filtered and aggregated representations
		prim = self.rop_primitive(cnd_x, cnd_m, cnd_k, cnd_v).squeeze(4).squeeze(3)
		prim_packed, cnd_t = self.pack_time(prim)
		state, c_rop, o_rop = self.rop_state(prim)
		state_padded = nn.functional.pad(state, (0, 0, 1, 0), value=0)
		state = self.pack_time(state)[0]
		state_padded = self.pack_time(state_padded)[0]

		cnd_repf = self.unpack_time(self.rop_representation(prim_packed, state), cnd_t)
		cnd_aggr = self.unpack_time(self.rop_aggregator(state_padded), cnd_t + 1)
		end_aggr = gamma * cnd_aggr[:, -2] + cnd_repf[:, -1]

		# --- Query representation primitives
		# 									-> (B, Tq, 256, 1, 1)
		qry_repp = self.rop_primitive(qry_x, qry_m, qry_k, qry_v)

		# --- LSTM hidden/cell/prior output gate for inference/generator operators
		h_iop = torch.zeros(1).expand(Bc, 256, 16, 16)
		c_iop = torch.zeros(1).expand(Bc, 256, 16, 16)
		o_iop = torch.zeros(1).expand(Bc, 256, 16, 16)
		h_gop = torch.zeros(1).expand(Bc, 256, 16, 16)
		c_gop = torch.zeros(1).expand(Bc, 256, 16, 16)
		o_gop = torch.zeros(1).expand(Bc, 256, 16, 16)
		u_gop = torch.zeros(1).expand(Bc, 256, 16, 16)

		# --- Rewind 
		# 								-> (B, Tq, 256), (B, 256)
		rwn_aggr, _ = self.rop_rewind(end_aggr, rewind_steps=rsteps)
		
		# tweaking dimensionality
		rwn_aggr = self.pack_time(rwn_aggr)[0].unsqueeze(2).unsqueeze(2).expand(-1, -1, 16, 16)
		qry_repp = self.pack_time(qry_repp)[0].expand(-1, -1, 16, 16)
		qry_v = self.pack_time(qry_v)[0].expand(-1, -1, 16, 16)

		# --- Inference/generation
		for ast in range(asteps):
			prior_dist, prior_mean, prior_logv = self.gop_prior(h_gop)

			input_iop = torch.cat([rwn_aggr, qry_repp, h_gop], dim=1)
			h_iop, c_iop, o_iop = self.iop_state(input_iop, h_iop, c_iop, o_iop)
			posterior_dist, posterior_mean, posterior_logv = self.iop_posterior(h_iop)
			posterior_z = posterior_dist.rsample()

			input_gop = torch.cat([rwn_aggr, posterior_z, qry_v])
			h_gop, c_gop, o_gop = self.gop_state(input_gop, h_gop, c_gop, o_gop)
			u_gop = u_gop + self.gop_delta(u_gop, h_gop)

			# collect means and log variances
			prior_means.append(prior_mean), prior_logvs.append(prior_logv)
			posterior_means.append(posterior_mean), posterior_logvs.append(posterior_logv)


		# --- Auxiliary classification task
		cat_dist = self.aux_class(u_gop)

		# --- Decoding
		dec_base = self.dop_base(u_gop)
		dec_heat = self.dop_heat(dec_base)
		dec_rgbv = self.dop_rgbv(dec_base, dec_heat)

		'rgbv', 'heat', 'label', 'prior_mean', 'prior_logv', 'posterior_mean', 'posterior_logv'
		return GernOutput(
			rgbv=dec_rgbv,
			heat=dec_heat,
			label=cat_dist,
			cnd_repr=cnd_repf,
			cnd_aggr=cnd_aggr,
			prior_mean=prior_means,
			prior_logv=prior_logvs,
			posterior_mean=posterior_means,
			posterior_logv=posterior_logvs,
			)

	def predict(self, vq, xk=None, mk=None, vk=None, asteps=7, rsteps=None):
		pass
		# agr_k = self.net_status['aggregation_state']
		# agr_h, agr_c = self.net_status['aggregation_hidden']
		# gen_h, gen_c = self.net_status['generator_hidden']
		# dec_h, dec_c = self.net_status['decoder_hidden']

		# if (xk, mk, vk) != (None, None, None):
		# 	rep_k = self.representations(xk, mk, vk_)
		# 	agr_k, (agr_h, agr_c) = self.rnn_representation(rep_k, agr_k, agr_h, agr_c)
		# agr_k_ = agr_k.expand(-1, -1, 16, 16)

		# if rsteps is None:
		# 	rsteps = vq.size(1)
		# else:
		# 	rsteps = min(rsteps, vq.size(1))

		# for rst in range(rsteps):
		# 	for ast in range(asteps):
		# 		prior_dist, prior_mean, prior_logv = self.prior_factor(gen_h)
		# 		prior_z = prior_dist.rsample()

		# 		rnn_generator_input = torch.cat([vq_.squeeze(1), agr_k_, dec_h, prior_z], dim=1)
		# 		gen_h, gen_c = self.rnn_generator(rnn_generator_input, gen_h, gen_c)

		# 		gen_u = gen_u + self.generator_delta(gen_h)


		# 	rnn_decoder_input = torch.cat([gen_u, agr_k_], dim=1)
		# 	dec_h, dec_c = self.rnn_decoder(rnn_decoder_input, (dec_h, dec_c))

		# 	dec_w = self.image_decoder('heatmap', x=dec_h)
		# 	dec_x = self.image_decoder('rgb', y=dec_w, x=torch.cat([dec_h, gen_u], dim=1))

		# 	output_heat.append(dec_w)
		# 	output_rgb.append(dec_x)

		# self.net_status.update({
		# 	'aggregator_state': agr_k,
		# 	'aggregator_hidden': (agr_h, agr_c),
		# 	'generator_hidden': (gen_h, gen_c),
		# 	'decoder_hidden': (dec_h, dec_c),
		# 	'cnd_repr': 
		# 	'cnd_appr': 
		# 	})


if __name__ == '__main__':
	# x = torch.randn(1, 3, 256, 256)
	# k = torch.randn(1, 1, 256, 256)
	# m = torch.randn(1, 3, 256, 256)
	# q = torch.randn(1, 7,   1,   1)
	# net = RepresentationEncoderPrimitive()
	# # print(net.features[0](torch.cat([x, k, m], dim=1)).size())
	# print(net(x, k, m, q).size())
	print('Called')