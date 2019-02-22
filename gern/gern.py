import torch, imageio, os, random
import torch.nn as nn
from numpy.random import randint
from torch.distributions import Normal
from tqdm import tqdm
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

	def forward(self, x, k, m, q):
		# x: RGB input (-1x3x256x256)
		# k: Openpose keypoint without blending (-1x1x256x256)
		# m: Openpose background heatmap
		# q: Query vector (-1x7x1x1)
		batch_size = x.size(0)
		num_steps = x.size(1)
		dev = x.device
		
		inp = torch.cat([x, k, m], dim=1)
		inp = self.features[0](inp)
		inp = torch.cat([inp, q.expand(-1, -1, inp.size(2), inp.size(3))], dim=1)
		out = self.features[1](inp)

		return out


class RepresentationEncoderState(nn.Module):
	def __init__(self, input_size=256, hidden_size=128, zoneout=.15, init=False):
		super(RepresentationEncoderState, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.network = LSTMCell(input_size, hidden_size, zoneout=zoneout, bias=True)

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
		if pog is None:
			try:
				pog = self._init_pog.expand(batch_size, -1)
			except:
				pog = torch.zeros(1, dtype=torch.float32, device=dev).expand(default_size)
		if hid is None:
			try:
				hid = self._init_hid.expand(batch_size, -1)
			except:
				hid = torch.zeros(1, dtype=torch.float32, device=dev).expand(default_size)
		if cel is None:
			try:
				cel = self._init_cel.expand(batch_size, -1)
			except:
				cel = torch.zeros(1, dtype=torch.float32, device=dev).expand(default_size)
		
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

		self.features = nn.Sequential(
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
			nn.Linear(hidden_size, primitive_size)
			)

		print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))

	def forward(self, prim, state):
		inp = torch.cat([prim, state], dim=1)

		return self.features(inp) + prim



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
	def __init__(self, input_size=256, hidden_size=128, learn_init=False):
		super(AggregateRewind, self).__init__()

		self.hidden_size = hidden_size
		self.algo = LSTMCell(input_size, hidden_size, zoneout=.15)
		self.rewind = nn.Sequential(
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

		if hid is None:
			try:
				hid = self.hid.expand(x.size(1), -1)
			except:
				hid = torch.zeros(1, device=x.device).expand(x.size(0), self.hidden_size)
		if cel is None:
			try:
				cel = self.cel.expand(x.size(1), -1)
			except:
				cel = torch.zeros(1, device=x.device).expand(x.size(0), self.hidden_size)
		if pog is None:
			pog = torch.zeros(1, device=x.device).expand(x.size(0), self.hidden_size)

		rewind = [x]
		for _ in range(rewind_steps):
			hid, cel, pog = self.algo(rewind[-1], hid, cel, pog)

			r_input = torch.cat([x, hid], dim=1)
			r = self.rewind(r_input) + x
			rewind.append(r)

		return torch.stack(rewind, dim=1), pog


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()

		self.decoders = nn.ModuleDict({
			'Head1': nn.Sequential(
				SkipConnect(
					nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1)),
				nn.BatchNorm2d(256),
				nn.ReLU(True)),
			'Head2': nn.Sequential(
				nn.Conv2d(512, 256, (3, 3), (1, 1), padding=1),
				nn.BatchNorm2d(256),
				nn.ReLU(True),
				SkipConnect(
					nn.Conv2d(256, 256, (1, 1), (1, 1), bias=False),
					nn.Sequential(
						nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1))),
				nn.BatchNorm2d(256),
				nn.ReLU(True)),
			'End1': nn.Conv2d(16, 1, (3, 3), (1, 1), padding=1),
			'End2': nn.Sequential(
				nn.Conv2d(17, 16, (3, 3), (1, 1), padding=1),
				nn.BatchNorm2d(16),
				nn.ReLU(True),
				nn.Conv2d(16, 3, (3, 3), (1, 1), padding=1)),
			'Shared': nn.Sequential(
				nn.Conv2d(256, 128, (3, 3), (1, 1), padding=1),
				nn.BatchNorm2d(128),
				nn.ReLU(True),
				SkipConnect(
					nn.Conv2d(128, 128, (1, 1), (1, 1), bias=False),
					nn.Sequential(nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1),
								  nn.BatchNorm2d(128), 
								  nn.ReLU(True),
								  nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1))),
				nn.BatchNorm2d(128),
				nn.ReLU(True),
				BilinearInterpolate((2, 2)),
				nn.Conv2d(128, 64, (3, 3), (1, 1), padding=1),
				nn.BatchNorm2d(64),
				nn.ReLU(True),
				SkipConnect(
					nn.Conv2d(64, 64, (1, 1), (1, 1), bias=False),
					nn.Sequential(nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1), 
								  nn.BatchNorm2d(64), 
								  nn.ReLU(True),
								  nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1))),
				nn.BatchNorm2d(64),
				nn.ReLU(True),
				BilinearInterpolate((2, 2)),
				nn.Conv2d(64, 32, (3, 3), (1, 1), padding=1),
				nn.BatchNorm2d(32),
				nn.ReLU(True),
				SkipConnect(
					nn.Conv2d(32, 32, (1, 1), (1, 1), bias=False),
					nn.Sequential(nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1), 
								  nn.BatchNorm2d(32), 
								  nn.ReLU(True),
								  nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1), 
								  nn.BatchNorm2d(32), 
								  nn.ReLU(True),
								  nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1), 
								  nn.BatchNorm2d(32), 
								  nn.ReLU(True),
								  nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1))),
				nn.BatchNorm2d(32),
				nn.ReLU(True),
				BilinearInterpolate((2, 2)),
				nn.Conv2d(32, 16, (3, 3), (1, 1), padding=1),
				nn.BatchNorm2d(16),
				nn.ReLU(True),
				SkipConnect(
					nn.Conv2d(16, 16, (1, 1), (1, 1), bias=False),
					nn.Sequential(nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1), 
								  nn.BatchNorm2d(16), 
								  nn.ReLU(True),
								  nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1), 
								  nn.BatchNorm2d(16), 
								  nn.ReLU(True),
								  nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1), 
								  nn.BatchNorm2d(16), 
								  nn.ReLU(True),
								  nn.Conv2d(16, 16, (3, 3), (1, 1), padding=1))),
				nn.BatchNorm2d(16),
				nn.ReLU(True))
			})

	def forward(self, target, x=None, y=None):
		if target == 'heatmap':
			out = self.decoders['Head1'](x)
			out = self.decoders['Shared'](out)
			out = self.decoders['End1'](out)
			return out
		elif target == 'rgb':
			out = self.decoders['Head2'](x)
			out = self.decoders['Shared'](out)
			out = self.decoders['End2'](torch.cat([out, y], dim=1))
			return out
		else:
			raise ValueError('')


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
	def __init__(self, input_channels, output_channels):
		super(RecurrentCell, self).__init__()
		self.input_channels = input_channels
		self.output_channels = output_channels

		self.features = nn.Conv2d(input_channels + output_channels, output_channels * 4, (3, 3), (1, 1), padding=1)

	def forward(self, x, h=None, c=None):
		batch_size, _, height, width = x.size()
		device = x.device

		if h is None or c is None:
			h = torch.zeros(batch_size, self.output_channels, height, width, device=device)
			c = torch.zeros(batch_size, self.output_channels, height, width, device=device)

		rnn_inp = torch.cat([x, h], dim=1)
		gate_f, gate_i, gate_s, gate_o = torch.chunk(self.features(rnn_inp), 4, dim=1)

		gate_f = torch.sigmoid(gate_f)
		gate_i = torch.sigmoid(gate_i)
		gate_s = torch.tanh(gate_s)
		gate_o = torch.sigmoid(gate_o)
		c_next = gate_f * c + gate_i * gate_s
		h_next = gate_o * torch.tanh(c_next)

		return h_next, c_next


class GeRN(nn.Module):
	def __init__(self):
		super(GeRN, self).__init__()

		self.vgg19_f26 = visionmodels.vgg19_bn(pretrained=True).eval().features[:27]
		self.representations = Representation()
		self.rnn_representation = RecurrentRepresentationAggregator(512, 512, 256)  # inp, hid, out channels
		self.rnn_inference = RecurrentRepresentationAggregator(512, 128, 128)
		self.rnn_encoder = RecurrentCell(1152, 256)
		self.rnn_generator = RecurrentCell(775, 256)
		self.rnn_decoder = RecurrentCell(512, 256)
		self.image_decoder = Decoder()
		self.prior_factor = GaussianFactor()
		self.posterior_factor = GaussianFactor()
		self.generator_delta = nn.Sequential(
			nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1),
			nn.ReLU(True),
			nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1))

		self.net_status = dict()
		self.reset_aggregator()
		self.reset_rnns()

		for p in self.vgg19_f26.parameters():
			p.requires_grad = False

		print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))

	def forward(self, xk, mk, vk, xq, mq, vq, asteps=7, rsteps=None):
		# (contextual inputs) t=[1, 2, 3 ,..., T]
		# 	xk: (-1xTx3x128x128)
		# 	mk: (-1xTx77x128x128)
		# 	vk: (-1xTx7x1x1)
		# (queries) t=(T, T-1, T-2, ...)
		# 	xq: 𝜏
		# 	mq: 
		# 	vq: (-1x1x7x1x1)
		# (sampling arguments)
		# 	asteps:
		# 	rsteps:

		prior_means, prior_logvs = [], []
		posterior_means, posterior_logvs = [], []
		output_heat = []
		output_rgb = []

		batch_size = xk.size(0)
		num_steps_k = xk.size(1)
		num_steps_q = xq.size(1)
		device = xk.device

		agr_k = self.net_status['aggregator_state']
		agr_h, agr_c = self.net_status['aggregator_hidden']
		gen_h, gen_c = self.net_status['generator_hidden']
		dec_h, dec_c = self.net_status['decoder_hidden']
			
		if gen_h is None or gen_c is None:
			gen_h = torch.zeros(batch_size, 256, 16, 16, device=device)
			gen_c = torch.zeros(batch_size, 256, 16, 16, device=device)
		if dec_h is None or dec_c is None:
			dec_h = torch.zeros(batch_size, 256, 16, 16, device=device)
			dec_c = torch.zeros(batch_size, 256, 16, 16, device=device)
		enc_h = torch.zeros(batch_size, 256, 16, 16, device=device)
		enc_c = torch.zeros(batch_size, 256, 16, 16, device=device)
		gen_u = torch.zeros(batch_size, 256, 16, 16, device=device)

		agr_q = torch.zeros(batch_size, 128, 1, 1, device=device)
		agr_g = torch.zeros(batch_size, 128, 1, 1, device=device)
		agr_b = torch.zeros(batch_size, 128, 1, 1, device=device)

		vk_ = vk.expand(-1, -1, -1, 16, 16)
		vq_ = vq.expand(-1, -1, -1, 16, 16)

		rep_k = self.representations(xk, mk, vk_)
		rep_q = self.representations(xq, mq, vq_)
		agr_k, (agr_h, agr_c) = self.rnn_representation(rep_k, agr_k, agr_h, agr_c)
		agr_k_ = agr_k.expand(-1, -1, 16, 16)

		if rsteps is None:
			rsteps = num_steps_q
		else:
			rsteps = min(rsteps, num_steps_q)

		for rst in range(rsteps):

			rnn_inference_input = rep_q[:, rst].unsqueeze(1)
			agr_q, (agr_g, agr_b) = self.rnn_inference(rnn_inference_input, agr_q, agr_g, agr_b)
			agr_q_ = torch.cat([agr_q, agr_k], dim=1).expand(-1, -1, 16, 16)

			for ast in range(asteps):
				prior_dist, prior_mean, prior_logv = self.prior_factor(gen_h)

				rnn_encoder_input = torch.cat([agr_q_, gen_h, gen_u, dec_h], dim=1)
				enc_h, enc_c = self.rnn_encoder(rnn_encoder_input, enc_h, enc_c)

				posterior_dist, posterior_mean, posterior_logv = self.posterior_factor(enc_h)
				posterior_z = posterior_dist.rsample()

				rnn_generator_input = torch.cat([vq_.squeeze(1), agr_k_, dec_h, posterior_z], dim=1)
				gen_h, gen_c = self.rnn_generator(rnn_generator_input, gen_h, gen_c)

				gen_u = gen_u + self.generator_delta(gen_h)

				prior_means.append(prior_mean)
				prior_logvs.append(prior_logv)
				posterior_means.append(posterior_mean)
				posterior_logvs.append(posterior_logv)

			rnn_decoder_input = torch.cat([gen_u, agr_k_], dim=1)
			dec_h, dec_c = self.rnn_decoder(rnn_decoder_input, (dec_h, dec_c))

			dec_w = self.image_decoder('heatmap', x=dec_h)
			dec_x = self.image_decoder('rgb', y=dec_w, x=torch.cat([dec_h, gen_u], dim=1))

			output_heat.append(dec_w)
			output_rgb.append(dec_x)

		self.net_status.update({
			'aggregator_state': agr_k,
			'aggregator_hidden': (agr_h, agr_c),
			'generator_hidden': (gen_h, gen_c),
			'decoder_hidden': (dec_h, dec_c)
			})

		return output_rgb, output_heat, (prior_means, prior_logvs), (posterior_means, posterior_logvs)

	def reset_aggregator(self):
		self.net_status.update({
			'aggregator_repr': None,
			'aggregator_hidden': (None, None)
			})
	def reset_rnns(self):
		self.net_status.update({
			'generator_hidden': (None, None),
			'decoder_hidden': (None, None)
			})

	def reset_net(self):
		self.reset_aggregator()
		self.reset_rnns()

	def predict(self, vq, xk=None, mk=None, vk=None, asteps=7, rsteps=None):
		
		agr_k = self.net_status['aggregation_state']
		agr_h, agr_c = self.net_status['aggregation_hidden']
		gen_h, gen_c = self.net_status['generator_hidden']
		dec_h, dec_c = self.net_status['decoder_hidden']

		if (xk, mk, vk) != (None, None, None):
			rep_k = self.representations(xk, mk, vk_)
			agr_k, (agr_h, agr_c) = self.rnn_representation(rep_k, agr_k, agr_h, agr_c)
		agr_k_ = agr_k.expand(-1, -1, 16, 16)

		if rsteps is None:
			rsteps = vq.size(1)
		else:
			rsteps = min(rsteps, vq.size(1))

		for rst in range(rsteps):
			for ast in range(asteps):
				prior_dist, prior_mean, prior_logv = self.prior_factor(gen_h)
				prior_z = prior_dist.rsample()

				rnn_generator_input = torch.cat([vq_.squeeze(1), agr_k_, dec_h, prior_z], dim=1)
				gen_h, gen_c = self.rnn_generator(rnn_generator_input, gen_h, gen_c)

				gen_u = gen_u + self.generator_delta(gen_h)


			rnn_decoder_input = torch.cat([gen_u, agr_k_], dim=1)
			dec_h, dec_c = self.rnn_decoder(rnn_decoder_input, (dec_h, dec_c))

			dec_w = self.image_decoder('heatmap', x=dec_h)
			dec_x = self.image_decoder('rgb', y=dec_w, x=torch.cat([dec_h, gen_u], dim=1))

			output_heat.append(dec_w)
			output_rgb.append(dec_x)

		self.net_status.update({
			'aggregator_state': agr_k,
			'aggregator_hidden': (agr_h, agr_c),
			'generator_hidden': (gen_h, gen_c),
			'decoder_hidden': (dec_h, dec_c)
			})


if __name__ == '__main__':
	x = torch.randn(1, 3, 256, 256)
	k = torch.randn(1, 1, 256, 256)
	m = torch.randn(1, 3, 256, 256)
	q = torch.randn(1, 7,   1,   1)
	net = RepresentationEncoderPrimitive()
	# print(net.features[0](torch.cat([x, k, m], dim=1)).size())
	print(net(x, k, m, q).size())