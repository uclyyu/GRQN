import torch, imageio, os, random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
from torch.distributions import Normal
from torch.nn.functional import interpolate
from tqdm import tqdm
from torchvision import models as visionmodels


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


class GroupNorm(nn.Module):
	def __init__(self, channels, groups, eps=1e-5):
		super(GroupNorm, self).__init__()

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


class Representation(nn.Module):
	def __init__(self):
		super(Representation, self).__init__()

		self.vgg19_f26 = visionmodels.vgg19_bn(pretrained=True).eval().features[:27]
		self.features = self.features = nn.ModuleList([
			# 0 ---
			nn.Sequential(
				nn.Conv2d(77, 32, (3, 3), (1, 1), padding=1), 
				nn.BatchNorm2d(32), 
				nn.ReLU(True),
				SkipConnect(
					nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1)), 
				nn.BatchNorm2d(32), 
				nn.ReLU(True),
				nn.MaxPool2d((2, 2), stride=(2, 2)),
				nn.Conv2d(32, 64, (3, 3), (1, 1), padding=1), 
				nn.BatchNorm2d(64), 
				nn.ReLU(True),
				SkipConnect(
					nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1)), 
				nn.BatchNorm2d(64), 
				nn.ReLU(True),
				nn.MaxPool2d((2, 2), stride=(2, 2)),
				nn.Conv2d( 64, 128, (3, 3), (1, 1), padding=1), 
				nn.BatchNorm2d(128), 
				nn.ReLU(True),
				SkipConnect(
					nn.Sequential(
						nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1), 
						nn.BatchNorm2d(128), 
						nn.ReLU(True),
						nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1), 
						nn.BatchNorm2d(128), 
						nn.ReLU(True),
						nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1))), 
				nn.BatchNorm2d(128), 
				nn.ReLU(True),
				nn.MaxPool2d((2, 2), stride=(2, 2))
			),
			# 1 ---
			nn.Sequential(
				nn.Conv2d(391, 512, (3, 3), (2, 2), padding=1), 
				nn.BatchNorm2d(512),
				nn.ReLU(True),
				SkipConnect(
					nn.Conv2d(512, 512, (1, 1), (1, 1), bias=False),
					nn.Sequential(
						nn.Conv2d(512, 512, (3, 3), (1, 1), padding=1),
						nn.BatchNorm2d(512),
						nn.ReLU(True),
						nn.Conv2d(512, 512, (3, 3), (1, 1), padding=1),
						nn.BatchNorm2d(512),
						nn.ReLU(True),
						nn.Conv2d(512, 512, (3, 3), (1, 1), padding=1))
					), 
				nn.BatchNorm2d(512),
				nn.ReLU(True),
				nn.Conv2d(512, 512, (3, 3), (2, 2), padding=1),
				nn.BatchNorm2d(512),
				nn.ReLU(True),
				nn.Conv2d(512, 512, (3, 3), (2, 2))
			)
		])


	def forward(self, x, m, q):
		# x: RGB input (-1xNx3x128x128) [-1, 1]
		# m: OpenPose heatmaps (-1xNx77x128x128) [-1, 1]
		# q: Query vector (-1xNx7x16x16) [-1, 1]
		# r: Latent representation (256x1x1; default: zeros) 
		# h: LSTM hidden state (512x1x1; default: zeros)
		# c: LSTM cell state (512x1x1; default: zeros)
		batch_size = x.size(0)
		num_steps = x.size(1)
		dev = x.device
		
		# collapse time axis
		x = x.view(-1,  3, 128, 128)
		m = m.view(-1, 77, 128, 128)
		q = q.view(-1,  7,  16,  16)

		# encode rgb images
		enc_x = self.vgg19_f26(x)
		# encode heatmaps
		enc_m = self.features[0](m)

		enc_z = self.features[1](torch.cat([enc_x, enc_m, q], dim=1))

		return enc_z.view(batch_size, num_steps, 512, 1, 1)


class RecurrentRepresentationAggregator(nn.Module):
	def __init__(self, input_channels=512, hidden_channels=512, output_channels=256):
		super(RecurrentRepresentationAggregator, self).__init__()

		self.input_channels = input_channels
		self.hidden_channels = hidden_channels
		self.output_channels = output_channels
		
		size_inp = input_channels + hidden_channels + output_channels
		size_out = hidden_channels * 4
		self.rnn = nn.Conv2d(size_inp, size_out, (1, 1), (1, 1))
		self.feature = nn.Conv2d(hidden_channels, output_channels, (1, 1), (1, 1))
		
	def forward(self, z, r=None, h=None, c=None):
		# z: Encoded representations
		# r: Latent representation (256x1x1; default: zeros) 
		# h: LSTM hidden state (512x1x1; default: zeros)
		# c: LSTM cell state (512x1x1; default: zeros)
		batch_size = z.size(0)
		num_steps = z.size(1)
		dev = z.device
		
		if r is None:
			r = torch.zeros(batch_size, self.output_channels, 1, 1, device=dev)
		if h is None or c is None:
			h = torch.zeros(batch_size, self.hidden_channels, 1, 1, device=dev)
			c = torch.zeros(batch_size, self.hidden_channels, 1, 1, device=dev)
		
		r_next = r
		h_next = h
		for n in range(num_steps):
			rnn_inp = torch.cat([r_next, h_next, z[:, n]], dim=1)
			gate_f, gate_i, gate_s, gate_o = torch.chunk(self.rnn(rnn_inp), 4, dim=1)

			gate_f = torch.sigmoid(gate_f)
			gate_i = torch.sigmoid(gate_i)
			gate_s = torch.tanh(gate_s)
			gate_o = torch.sigmoid(gate_o)
			c_next = gate_f * c + gate_i * gate_s
			h_next = gate_o * torch.tanh(c_next)

			r_next = r_next + self.feature(h_next)
		
		return r_next, (h_next, c_next)


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


class PerceptualLoss(nn.Module):
	def __init__(self, vgg_layers=[6, 13, 26, 39, 52]):
		super(PerceptualLoss, self).__init__()
		self.vggf = visionmodels.vgg19_bn(pretrained=True).eval().features
		self.hook_handles = []
		self.outputs_pred = []
		self.outputs_targ = []
		self.vgg_layers = vgg_layers

		for p in self.vggf.parameters():
			p.requires_grad = False

		print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))

	def _deregister_hooks(self):
		print('D: {}'.format(len(self.hook_handles)))
		while len(self.hook_handles) > 0:
			h = self.hook_handles.pop()
			h.remove()
			print('D : {}'.format(len(self.hook_handles)))

	def _register_hook_pred(self):
		def _hook(module, inp, out):
			self.outputs_pred.append(out)

		self._deregister_hooks()
		for l in self.vgg_layers:
			h = self.vggf[l].register_forward_hook(_hook)
			self.hook_handles.append(h)
			print('Rp: {}'.format(len(self.hook_handles)))

	def _register_hook_targ(self):
		def _hook(module, inp, out):
			self.outputs_targ.append(out)

		self._deregister_hooks()
		for l in self.vgg_layers:
			h = self.vggf[l].register_forward_hook(_hook)
			self.hook_handles.append(h)
			print('Rt: {}'.format(len(self.hook_handles)))

	def forward(self, pred, targ):
		self._register_hook_pred()
		self.vggf(pred)

		self._register_hook_targ()
		self.vggf(targ)

		loss = 0
		while len(self.outputs_pred) > 0:
			mse = (self.outputs_pred.pop() - self.outputs_targ.pop()).pow(2).sum(dim=[1, 2, 3]).mean()
			loss = loss + mse 

		return loss / len(self.vgg_layers)

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