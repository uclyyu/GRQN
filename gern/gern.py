import torch, random
import torch.nn as nn
from numpy.random import randint
from torch.distributions import Normal
from torch.utils import checkpoint as ptcp
from collections import namedtuple
from .utils import count_parameters
from .model import *


class GeRN(torch.jit.ScriptModule):
	def __init__(self):
		super(GeRN, self).__init__()

		# --- default sizes/dimensionality
		Nr = 256  # aggregated representation
		Nh = 256  # recurrent cell hidden
		Nv =   7  # query vector

		# --- representaiton operators
		self.rop_primitive = RepresentationEncoderPrimitive()
		self.rop_state = RepresentationEncoderState()
		self.rop_representation = RepresentationEncoder()

		# --- inference operators
		self.iop_posterior = GaussianFactor()
		self.iop_state = RecurrentCell(Nr * 2 + Nh, Nh)

		# --- generation operators
		self.gop_prior = GaussianFactor()
		self.gop_state = RecurrentCell(Nr + Nh + Nv, Nh)
		self.gop_delta = GeneratorDelta()

		# --- classifier
		self.aux_class = LatentClassifier()

		# --- decoding operators
		self.dop_rgbv = DecoderRGBVision()

		print('{}: {:,} trainable parameters.'.format(self.__class__.__name__, count_parameters(self)))

	def pack_time(self, x):
		size = x.size()
		T = size[1]
		new_size = torch.Size([-1]) + size[2:]
		return x.contiguous().view(new_size), T

	def unpack_time(self, x, t):
		size = x.size()
		new_size = torch.Size([-1, t]) + size[1:]
		return x.contiguous().view(new_size)

	@torch.jit.script_method
	def _forward_mc_loop(self, asteps, cnd_repr, qry_repp, qry_v, h_iop, c_iop, o_iop, h_gop, c_gop, o_gop, u_gop, prior_means, prior_logvs, posterior_means, posterior_logvs):
		# type: (int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tuple[Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]]

		for ast in range(asteps):
			prior_z, prior_mean, prior_logv = self.gop_prior(h_gop)

			input_iop = torch.cat([cnd_repr, qry_repp, h_gop], dim=1)
			h_iop, c_iop, o_iop = self.iop_state(input_iop, h_iop, c_iop, o_iop)
			posterior_z, posterior_mean, posterior_logv = self.iop_posterior(h_iop)

			input_gop = torch.cat([cnd_repr, posterior_z, qry_v], dim=1)
			h_gop, c_gop, o_gop = self.gop_state(input_gop, h_gop, c_gop, o_gop)
			u_gop = u_gop + self.gop_delta(u_gop, h_gop)

			# collect means and log variances
			prior_means.append(prior_mean), prior_logvs.append(prior_logv)
			posterior_means.append(posterior_mean), posterior_logvs.append(posterior_logv)

		return u_gop, prior_means, prior_logvs, posterior_means, posterior_logvs

	@torch.jit.script_method
	def _predict_mc_loop(self, asteps, cnd_repr, qry_v, h_gop, c_gop, o_gop, u_gop, prior_means=[], prior_logvs=[]):
	# type: (int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, List[Tensor], List[Tensor]]
		for ast in range(asteps):
			prior_z, prior_mean, prior_logv = self.gop_prior(h_gop)

			input_gop = torch.cat([cnd_repr, prior_z, qry_v], dim=1)
			h_gop, c_gop, o_gop = self.gop_state(input_gop, h_gop, c_gop, o_gop)
			u_gop = u_gop + self.gop_delta(u_gop, h_gop)

			# collect means and log variances
			prior_means.append(prior_mean), prior_logvs.append(prior_logv)

		return u_gop, prior_means, prior_logvs

	def forward(self, 
				cnd_x, cnd_v, 
				qry_x, qry_v,
				label,
				asteps=16):
		# --- Conditional (cnd_*) and query (qry_*) inputs
		# cnd/qry_x: RGB image (B, T, 3, 64, 64) or (B, , 6, 64, 64)
		# cnd/qry_v: Orientation vector (B, T, 7, 1, 1)
		# * For conditional inputs, time (T) will be reduced, this is not the case
		#  for query inputs. Different T's are allowed for each input types. 
		# ---

		# Containers to hold outputs.
		prior_means, prior_logvs = [], []
		posterior_means, posterior_logvs = [], []
		output_heat = []
		output_rgb = []

		# Size information.
		Bc, Tc, _, Hc, Wc = cnd_x.size()
		Qc = qry_v.size(1)
		dev = cnd_x.device

		# --- Conditional filtered and aggregated representations
		cnd_prim = self.rop_primitive(cnd_x, cnd_v)
		cnd_state, c_rop_sta, o_rop_sta = self.rop_state(cnd_prim)
		h_rop_enc, cnd_repr, o_rop_enc = self.rop_representation(cnd_prim, cnd_state)

		# --- Query representation primitives
		qry_repp = self.rop_primitive(qry_x, qry_v)

		# --- LSTM hidden/cell/prior output gate for inference/generator operators
		h_iop = torch.zeros(1, device=dev).expand(Bc * Qc * Tc, 256, 16, 16)
		c_iop = torch.zeros(1, device=dev).expand(Bc * Qc * Tc, 256, 16, 16)
		o_iop = torch.zeros(1, device=dev).expand(Bc * Qc * Tc, 256, 16, 16)
		h_gop = torch.zeros(1, device=dev).expand(Bc * Qc * Tc, 256, 16, 16)
		c_gop = torch.zeros(1, device=dev).expand(Bc * Qc * Tc, 256, 16, 16)
		o_gop = torch.zeros(1, device=dev).expand(Bc * Qc * Tc, 256, 16, 16)
		u_gop = torch.zeros(1, device=dev).expand(Bc * Qc * Tc, 256, 16, 16)
		
		# tweaking dimensionality
		cnd_repr = (cnd_repr  # B, C, ---> B, (Q, T), C, 16, 16
					.unsqueeze(1).unsqueeze(2).unsqueeze(4).unsqueeze(5)
					.expand(-1, Qc, Tc, -1, 16, 16)
					.view(Bc * Qc * Tc, -1, 16, 16))
		qry_repp = (qry_repp  # B, Q, T, C,  ---> B, Q, T, C, (16, 16)
					.unsqueeze(4).unsqueeze(5)
					.expand(-1, -1, -1, -1, 16, 16)
					.view(Bc * Qc * Tc, -1, 16, 16))
		qry_v = (qry_v  # B, Q, C, 1, 1, ---> B, Q, (T,), C, 16, 16
				 .unsqueeze(2)
				 .expand(-1, -1, Tc, -1, 16, 16)
				 .view(Bc * Qc * Tc, 7, 16, 16))

		# --- Inference/generation
		u_gop, prior_means, prior_logvs, posterior_means, posterior_logvs = self._forward_mc_loop(
			asteps, cnd_repr, qry_repp, qry_v,
			h_iop, c_iop, o_iop, 
			h_gop, c_gop, o_gop, u_gop,
			prior_means, prior_logvs, posterior_means, posterior_logvs)

		# --- Auxiliary classification task
		cat_dist = self.aux_class(u_gop)

		import pdb
		pdb.set_trace()

		# --- Decoding
		# dec_base = self.dop_base(u_gop)
		# dec_heat = self.dop_heat(dec_base)
		# dec_rgbv = self.dop_rgbv(dec_base, dec_heat)

		return GernOutput(
			rgbv=dec_rgbv,
			heat=dec_heat,
			label=cat_dist,
			gamma=gamma,
			cnd_repr=cnd_repf,
			cnd_aggr=cnd_aggr,
			prior_mean=prior_means,
			prior_logv=prior_logvs,
			posterior_mean=posterior_means,
			posterior_logv=posterior_logvs,
			)

	def make_target(self, qry_x, qry_m, qry_k, qry_v, label, rsteps=None):
		Tq = qry_v.size(1)
		if rsteps is not None:
			Tq = min(Tq, rsteps + 1)
			qry_x = qry_x[:, :Tq]
			qry_m = qry_m[:, :Tq]
			qry_k = qry_k[:, :Tq]
			qry_v = qry_v[:, :Tq]
			
		label = label.unsqueeze(1).expand(-1, qry_x.size(1))
		return GernTarget(
			rgbv=self.pack_time(qry_x)[0],
			heat=self.pack_time(qry_m)[0],
			label=self.pack_time(label)[0])

	def predict(self, cnd_x, cnd_m, cnd_k, cnd_v, qry_v, 
				gamma=.95, 
				asteps=7, rsteps=None):
		# --- Conditional (cnd_*) and query (qry_*) inputs
		# cnd_x    : RGB image (B, T, 3, 256, 256)
		# cnd_m    : 'Background' heatmap (B, T, 1, 256, 256)
		# cnd_k    : Rendered skeleton (B, T, 3, 256, 256)
		# cnd/qry_v: Orientation vector (B, T, 7, 1, 1)
		# * For conditional inputs, time (T) will be reduced, this is not the case
		#  for query inputs. Different T's are allowed for each input types. 
		# ---

		# Containers to hold outputs.
		prior_means, prior_logvs = [], []
		posterior_means, posterior_logvs = [], []
		output_heat = []
		output_rgb = []

		# Size information.
		Bc, Tc, _, Hc, Wc = cnd_x.size()
		Tq = qry_v.size(1)
		dev = cnd_x.device

		# Number of steps to query backward in time.
		if rsteps is None:
			rsteps = Tq - 1
		else:
			rsteps = min(Tq - 1, rsteps)
			Tq = rsteps + 1
			qry_v = qry_v[:, :Tq]


		cnd_prim = self.rop_primitive(cnd_x, cnd_v)
		cnd_state, c_rop_sta, o_rop_sta = self.rop_state(cnd_prim)
		h_rop_enc, cnd_repr, o_rop_enc = self.rop_representation(cnd_prim, cnd_state)

		# --- Conditional filtered and aggregated representations
		cnd_prim = self.rop_primitive(cnd_x, cnd_v)
		cnd_state, c_rop_sta, o_rop_sta = self.rop_state(cnd_prim)
		h_rop_enc, cnd_repr, o_rop_enc = self.rop_representation(cnd_prim, cnd_state)

		# --- LSTM hidden/cell/prior output gate for inference/generator operators
		h_iop = torch.zeros(1, device=dev).expand(Bc * Tq, 256, 16, 16)
		c_iop = torch.zeros(1, device=dev).expand(Bc * Tq, 256, 16, 16)
		o_iop = torch.zeros(1, device=dev).expand(Bc * Tq, 256, 16, 16)
		h_gop = torch.zeros(1, device=dev).expand(Bc * Tq, 256, 16, 16)
		c_gop = torch.zeros(1, device=dev).expand(Bc * Tq, 256, 16, 16)
		o_gop = torch.zeros(1, device=dev).expand(Bc * Tq, 256, 16, 16)
		u_gop = torch.zeros(1, device=dev).expand(Bc * Tq, 256, 16, 16)

		# tweaking dimensionality
		cnd_repr = cnd_repr.unsqueeze(2).unsqueeze(3).expand(-1, -1, 16, 16)
		qry_v = qry_v.expand(-1, -1, 16, 16)

		# --- Inference/generation
		u_gop, prior_means, prior_logvs = self._predict_mc_loop(
			asteps, cnd_repr, qry_v,
			h_gop, c_gop, o_gop, u_gop,
			prior_means, prior_logvs)

		# --- Auxiliary classification task
		cat_dist = self.aux_class(u_gop)

		# --- Decoding
		dec_base = self.dop_base(u_gop)
		dec_heat = self.dop_heat(dec_base)
		dec_rgbv = self.dop_rgbv(dec_base, dec_heat)

		return GernOutput(
			rgbv=dec_rgbv,
			heat=dec_heat,
			label=cat_dist,
			gamma=gamma,
			cnd_repr=cnd_repf,
			cnd_aggr=cnd_aggr,
			prior_mean=prior_means,
			prior_logv=prior_logvs,
			posterior_mean=posterior_means,
			posterior_logv=posterior_logvs,
			)


if __name__ == '__main__':
	from .data import BallTubeDataLoader

	loader = BallTubeDataLoader('/home/yen/data/balltube/train', subset_size=1, batch_size=1)
	net = GeRN()

	for kx, kv, qx, qv, label, qvi in loader:
		net(kx, kv, qx, qv, label)
