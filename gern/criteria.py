from torchvision import models as visionmodels
from torch.utils import checkpoint as ptcp
import torch
import torch.nn as nn
import numpy as np
import scipy.stats as ss


def gaussian_kernel(size, sigma):
	assert sigma > 0
	b = size // 2 + (size % 2) / 2
	x = np.linspace(-b, b, size + 1)
	k1d = np.diff(ss.norm(0, sigma).cdf(x))
	k2d = np.outer(k1d, k1d)
	k2d = k2d.reshape([1, 1, size, size]) / k2d.sum()
	return k2d


class LaplacianPyramidLoss(nn.Module):
	def __init__(self, ksize, sigma, plevel):
		super(LaplacianPyramidLoss, self).__init__()
		self.register_buffer('kernel', torch.tensor(gaussian_kernel(ksize, sigma), dtype=torch.float32))
		self.kernel_size = ksize
		self.padding = ksize // 2
		self.sigma = sigma
		self.pyramid_levels = plevel

	def lappyr(self, x):
		# returns the laplacian pyramid given input
		pyr = []
		cur = x
		for l in range(self.pyramid_levels):
			blr = nn.functional.conv2d(cur, self.kernel.expand(-1, cur.size(1), -1, -1), padding=self.padding)
			pyr.append(cur - blr)
			cur = nn.functional.avg_pool2d(blr, 2, 2)

		pyr.append(cur)
		return pyr

	def forward(self, output, target):
		pyr_output = self.lappyr(output)
		pyr_target = self.lappyr(target)

		L = [torch.norm(o - t, p=1) / o.numel() for o, t in zip(pyr_output, pyr_target)]
		L = sum(L) * output.size(0)

		return L


class PerceptualLoss(nn.Module):
	def __init__(self, vgg_layers=[6, 13, 26, 39, 52]):
		super(PerceptualLoss, self).__init__()
		self.vggf = visionmodels.vgg19_bn(pretrained=True).eval().features
		self.N = len(vgg_layers)
		self.mseloss = nn.MSELoss(reduction='sum')
		self.hook_handles = []
		self.hidden_outputs = []

		for p in self.vggf.parameters():
			p.requires_grad = False

		# register hooks
		def fhk(module, inputs, output):
			self.hidden_outputs.append(output)

		for l in vgg_layers:
			hh = self.vggf[l].register_forward_hook(fhk)
			self.hook_handles.append(hh)

	def deregister_hooks(self):
		# exhause hook handles and remove.
		while len(self.hook_handles) > 0:
			self.hook_handles.pop().remove()

	def forward(self, output, target):
		# exhaust intermediate outputs and evaluate mse loss.
		inp = torch.cat([output, target], dim=0)
		self.vggf(inp)

		running_loss = 0.
		while len(self.hidden_outputs) > 0:
			pred, targ = torch.chunk(self.hidden_outputs.pop(), 2, dim=0)
			running_loss += (self.mseloss(pred, targ) / pred.size(0))

		return running_loss / self.N


def heatmap_loss(output, target):
	loss = nn.functional.binary_cross_entropy_with_logits(output.heat, target.heat, reduction='sum')
	loss = loss / output.heat.size(0)
	return loss

def classifier_loss(output, target):
	loss = nn.functional.cross_entropy(output.label, target.label, reduction='sum')
	loss = loss / output.label.size(0)
	return loss

def classifier_accuracy(output, target):
	label = output.label.max(dim=1)[1].view(-1)
	accu = (label == target.label).float().sum() / label.size(0) * 100
	return accu

def aggregator_loss(output):
	rep = output.cnd_repr
	agg = output.cnd_aggr
	target = (output.gamma * agg[:, :-1] + rep).detach()
	loss = 0.5 * (agg[:, 1:] - target).pow(2).sum() / (rep.size(0) * rep.size(1))
	return loss

def kl_divergence(output):
	kl = 0.
	B = output.posterior_mean[0].size(0)
	N = len(output.posterior_mean)

	statz = zip(output.posterior_mean, output.posterior_logv, output.prior_mean, output.prior_logv)
	for qp in statz:
		qm, qv, pm, pv = map(lambda v: v.view(B, -1), qp)
		kl += (-pv + qv).exp().sum(1) + (qm - pm).pow(2).div(pv.exp()).sum(1) + pv.sum(1) - qv.sum(1)
	kl = kl.mean() / N
	return kl


class GernCriterion(nn.Module):
	def __init__(self):
		super(GernCriterion, self).__init__()
		# loss functions
		# - Perceptual loss to replace naive L2 loss in pixel space
		self.lfcn_percept = PerceptualLoss()
		# - Heatmap loss is a binary cross-entropy with logits
		self.lfcn_heatmap = heatmap_loss
		# - Classifier loss is a cross-entropy loss
		self.lfcn_classifier = classifier_loss
		# - Aggregate loss is a temporal difference loss (L2)
		self.lfcn_aggregate = aggregator_loss
		# - KL divergence
		self.lfcn_kldiv = kl_divergence
		# - Classifier accuracy
		self.accu_classifier = classifier_accuracy

		# states
		self.l_percept = None
		self.l_heatmap = None
		self.l_classifier = None
		self.l_aggregate = None
		self.l_kldiv = None
		self.accuracy = None

	def forward(self, gern_output, gern_target, weights):
		self.l_percept = ptcp.checkpoint(self.lfcn_percept, gern_output.rgbv, gern_target.rgbv)
		self.l_heatmap = self.lfcn_heatmap(gern_output, gern_target)
		self.l_classifier = self.lfcn_classifier(gern_output, gern_target)
		self.l_aggregate = self.lfcn_aggregate(gern_output)
		self.l_kldiv = self.lfcn_kldiv(gern_output)
		self.accuracy = self.accu_classifier(gern_output, gern_target).item()
		return self.weighted_sum(weights), self.accuracy

	def weighted_sum(self, weights):
		L = [self.l_percept, self.l_heatmap, self.l_classifier, self.l_aggregate, self.l_kldiv]
		assert len(L) == len(weights)
		return sum(map(lambda l, w: l * w, L, weights))

	def item(self):
		"""Return a 5-tuple consisting of all criteria except accuracy."""
		return (self.l_percept.item(), self.l_heatmap.item(), 
				self.l_classifier.item(), self.l_aggregate.item(), 
				self.l_kldiv.item())