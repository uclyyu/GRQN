from torchvision import models as visionmodels
from torch.utils import checkpoint as ptcp
import torch
import torch.nn as nn

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


def kl_divergence(prior_means, prior_logvs, posterior_means, posterior_logvs):
	kl = 0.
	B = posterior_mean[0].size(0)
	N = len(posterior_mean)

	statz = zip(posterior_mean, posterior_logv, prior_mean, prior_logv)
	for qp in statz:
		qm, qv, pm, pv = map(lambda v: v.view(B, -1), qp)
		kl += (-pv + qv).exp().sum(1) + (qm - pm).pow(2).div(pv.exp()).sum(1) + pv.sum(1) - qv.sum(1)
	kl = kl.mean() / N
	return kl


class GernCriterion(nn.Module):
	def __init__(self):
		super(GernCriterion, self).__init__()
		# states
		self.l_rgbv = None
		self.l_classifier = None
		self.l_kldiv = None
		self.accuracy = None

	def forward(self, qry_x, dec_rgbv, cat_dist, prior_means, prior_logvs, posterior_means, posterior_logvs, weights, label):

		self.l_rgbv, self.l_rgbv_index = (dec_rgbv - qry_x).mean(dim=[3, 4, 5]).min(dim=2)
		self.l_rgbv = self.l_rgbv.mean()

		self.l_classifier = (cat_dist[torch.arange(len(label)), :, :, label]
							 .add(1e-5).log().mul(-1)
							 .min(dim=2)[0]
							 .min(dim=1)[0]).mean()

		self.l_kldiv = kl_divergence(prior_means, prior_logvs, posterior_means, posterior_logvs)

		self.accuracy = ((cat_dist
						  .max(dim=1)[0]
						  .max(dim=1)[0]
						  .max(dim=1)) == label).float().mean() * 100

		return sum(map(lambda l, w: l * w, [self.l_rgbv, self.l_classifier, self.l_kldiv], weights))


	def item(self):
		"""Return a 5-tuple consisting of all criteria except accuracy."""
		return self.l_rgbv.item(),	self.l_classifier.item(), self.l_kldiv.item()