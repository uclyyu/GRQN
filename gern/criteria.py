from torchvision import models as visionmodels
import torch
import torch.nn as nn
from .utils import count_parameters

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