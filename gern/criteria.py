from torchvision import models as visionmodels
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

	def forward(self, pred, targ):
		# exhaust intermediate outputs and evaluate mse loss.
		inp = torch.cat([pred, targ], dim=0)
		self.vggf(inp)

		running_loss = 0.
		while len(self.hidden_outputs) > 0:
			running_loss += self.mseloss(*torch.chunk(self.hidden_outputs.pop(), 2, dim=0))

		return running_loss / (self.N * pred.size(0))


class GernCriterion(nn.Module):
	# Need to track the following loss:
	# - Perceptual loss
	# 	- This is to replace the default L2 loss in pixel space. 
	# 	- Comparisons are made between predicted "rewind" and target "rewind".
	# - Binary cross entropy
	# 	- This will be the loss for predicted heatmaps as they will fall within the [0, 1] value range. 
	# 	  And we might as well treat it as probabilities. 
	# - Categorical cross entropy
	# 	- Clearly this is for our classifier which uses softmax output layer.
	def __init__(self):
		# loss functions
		self.fcn_percept = PerceptualLoss()
		self.fcn_bce = None
		self.fcn_cel = None

		# states
		self.l_percept = None
		self.l_bce = None
		self.l_cel = None

	def forward(self, gern_output):
		pass

	def sum(self):
		return 