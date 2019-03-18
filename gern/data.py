import torch, torchvision, os, json, cv2
from PIL import Image
from collections import namedtuple
import numpy as np



def _default_transforms_(obj, cuda):
	if cuda is None:
		device = 'cpu'
	else:
		device = 'cuda:{}'.format(cuda)

	itransform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Lambda(lambda x: torch.stack(x, dim=0)),
		torchvision.transforms.Lambda(lambda x: x if cuda is None else x.cuda(device))
		])
	vtransform = torchvision.transforms.Compose([
		torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32, device=device)),
		torchvision.transforms.Lambda(lambda x: x.unsqueeze(2).unsqueeze(3))
		])
	ltransform = torchvision.transforms.Compose([
		torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long, device=device)),
		torchvision.transforms.Lambda(lambda x: x.view(-1))
		])
	
	setattr(obj, 'itransform', itransform)
	setattr(obj, 'vtransform', vtransform)
	setattr(obj, 'ltransform', ltransform)

class GernDataset(torch.utils.data.Dataset):
	def __init__(self, rootdir, manifest='manifest.json', cuda=None):
		self.rootdir = os.path.abspath(rootdir)
		self.manifest = manifest
		self.data = sorted(os.listdir(self.rootdir))
		_default_transforms_(self, cuda)

		assert os.exists(self.rootdir)
		assert len(self.data) > 0

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		manifest_name = os.path.join(self.rootdir, self.data[index], self.manifest)

		with open(manifest_name, 'r') as jr:
			manifest = json.load(jr)

		# --- conditionals
		Xc = [Image.open(f) for f in manifest['visuals.condition']]
		Mc = [Image.open(f) for f in manifest['heatmaps.condition']]
		Kc = [Image.open(f) for f in manifest['skeletons.condition']]
		Vc = manifest['pov.readings.condition']

		Nc = manifest['num.pre'] + manifest['num.post']
		assert len(Xc) == Nc 
		assert len(Mc) == Nc
		assert len(Kc) == Nc
		assert len(Vc) == Nc

		# . transformation
		Xc = torch.stack([self.itransform(xc) for xc in Xc], dim=0)
		Mc = torch.stack([self.itransform(mc) for xc in Mc], dim=0)
		Kc = torch.stack([self.itransform(kc) for xc in Kc], dim=0)
		Vc = self.vtransform(Vc)

		# --- queries
		Xq = [Image.open(f) for f in manifest['visuals.rewind']]
		Mq = [Image.open(f) for f in manifest['heatmaps.rewind']]
		Kq = [Image.open(f) for f in manifest['skeletons.rewind']]
		Vq = manifest['pov.readings.condition']

		# . transformation
		Xq = torch.stack([self.itransform(xq) for xq in Xq], dim=0)
		Mq = torch.stack([self.itransform(mq) for xq in Mq], dim=0)
		Kq = torch.stack([self.itransform(kq) for xq in Kq], dim=0)
		Vq = self.vtransform(Vq)

		# --- activity label
		Lq = self.ltransform(manifest['label'])


class GernSampler(torch.utils.data.Sampler):
	def __init__(self, data_source, num_samples):
		self.max_n = len(data_source)
		self.n_samples = num_samples

		assert self.n_samples < self.max_n

	def __iter__(self):
		yield from np.random.randint(0, self.max_n, self.n_samples)

	def __len__(self):
		return self.n_samples


class GernBatchSampler(torch.utils.data.Sampler):
	def __init__(self, sampler, batch_size, drop_last=True):
		self.sampler = sampler
		self.batch_size = batch_size
		self.drop_last = drop_last
	



