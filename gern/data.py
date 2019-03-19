import torch, torchvision, os, json, cv2
import numpy as np
from PIL import Image
from collections import namedtuple
from functools import reduce


def _default_transforms_(obj, cuda):
	if cuda is None:
		device = 'cpu'
	else:
		cuda = str(cuda)
		assert cuda.isdigit()
		device = 'cuda:{}'.format(cuda)

	itransform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
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


def _rebase_(root, serial, filename):
	name = filename.split(os.path.sep)[-1]
	return os.path.join(root, serial, filename)


def _collate_(samples):
	# samples: [(C, Q, L), (C, Q, L), ...]
	C, Q, L = zip(*samples)
	# C: [(Xc, Mc, Kc, Vc), (Xc, Mc, Kc, Vc), ...]
	Xc, Mc, Kc, Vc = zip(*C)
	Xq, Mq, Kq, Vq = zip(*Q)

	# Determine shorted time steps.
	tc = reduce(lambda x, y: min(x, y), [x.size(0) for x in Xc])
	tq = reduce(lambda x, y: min(x, y), [x.size(0) for x in Xq])

	# Crop time series
	Xc = torch.stack([x[-tc:] for x in Xc], dim=0)
	Mc = torch.stack([m[-tc:] for m in Mc], dim=0)
	Kc = torch.stack([k[-tc:] for k in Kc], dim=0)
	Vc = torch.stack([v[-tc:] for v in Vc], dim=0)

	Xq = torch.stack([x[:tq] for x in Xq], dim=0)
	Mq = torch.stack([m[:tq] for m in Mq], dim=0)
	Kq = torch.stack([k[:tq] for k in Kq], dim=0)
	Vq = torch.stack([v[:tq] for v in Vq], dim=0)	

	L = torch.stack(L, dim=0)

	return (Xc, Mc, Kc, Vc), (Xq, Mq, Kq, Vq), L



class GernDataset(torch.utils.data.Dataset):
	def __init__(self, rootdir, manifest='manifest.json', cuda=None):
		self.rootdir = os.path.abspath(rootdir)
		self.manifest = manifest
		self.data = sorted(os.listdir(self.rootdir))
		_default_transforms_(self, cuda)

		assert os.path.exists(self.rootdir)
		assert len(self.data) > 0

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		serial = self.data[index]  # 8-digit folder name
		manifest_name = os.path.join(self.rootdir, serial, self.manifest)

		with open(manifest_name, 'r') as jr:
			manifest = json.load(jr)

		# --- conditionals
		Xc = [Image.open(_rebase_(self.rootdir, serial, f)) for f in manifest['visuals.condition']]
		Mc = [Image.open(_rebase_(self.rootdir, serial, f)) for f in manifest['heatmaps.condition']]
		Kc = [Image.open(_rebase_(self.rootdir, serial, f)) for f in manifest['skeletons.condition']]
		Vc = manifest['pov.readings.condition']

		Nc = manifest['num.pre'] + manifest['num.post']
		assert len(Xc) == Nc 
		assert len(Mc) == Nc
		assert len(Kc) == Nc
		assert len(Vc) == Nc

		# . transformation
		Xc = torch.stack([self.itransform(xc) for xc in Xc], dim=0)
		Mc = torch.stack([self.itransform(mc) for mc in Mc], dim=0)
		Kc = torch.stack([self.itransform(kc) for kc in Kc], dim=0)
		Vc = self.vtransform(Vc)

		# --- queries
		Xq = [Image.open(_rebase_(self.rootdir, serial, f)) for f in manifest['visuals.rewind']]
		Mq = [Image.open(_rebase_(self.rootdir, serial, f)) for f in manifest['heatmaps.rewind']]
		Kq = [Image.open(_rebase_(self.rootdir, serial, f)) for f in manifest['skeletons.rewind']]
		Vq = manifest['pov.readings.rewind']

		Nq = manifest['num.rewind']
		assert len(Xq) == Nq
		assert len(Mq) == Nq
		assert len(Kq) == Nq
		assert len(Vq) == Nq

		# . transformation
		Xq = torch.stack([self.itransform(xq) for xq in Xq], dim=0)
		Mq = torch.stack([self.itransform(mq) for mq in Mq], dim=0)
		Kq = torch.stack([self.itransform(kq) for kq in Kq], dim=0)
		Vq = self.vtransform(Vq)

		# --- activity label
		Lq = self.ltransform(manifest['label'])

		return (Xc, Mc, Kc, Vc), (Xq, Mq, Kq, Vq), Lq


class GernSampler(torch.utils.data.Sampler):
	def __init__(self, data_source, num_samples):
		self.max_n = len(data_source)
		self.n_samples = num_samples

		assert self.n_samples < self.max_n

	def __iter__(self):
		yield from np.random.randint(0, self.max_n, self.n_samples)

	def __len__(self):
		return self.n_samples



class GernDataLoader(torch.utils.data.DataLoader):
	def __init__(self, rootdir, cuda=None, subset_size=64, batch_size=8, drop_last=True, **kwargs):
		dataset = GernDataset(rootdir, cuda=cuda)
		sampler = GernSampler(dataset, subset_size)
		batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=drop_last)

		super(GernDataLoader, self).__init__(dataset, batch_sampler=batch_sampler, collate_fn=_collate_, **kwargs)
	



