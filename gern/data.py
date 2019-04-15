import torch, torchvision, os, json, cv2
import numpy as np
from PIL import Image
from collections import namedtuple
from functools import reduce


def _default_transforms_(obj):
	# transform rgb images
	itransform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	# transform monochromatic images
	mtransform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		])
	# transform orientation vector
	vtransform = torchvision.transforms.Compose([
		torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
		torchvision.transforms.Lambda(lambda x: x.unsqueeze(2).unsqueeze(3))
		])
	# transform target label
	ltransform = torchvision.transforms.Compose([
		torchvision.transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)),
		torchvision.transforms.Lambda(lambda x: x.view(-1))
		])
	
	setattr(obj, 'itransform', itransform)
	setattr(obj, 'mtransform', mtransform)
	setattr(obj, 'vtransform', vtransform)
	setattr(obj, 'ltransform', ltransform)

def _rebase_(root, serial, filename):
	name = filename.split(os.path.sep)[-1]
	return os.path.join(root, serial, name)

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

	L = torch.cat(L, dim=0)

	return (Xc, Mc, Kc, Vc), (Xq, Mq, Kq, Vq), L

def _load_images(manifest, root, serial, transform):
	stack = []
	for file in manifest:
		img = Image.open(_rebase_(root, serial, file))
		arr = np.array(img, dtype=np.float32)
		img.close()
		stack.append(transform(arr))

	return torch.stack(stack, dim=0)


class GernDataset(torch.utils.data.Dataset):
	def __init__(self, rootdir, manifest='manifest.json'):
		self.rootdir = os.path.abspath(rootdir)
		self.manifest = manifest
		self.data = sorted(os.listdir(self.rootdir))
		_default_transforms_(self)

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
		tXc = _load_images(manifest['visuals.condition'], self.rootdir, serial, self.itransform)
		tMc = _load_images(manifest['heatmaps.condition'], self.rootdir, serial, self.mtransform)
		tKc = _load_images(manifest['skeletons.condition'], self.rootdir, serial, self.itransform)
		tVc = self.vtransform(manifest['pov.readings.condition'])

		Nc = manifest['num.pre'] + manifest['num.post']
		assert tXc.size(0) == Nc 
		assert tMc.size(0) == Nc
		assert tKc.size(0) == Nc
		assert tVc.size(0) == Nc

		# --- queries
		tXq = _load_images(manifest['visuals.rewind'], self.rootdir, serial, self.itransform)
		tMq = _load_images(manifest['heatmaps.rewind'], self.rootdir, serial, self.mtransform)
		tKq = _load_images(manifest['skeletons.rewind'], self.rootdir, serial, self.itransform)
		tVq = self.vtransform(manifest['pov.readings.rewind'])

		Nq = manifest['num.rewind']
		assert tXq.size(0) == Nq 
		assert tMq.size(0) == Nq
		assert tKq.size(0) == Nq
		assert tVq.size(0) == Nq

		# --- activity label
		Lq = self.ltransform(manifest['label'])

		return (tXc, tMc, tKc, tVc), (tXq, tMq, tKq, tVq), Lq


class GernSampler(torch.utils.data.Sampler):
	def __init__(self, data_source, num_samples):
		self.max_n = len(data_source)
		self.n_samples = num_samples

		assert self.n_samples <= self.max_n

	def __iter__(self):
		yield from np.random.randint(0, self.max_n, self.n_samples)

	def __len__(self):
		return self.n_samples


class GernDataLoader(torch.utils.data.DataLoader):
	def __init__(self, rootdir, subset_size=64, batch_size=8, drop_last=True, **kwargs):
		dataset = GernDataset(rootdir)
		sampler = GernSampler(dataset, subset_size)
		batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=drop_last)

		super(GernDataLoader, self).__init__(dataset, batch_sampler=batch_sampler, collate_fn=_collate_, **kwargs)
