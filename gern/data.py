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
		fXc = [Image.open(_rebase_(self.rootdir, serial, f)) for f in manifest['visuals.condition']]
		fMc = [Image.open(_rebase_(self.rootdir, serial, f)) for f in manifest['heatmaps.condition']]
		fKc = [Image.open(_rebase_(self.rootdir, serial, f)) for f in manifest['skeletons.condition']]
		fVc = manifest['pov.readings.condition']

		Nc = manifest['num.pre'] + manifest['num.post']
		assert len(fXc) == Nc 
		assert len(fMc) == Nc
		assert len(fKc) == Nc
		assert len(fVc) == Nc

		# . transformation
		tXc = torch.stack([self.itransform(xc) for xc in fXc], dim=0)
		tMc = torch.stack([self.mtransform(mc) for mc in fMc], dim=0)
		tKc = torch.stack([self.itransform(kc) for kc in fKc], dim=0)
		tVc = self.vtransform(fVc)

		#	close files
		for c in [fXc, fMc, fKc]:
			for f in c: 
				f.close()

		# --- queries
		fXq = [Image.open(_rebase_(self.rootdir, serial, f)) for f in manifest['visuals.rewind']]
		fMq = [Image.open(_rebase_(self.rootdir, serial, f)) for f in manifest['heatmaps.rewind']]
		fKq = [Image.open(_rebase_(self.rootdir, serial, f)) for f in manifest['skeletons.rewind']]
		fVq = manifest['pov.readings.rewind']

		Nq = manifest['num.rewind']
		assert len(fXq) == Nq
		assert len(fMq) == Nq
		assert len(fKq) == Nq
		assert len(fVq) == Nq

		# . transformation
		tXq = torch.stack([self.itransform(xq) for xq in fXq], dim=0)
		tMq = torch.stack([self.mtransform(mq) for mq in fMq], dim=0)
		tKq = torch.stack([self.itransform(kq) for kq in fKq], dim=0)
		tVq = self.vtransform(fVq)

		#	close files
		for q in [fXq, fMq, fKq]:
			for f in q: 
				f.close()

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
	



