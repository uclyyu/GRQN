import torch, torchvision, os, cv2, pandas, random
import numpy as np


class GernDataset(torch.utils.data.Dataset):
	def __init__(self, rootdir, window_size=35, min_k=1, max_k=7, min_q=3, max_q=9, transforms=None, manifest='manifest.csv'):
		"""Dataset
		
		Args:
		    rootdir (str): Relative or absolute path to the folder holding all the data.
		    min_k (int, optional): Minimum number of context viewpoints
		    max_k (int, optional): Maximum number of context viewpoints
		    min_q (int, optional): Minimum number of query viewpoints
		    max_q (int, optional): Maximum number of query viewpoints
		    transforms (None, optional): torchvision.transforms object
		    manifest (str, optional): Filename of data manifest (default='manifest.csv')
		"""
		self.rootdir = os.path.abspath(rootdir)
		self.manifest = manifest
		assert os.path.exists(self.rootdir)

		self.data = os.listdir(self.rootdir)
		assert len(self.data) > 0

		# _default_transforms_(self)

		self.transforms = torchvision.transforms.ToTensor() if transforms is None else transforms
		self.window_size = window_size
		self.min_k = min_k
		self.max_k = max_k
		self.min_q = min_q
		self.max_q = max_q
		self.num_k = None
		self.num_q = None

		self.renew_dataset_state()

	def renew_dataset_state(self):
		self.num_k = np.random.randint(self.min_k, self.max_k + 1)
		self.num_q = np.random.randint(self.min_q, self.max_q + 1)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		# Get video filenames
		serial = self.data[index]  # 8-digit folder name
		manifest_name = os.path.join(self.rootdir, serial, self.manifest)
		manifest = pandas.read_csv(manifest_name)
		fp_camv = os.path.join(self.rootdir, serial, manifest.loc[0]['camv-filename'])
		fp_skel = os.path.join(self.rootdir, serial, manifest.loc[0]['skel-filename'])

		label = torch.tensor(manifest.loc[0]['activity'], dtype=torch.long)  # training label

		# Each video consists of concatenated clips of equal length
		n_phase = len(manifest)
		clip_length = manifest.loc[0]['length']

		# Slice time window		
		inclip_indices = np.arange(0, clip_length)
		window_start = np.random.randint(0, len(inclip_indices) - self.window_size + 1)
		inclip_window = inclip_indices[window_start:(window_start + self.window_size)]

		# Contextual and query viewpoints
		phases = set(range(n_phase))
		k_phase = np.array(random.sample(phases, self.num_k))
		q_phase = random.sample(phases.difference(k_phase), self.num_q)

		# Apply offsets to time window
		k_frame_offset_indices = k_phase[np.random.randint(0, self.num_k, size=self.window_size)]
		k_frame_offsets = manifest['start-frame'][k_frame_offset_indices].values
		k_offset_window = inclip_window + k_frame_offsets

		# Contextual viewpoints
		k_pose = (
			manifest['camera-pose'][k_frame_offset_indices]
			.str[1:-1]  # remove '[' and ']'
			.str.split(',')
			.apply(pandas.to_numeric, errors='raise', downcast='float')
			.tolist()
			)
		k_pose = self.transforms(np.array(k_pose)).view(self.window_size, 7, 1, 1) 

		camvcap = cv2.VideoCapture(fp_camv)
		skelcap = cv2.VideoCapture(fp_skel)

		# Contextual camera views
		kstack_camv = []
		kstack_skel = []

		for ki in k_offset_window:
			camvcap.set(cv2.CAP_PROP_POS_FRAMES, ki)
			skelcap.set(cv2.CAP_PROP_POS_FRAMES, ki)
			success, readout = camvcap.read()
			assert success
			kstack_camv.append(self.transforms(readout))
			success, readout = skelcap.read()
			assert success
			kstack_skel.append(self.transforms(readout))

		# Query viewpoints
		q_pose = (
			manifest['camera-pose'].loc[q_phase]
			.str[1:-1]
			.str.split(',')
			.apply(pandas.to_numeric, errors='raise', downcast='float')
			.tolist()
			)
		q_pose = self.transforms(np.array(q_pose)).view(self.num_q, 7, 1, 1)

		# Query camera views
		qstack_camv = []
		qstack_skel = []

		for qph in q_phase:
			q_camv = []
			q_skel = []
			for frame in inclip_window + manifest.loc[qph]['start-frame']:
				camvcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
				skelcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
				success, readout = camvcap.read()
				assert success
				q_camv.append(self.transforms(readout))
				success, readout = skelcap.read()
				assert success
				q_skel.append(self.transforms(readout))
			qstack_camv.append(torch.stack(q_camv, dim=0))
			qstack_skel.append(torch.stack(q_skel, dim=0))

		camvcap.release()
		skelcap.release()

		kX = torch.stack(kstack_camv, dim=0)
		kK = torch.stack(kstack_skel, dim=0)
		kV = k_pose
		qX = torch.stack(qstack_camv, dim=0)
		qK = torch.stack(qstack_skel, dim=0)
		qV = q_pose

		return kX, kK, kV, qX, qK, qV, label


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
	def __init__(self, rootdir, subset_size, batch_size, drop_last=True, **kwargs):
		dataset = GernDataset(rootdir)
		sampler = GernSampler(dataset, subset_size)
		batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=drop_last)

		super(GernDataLoader, self).__init__(dataset, batch_sampler=batch_sampler, **kwargs)
