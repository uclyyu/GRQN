import os, argparse, json, re, pickle, shutil
import numpy as np
import torch
import pandas as pd
from gern.gern import GeRN
from gern.data import GernDataLoader
from gern.criteria import GernCriterion
from loguru import logger
from glob import glob

# Dataframe
#	{epoch}
# 	{node}
# 	{rank}
# 	{run}
# 	{data source} : train|test|test.2
# 	{data subset size}
# 	{ar-steps} : 7, 16, 32, 64, 128

def sample2gpu(sample):
	K, Q, L = sample
	K = [k.cuda() for k in K]
	Q = [q.cuda() for q in Q]
	L = L.cuda()
	return K, Q, L

def slicetime(sample, slc=slice(0, 1)):
	K, Q, L = sample
	K = [k[:, slc] for k in K]
	Q = [q[:, slc] for q in Q]
	return K, Q, L

def mseloss(output, target):
	return (output - target).pow(2).mean()

def extract_epoch(filename):
	return int(re.match('.*_(\d{8})\.pth', filename).group(1))


class Report:
	def __init__(self, criterion):
		self.criterion = criterion
		self.reset()

	def add(self, output, target):
		lrgbv = mseloss(output.rgbv, target.rgbv).item()
		lheat = self.criterion.lfcn_heatmap(output, target).item()
		lclassify = self.criterion.lfcn_classifier(output, target).item()
		laggregate = self.criterion.lfcn_aggregate(output).item()
		accuracy = self.criterion.accu_classifier(output, target).item()
		crit = [lrgbv, lheat, lclassify, laggregate, accuracy]

		self.sum_loss = list(map(lambda i, j: i + j, self.sum_loss, crit))
		self.N += 1

	def report(self):
		return list(map(lambda i: i / self.N, self.sum_loss))

	def reset(self):
		self.N = 0
		self.sum_loss = [0., 0., 0., 0., 0.]


if __name__ == '__main__':
	col_hpar = ['epoch', 'node', 'rank', 'run', 'data-source', 'data-subset-size', 'ar-steps', 'repeat']
	col_loss = ['l-rgbv', 'l-heat', 'l-classify', 'l-aggregate', 'accuracy']

	if os.path.isfile('validate.pkl'):
		shutil.copyfile('validate.pkl', 'validate.pkl-bak')
		with open('validate.pkl', 'rb') as pkl:
			df = pickle.load(pkl)
	else:
		df = pd.DataFrame(columns=col_hpar + col_loss)

	ckpt_rootdir = '/home/yen/data/gern/results/checkpoints'
	ar_steps = [64, 7]
	subset_size = 1000
	batch_size = 50
	nrepeat = 1
	nrank = 2
	nrun = 1
	loc = 0

	dataloaders = {
		'train': GernDataLoader('/home/yen/data/gern/phase/train/samples', subset_size=subset_size, batch_size=batch_size),
		'test.1': GernDataLoader('/home/yen/data/gern/phase/test/samples', subset_size=subset_size, batch_size=batch_size),
		'test.2': GernDataLoader('/home/yen/data/gern/phase/test/samples.2', subset_size=subset_size, batch_size=batch_size)
		}

	network = GeRN().eval().cuda()
	report = Report(GernCriterion().cuda())

	for repeat in range(nrepeat):
		logger.info('On repeat #{repeat}', repeat=repeat)
		# gather checkpoint information
		for rank in range(nrank):
			rank_str = os.path.join('rank', '{:02d}'.format(rank))
			for run in range(nrun):
				run_str = os.path.join('run', '{:03d}'.format(run))
				ckptdir = os.path.join(ckpt_rootdir, rank_str, run_str)

				# loop over checkpoints in order
				for ckpt in reversed(sorted(glob(os.path.join(ckptdir, '*.pth')))):
					logger.info('\tLoad checkpoint {ckpt}.', ckpt=ckpt)
					network.load_state_dict(torch.load(ckpt))
					epoch = extract_epoch(ckpt)
					# loop over number of gaussian factors
					for astep in ar_steps:
						logger.info('\t\tSet asteps={astep}', astep=astep)
						# loop over dataloaders
						for datk, datl in dataloaders.items():
							# check whether we need to process the samples
							findrow = {
								'epoch': [epoch], 'node': [0], 'rank': [rank], 'run': [run],
								'data-source': [datk], 'data-subset-size': [subset_size], 
								'ar-steps': [astep], 'repeat': [repeat]}
							if not df.isin(findrow)[col_hpar].all(axis=1).any():
								logger.info('\t\t\tProcessing loader {loader}', loader=datk)
								report.reset()
								for sample in datl:
									with torch.set_grad_enabled(False):
										(cnd_x, cnd_m, cnd_k, cnd_v), (qry_x, qry_m, qry_k, qry_v), lab = sample2gpu(slicetime(sample))
										output = network.predict(cnd_x, cnd_m, cnd_k, cnd_v, qry_v, asteps=astep)
										target = network.make_target(qry_x, qry_m, qry_k, qry_v, lab)
										report.add(output, target)

								# push dataframe entry
								df.loc[loc] = [epoch, 0, rank, run, datk, subset_size, astep, repeat] + report.report()
							else:
								logger.info('\t\t\tSkipping')
							loc += 1

					# save after every checkpoint
					df.to_pickle('validate.pkl')