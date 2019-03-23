#!/usr/bin/env python
import cv2
from glob import glob
from tqdm import tqdm

for phase in ['train', 'test']:
	patt = '/home/yen/data/gern/samples_{}/dataset/**/visual-cond-000.jpg'.format(phase)
	for file in tqdm(sorted(glob(patt))):
		img = cv2.imread(file)
		if (img == 0).all():
			print('Blank image: {}'.format(file))
