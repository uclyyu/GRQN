#!/usr/bin/env python
from PIL import Image
from glob import glob
from tqdm import tqdm

for phase in ['train', 'test']:
	patt = '/home/yen/data/gern/samples_{}/dataset/**/visual-cond-000.jpg'.format(phase)
	captures = []
	for file in tqdm(sorted(glob(patt))):
		img = Image.open(file)

		if img.size != (256, 256):
			print(file)
			# captures.append(file)

	# file_name = '{}_wrong_size.csv'.format(phase)
	# with open(file_name, 'w') as file_handle:
	# 	for row in sorted(captures):
	# 		file_handle.write(row + '\n')


