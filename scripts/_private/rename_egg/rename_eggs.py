import re, shutil, argparse, os, csv, json, random
from glob import glob
from tqdm import tqdm
from datetime import datetime

if __name__ == '__main__':
	rigg_i = 0
	anim_i = 0
	n_train_rigg = 100
	n_train_anim = 0
	files = glob('./**/*.egg')
	texs = glob('./**/tex')
	pattern = '(?P<rigg>(wo)?m(a|e)n\d{1,2}[a-z]{1,3}\d?\.egg)|(?P<anim>.*-.*egg)'
	record = {'source': None, 'dest': None}
	report = {'num_rigg': 0, 'num_anim': 0, 'timestamp': None,
			  'num_rigg_train': n_train_rigg, 'num_rigg_test': 0, 
			  'num_anim_train': n_train_anim, 'num_anim_test': 0}

	# --- Make copies of .egg files
	with open('egg_manifest.csv', 'w') as csvfile:
		writer = csv.DictWriter(csvfile, record.keys())

		random.shuffle(files)
		for file in tqdm(files):
			name = file.split(os.path.sep)[-1]
			m = re.match(pattern, name)
			if m:
				if m.group('rigg') is not None:
					rname = name
					new_rname = 'rigg_{:04d}.egg'.format(rigg_i)
					record.update({'source': name, 'dest': new_rname})
					writer.writerow(record)

					if rigg_i < n_train_rigg:
						shutil.copy(file, os.path.sep.join(['../train', new_rname]))
					else:
						shutil.copy(file, os.path.sep.join(['../test', new_rname]))

					anim_files = glob(file.split('.egg')[-2] + '-*.egg')
					anim_j = 0
					for afile in anim_files:
						aname = afile.split(os.path.sep)[-1]
						new_aname = 'anim_{:04d}_{:04d}.egg'.format(rigg_i, anim_j)
						record.update({'source': aname, 'dest': new_aname})
						writer.writerow(record)
						
						if rigg_i < n_train_rigg:
							shutil.copy(afile, os.path.sep.join(['../train', new_aname]))
							n_train_anim += 1
						else:
							shutil.copy(afile, os.path.sep.join(['../test', new_aname]))
						anim_i += 1
						anim_j += 1

					rigg_i += 1

			else:
				print('Cannot match: {}...skipping'.format(name))

	# --- Report numbers
	with open('egg_report.json', 'w') as jw:
		report.update({'num_rigg': rigg_i, 'num_anim': anim_i})
		report.update({'num_rigg_test': rigg_i - n_train_rigg, 
					   'num_anim_train': n_train_anim,
					   'num_anim_test': anim_i - n_train_anim})
		report.update({'timestamp': str(datetime.now())})
		json.dump(report, jw)

	# --- Make copies of texture images
	for tex in tqdm(texs):
		for tex_file in glob(os.path.sep.join([tex, '*'])):
			dest_train = '../train/tex'
			dest_test = '../test/tex'

			if not os.path.exists(dest_train):
				os.mkdir(dest_train)
			if not os.path.exists(dest_test):
				os.mkdir(dest_test)

			shutil.copy(tex_file, dest_train)
			shutil.copy(tex_file, dest_test)


