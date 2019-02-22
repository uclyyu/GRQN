import re, shutil, argparse, os, csv, json, random
from glob import glob
from tqdm import tqdm
from datetime import datetime

if __name__ == '__main__':

	egg_files = glob('./**/*.egg')
	img_textures = glob('./**/tex')

	activity_mapping = {
		# id: [label, 'activity', 'item']
		'bt':  [ 0, 'Brushing teeth',        'Toothbrush'],
		'em':  [ 1, 'Eating meal',           'Knife and fork'],
		'mj':  [ 2, 'Making juice',          'Blender'],
		'mt1': [ 3, 'Making tea',            'Tea caddy'],
		'mt2': [ 4, 'Making tea',            'Kettle'],
		'pm1': [ 5, 'Preparing meal',        'Knife and vegetable'], 
		'pm2': [ 6, 'Preparing meal',        'Frying pan'],
		'rb':  [ 7, 'Reading book',          'Book'],
		'ts1': [ 8, 'Talking on smartphone', 'Smartphone (left handed)'],
		'ts2': [ 8, 'Talking on smartphone', 'Smartphone (right handed)'],
		'us':  [ 9, 'Using smartphone',      'Smartphone'],
		'wd':  [10, 'Washing dish',          'Dish and sponge'],
		'wf':  [11, 'Washing face',          'None'],
		'wtv': [12, 'Watching TV',           'Remote control'],
		'sp':  [13, 'Sleeping',              'None'],
		'wk':  [14, 'Walking',               'None']
	}

	# regex pattern for valid activity ids (leaving out sp and wk)
	pat_vac = (
		'(?P<sfx>'
			'-?'
			'(?P<apt>bt|em|mj|mt[12]|pm[12]|rb|ts[12]|us|wd|wf|wtv)?'
			'(?P<ignore>sp|wk)?'
			'(\d{1,3})?'
		')'
		)

	# regex pattern for file prefix (man/men/woman/women)
	pat_pfx = (
		'(?P<pfx>'
			'(wo)?m(a|e)n'
			'\d{1,2}'
			'([a-z]{1,3})?'
			'\d?'
		')'
		)

	pat_ext = '(?P<ext>\.egg)'


	pat_all = (
		'(?P<filename>'
		'{}{}{}'
		')').format(pat_pfx, pat_vac, pat_ext)

	rig_re_naming = {}  # map rig -> rig_re

	# counter 
	rig_re_count = {}

	json_data = {
		k: {'label': 0, 'activity': None, 'object': None, 'rig': [], 'animation': [], 'rig_re': [], 'animation_re': []}
		for k in range(15)
	}

	for egg in egg_files:
		path = egg.split(os.path.sep)[:-1]
		mat = re.match(pat_all, egg)

		assert mat, 'Regex returns None: {}'.format(egg)

		prefix = mat.group('pfx')
		label, activity, item = activity_mapping[prefix]

		is_animation = mat.group('sfx')
		if is_animation:

			rig_name = mat.group('pfx') + mat.group('ext')
			animation_name = mat.group('filename')

			try:
				rig_count = rig_re_count[label]
			except KeyError:
				rig_re_count[label] = 0
				rig_count = 0

			try:
				rig_re = rig_re_naming[rig_name]
			except KeyError:
				rig_re_naming[rig_name] = 'rigg_{:03d}_{:04d}.egg'.format(label, rig_count)
				rig_re_count[label] += 1
				rig_re = rig_re_naming[rig_name]





		json_data[label].update({'label': label, 'activity': activity, 'object': item})
		json_data[label]['rig'].append(rig_name)
		json_data[label]['rig_re'].append(rig_re)
		json_data[label]['animation'].append()




	# record = {'source': None, 'dest': None}
	# report = {'num_rigg': 0, 'num_anim': 0, 'timestamp': None,
	# 		  'num_rigg_train': n_train_rigg, 'num_rigg_test': 0, 
	# 		  'num_anim_train': n_train_anim, 'num_anim_test': 0}

	# # --- Make copies of .egg files
	# with open('egg_manifest.csv', 'w') as csvfile:
	# 	writer = csv.DictWriter(csvfile, record.keys())

	# 	random.shuffle(files)
	# 	for file in tqdm(files):
	# 		name = file.split(os.path.sep)[-1]
	# 		m = re.match(pattern, name)
	# 		if m:
	# 			if m.group('rigg') is not None:
	# 				rname = name
	# 				new_rname = 'rigg_{:04d}.egg'.format(rigg_i)
	# 				record.update({'source': name, 'dest': new_rname})
	# 				writer.writerow(record)

	# 				if rigg_i < n_train_rigg:
	# 					shutil.copy(file, os.path.sep.join(['../train', new_rname]))
	# 				else:
	# 					shutil.copy(file, os.path.sep.join(['../test', new_rname]))

	# 				anim_files = glob(file.split('.egg')[-2] + '-*.egg')
	# 				anim_j = 0
	# 				for afile in anim_files:
	# 					aname = afile.split(os.path.sep)[-1]
	# 					new_aname = 'anim_{:04d}_{:04d}.egg'.format(rigg_i, anim_j)
	# 					record.update({'source': aname, 'dest': new_aname})
	# 					writer.writerow(record)
						
	# 					if rigg_i < n_train_rigg:
	# 						shutil.copy(afile, os.path.sep.join(['../train', new_aname]))
	# 						n_train_anim += 1
	# 					else:
	# 						shutil.copy(afile, os.path.sep.join(['../test', new_aname]))
	# 					anim_i += 1
	# 					anim_j += 1

	# 				rigg_i += 1

	# 		else:
	# 			print('Cannot match: {}...skipping'.format(name))

	# --- Report numbers
	# with open('egg_report.json', 'w') as jw:
	# 	report.update({'num_rigg': rigg_i, 'num_anim': anim_i})
	# 	report.update({'num_rigg_test': rigg_i - n_train_rigg, 
	# 				   'num_anim_train': n_train_anim,
	# 				   'num_anim_test': anim_i - n_train_anim})
	# 	report.update({'timestamp': str(datetime.now())})
	# 	json.dump(report, jw)

	# # --- Make copies of texture images
	# for tex in tqdm(texs):
	# 	for tex_file in glob(os.path.sep.join([tex, '*'])):
	# 		dest_train = '../train/tex'
	# 		dest_test = '../test/tex'

	# 		if not os.path.exists(dest_train):
	# 			os.mkdir(dest_train)
	# 		if not os.path.exists(dest_test):
	# 			os.mkdir(dest_test)

	# 		shutil.copy(tex_file, dest_train)
	# 		shutil.copy(tex_file, dest_test)


