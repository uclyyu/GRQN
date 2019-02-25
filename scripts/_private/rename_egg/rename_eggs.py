import re, shutil, argparse, os, csv, json, random, subprocess
from glob import glob
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

# Lxx.Ryyyy.egg
# Lxx.Ryyyy.Azzzz.egg

_LABEL_MAPPING_ = {
		# txt_label: [num_label, 'activity', 'item']
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

_PATTERN_ACTIVITY_ = 'bt|em|mj|mt[12]|pm[12]|rb|ts[12]|us|wd|wf|wtv'

_PATTERN_IGNORE_ = 'sp|wk'


# regex pattern for valid activity label (leaving out sp and wk)
_PATTERN_SUFFIX_ = (
	'(?P<suffix>'
		'-?'
		'(?P<activity>{})?'
		'(?P<ignore>{})?'
		'({})?'
	')'
	).format(_PATTERN_ACTIVITY_, _PATTERN_IGNORE_, '\d{1,3}')

# regex pattern for file prefix (man/men/woman/women)
_PATTERN_PREFIX_ = (
	'(?P<prefix>'
		'\/(wo)?m(a|e)n[0-9]\/'
		'(wo)?m(a|e)n'
		'{}'
		'({})?'
		'({})?'
		'\d?'
	')'
	).format('\d{1,2}', _PATTERN_ACTIVITY_, _PATTERN_IGNORE_)

_PATTERN_EXTENSION_ = '(?P<extension>\.egg)'

_PATTERN_FULL_ = (
	'(?P<filename>.*'
	'{}{}{}'
	')').format(_PATTERN_PREFIX_, _PATTERN_SUFFIX_, _PATTERN_EXTENSION_)


class KeyCollisionError(Exception):
	pass


class AnimationData(object):
	def __init__(self, source, renamed):
		self.source = source
		self.renamed = int(renamed)

	def __repr__(self):
		return 'A{:04d}'.format(self.renamed)


class Animation(object):
	def __init__(self):
		self._animation_data = dict()

	def __getitem__(self, key):
		return self._animation_data[key]

	def __iter__(self):
		for k in self._animation_data.keys():
			yield self._animation_data[k]

	def __len__(self):
		return len(self._animation_data)

	def size(self):
		return len(self)

	def new(self, key):
		if key in self._animation_data.keys():
			raise KeyCollisionError
		else:
			new = AnimationData(key, self.size())
			self._animation_data[key] = new
			return new


class RigData(object):
	def __init__(self, source, renamed):
		self.source = source
		self.renamed = int(renamed)
		self.animation = Animation()

	def __repr__(self):
		return 'R{:04d}'.format(self.renamed)


class Rig(object):
	def __init__(self):
		self._rig_data = dict()

	def __getitem__(self, key):
		return self._rig_data[key]

	def __iter__(self):
		for k in self._rig_data.keys():
			yield self._rig_data[k]

	def __len__(self):
		return len(self._rig_data)

	def size(self):
		return len(self)

	def new(self, key):
		if key in self._rig_data.keys():
			raise KeyCollisionError
		else:
			new = RigData(key, self.size())
			self._rig_data[key] = new
			return new


class LabelData(object):
	def __init__(self, key, num_label, activity, item):
		self.source = key
		self.renamed = int(num_label)
		self.activity_string = str(activity)
		self.object_string = str(item)
		self.rig = Rig()

	def __repr__(self):
		return 'L{:02d}'.format(self.renamed)


class Label(object):
	def __init__(self):
		self._label_data = dict()

	def __iter__(self):
		for k in self._label_data.keys():
			yield self._label_data[k]

	def __getitem__(self, key):
		return self._label_data[key]

	def __len__(self):
		return len(self._label_data)

	def size(self):
		return len(self)

	def new(self, key):
		if key in self._label_data.keys():
			raise KeyCollisionError
		else:
			# print(key)
			num_label, activity, item = _LABEL_MAPPING_[key]
			new = LabelData(key, num_label, activity, item)
			self._label_data[key] = new
			return  new


def distill(path_pattern):
	labels = Label()

	files = glob(path_pattern)
	commonpath = os.path.commonpath(files)

	for file in files:
		name = file.split(commonpath)[-1]

		mat = re.match(_PATTERN_FULL_, name)
		if mat is None:
			print('mat is None --- ', name)
			raise 

		if mat.group('ignore'):
			# print('--- Skipping: ', name)
			continue

		if mat.group('activity') is None:
			# print('--- Rig: ', name)
			continue

		activity = mat.group('activity')
		suffix = mat.group('suffix')

		try:
			ld = labels[activity]
		except KeyError:
			ld = labels.new(activity)

		# prefix = mat.group('prefix').split(commonpath)[-1]
		prefix = mat.group('prefix')
		rkey = prefix + mat.group('extension')
		try:
			rd = ld.rig[rkey]
		except KeyError:
			rd = ld.rig.new(rkey)

		if suffix:
			akey = mat.group('filename')
			try:
				ad = rd.animation[akey]
			except KeyError:
				ad = rd.animation.new(akey)

	return labels, commonpath


def dot_join(*args):
	return '.'.join(map(str, args))


def jsonify(labels, commonpath, output_name='labels.json'):
	json_dict = {'label_size': labels.size(), 'top_path': commonpath, 'labels': {}, 'rig_count': 0, 'anim_count': 0}
	for ld in labels:
		ld_dict = {'rig_size': ld.rig.size(), 'rigs': {}, 'rig_count': 0, 'anim_count': 0}

		for rd in ld.rig:
			rd_dict = {'animation_size': rd.animation.size(), 'animations': {}, 'anim_count': 0}
			
			for ad in rd.animation:
				json_dict['anim_count'] += 1
				ld_dict['anim_count'] += 1
				rd_dict['anim_count'] += 1
				rd_dict['animations'].update({ad.source: {'target': dot_join(ld, rd, ad, 'egg')}})

			json_dict['rig_count'] += 1
			ld_dict['rig_count'] += 1
			ld_dict['rigs'].update({rd.source: {'target': dot_join(ld, rd, 'egg'), 'data': rd_dict}})
		json_dict['labels'].update({ld.source: {'target': str(ld), 'data': ld_dict}})

		print('{}({:3s}) --- R#{}  A#{}'.format(str(ld), ld.source, ld_dict['rig_count'], ld_dict['anim_count']))

	with open(output_name, 'w') as jw:
		json.dump(json_dict, jw)

	return json_dict
		

def csvify(labels, output_name='labels.csv'):
	entry = OrderedDict([
		('label-txt', None), ('label-num', None), 
		('rig', None), ('rig-rename', None), 
		('anim', None), ('anim-rename', None), 
		('category', None)	
		])
	count = {'L{:02d}'.format(k): 0 for k in range(13)}

	with open(output_name, 'w') as cw:
		writer = csv.DictWriter(cw, entry.keys())
		writer.writeheader()

		for ld in labels:
			for rd in ld.rig:
				for ad in rd.animation:
					if count[str(ld)] >= 45:
						cat = 'test'
					else:
						cat = 'train'
					entry.update({
						'label-txt': ld.source,
						'label-num': str(ld),
						'rig': rd.source,
						'rig-rename': dot_join(ld, rd, 'egg'),
						'anim': ad.source,
						'anim-rename': dot_join(ld, rd, ad, 'egg'),
						'category': cat
						})
					count[str(ld)] += 1
					writer.writerow(entry)
		print(count)


def copyfile(csvfile, source_path, target_path, usebam=True):
	if usebam:
		def lazycopy(source, target):
			if not os.path.exists(target):
				subprocess.call('egg2bam -o {} {}'.format(target, source).split(' '))
			else:
				print('Lazily skipped {}'.format(target))
	else:
		def lazycopy(source, target):
			if not os.path.exists(target):
				shutil.copy(source, target)
			else:
				print('Lazily skipped {}'.format(target))

	with open(csvfile, 'r') as cr:
		reader = csv.DictReader(cr)
		for row in reader:
			category = row['category']
			source_rig_file = os.path.sep.join([source_path, row['rig']])
			target_rig_file = os.path.join(target_path, category, row['rig-rename'])
			source_anim_file = os.path.sep.join([source_path, row['anim']])
			target_anim_file = os.path.join(target_path, category, row['anim-rename'])

			if usebam:
				target_rig_file = target_rig_file.replace('.egg', '.bam')
				target_anim_file = target_anim_file.replace('.egg', '.bam')
			
			lazycopy(source_rig_file, target_rig_file)
			lazycopy(source_anim_file, target_anim_file)


def copytexture(source_path_pattern, target_path):
	for texdir in tqdm(glob(source_path_pattern)):
		for tex_file in glob(os.path.join(texdir, '*')):
			target_dir = os.path.join(target_path, 'tex')
			filename = tex_file.split(os.path.sep)[-1]
			target_file = os.path.join(target_dir, filename)

			if not os.path.exists(target_dir):
				os.mkdir(target_dir)

			if not os.path.exists(target_file):
				shutil.copy(tex_file, target_file)


if __name__ == '__main__':

	labels, commonpath = distill('../../../../../data/gern/egg_human/original/**/*.egg')
	jsonify(labels, commonpath)
	csvify(labels)

	copyfile('labels.csv', commonpath, '../../../../../data/gern/egg_human')	
	copytexture('../../../../../data/gern/egg_human/original/**/tex', '../../../../../data/gern/egg_human/train')
	copytexture('../../../../../data/gern/egg_human/original/**/tex', '../../../../../data/gern/egg_human/test')
