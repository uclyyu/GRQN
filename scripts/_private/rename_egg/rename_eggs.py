import re, shutil, argparse, os, csv, json, random, subprocess, pandas, signal, time
from glob import glob
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict
from panda3d.core import NodePath, loadPrcFileData
loadPrcFileData('', 'window-type offscreen')
loadPrcFileData('', 'audio-library-name null')
from direct.actor.Actor import Actor 
from direct.showbase.ShowBase import ShowBase, WindowProperties, FrameBufferProperties, GraphicsPipe, GraphicsPipeSelection

# Lxx.Ryyyy.egg
# Lxx.Ryyyy.Azzzz.egg

_COLUMNS_ = [
	'group', 
	'activity-name', 'activity-id', 'activity-descrip', 'activity-item', 
	'rig-name', 'rig-id', 
	'animation-name', 'animation-id', 'animation-numframe', 
	'file-rig', 'file-animation']

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


class ActorLoader(ShowBase):
	"""The SceneManager will import two models: one for the static 
	environment layout, the other for an animated human animation. """
	def __init__(self, size=(256, 256), zNear=0.1, zFar=1000.0, fov=70.0, showPosition=False):
		super().__init__()

		self.__dict__.update(size=size, zNear=zNear, zFar=zFar, fov=fov, showPosition=showPosition)

		self.disableMouse()

		# --- Configure offsceen properties
		flags = GraphicsPipe.BFFbPropsOptional | GraphicsPipe.BFRefuseWindow
		wprop = WindowProperties(WindowProperties.getDefault())
		wprop.setSize(size[0], size[1])
		fprop = FrameBufferProperties(FrameBufferProperties.getDefault())
		fprop.setRgbColor(1)
		fprop.setColorBits(24)
		fprop.setAlphaBits(8)
		fprop.setDepthBits(1)
		self.pipe = GraphicsPipeSelection.getGlobalPtr().makeDefaultPipe()
		self.win = self.graphicsEngine.makeOutput(
			pipe=self.pipe, name='RGB buffer', sort=0, 
			fb_prop=fprop, win_prop=wprop, flags=flags)
		self.displayRegion = self.win.makeDisplayRegion()
		self.displayRegion.setCamera(self.cam)

		self.actor = None

	def swapActor(self, actor, animation):
		if isinstance(self.actor, NodePath):
			self.actor.detachNode()
			self.actor.cleanup()
		self.actor = Actor(actor, {'act': animation})
		self.actor.reparentTo(self.render)

		return self.actor.getNumFrames('act')
	

class KeyCollisionError(Exception):
	pass


class AnimationData(object):
	def __init__(self, source, renamed):
		self.source = source[1:]
		self.renamed = int(renamed)

	def __repr__(self):
		return 'A{:04d}'.format(self.renamed)


class Animation(object):
	def __init__(self):
		self._animation_data = OrderedDict()

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
		self.source = source[1:]
		self.renamed = int(renamed)
		self.animation = Animation()

	def __repr__(self):
		return 'R{:04d}'.format(self.renamed)


class Rig(object):
	def __init__(self):
		self._rig_data = OrderedDict()

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
		return 'L{:04d}'.format(self.renamed)


class Label(object):
	def __init__(self):
		self._label_data = OrderedDict()

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

	files = sorted(glob(path_pattern))
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


def csvify(labels, common_path, outname):
	alo = ActorLoader()
	df = pandas.DataFrame(columns=_COLUMNS_)
	count = 0
	num = 0

	# signal.signal(signal.SIGALRM, handler)
	for ld in tqdm(labels):
		for rd in ld.rig:
			for ad in rd.animation:

				if rd.source[:4] == 'man0':
					cat = 'test'
				else:
					cat = 'train'

				rf = os.path.join(common_path, rd.source)
				af = os.path.join(common_path, ad.source)
				print('processing... {}'.format(af))
	
				num = alo.swapActor(rf, af)
				print(num)
				df.loc[count] = [
					cat,
					ld.source, ld.renamed, ld.activity_string, ld.object_string,
					rd.source, rd.renamed,
					ad.source, ad.renamed, num,
					dot_join(ld, rd, 'egg'), dot_join(ld, rd, ad, 'egg')
				]
				count += 1


	df.to_csv(outname, encoding='utf-8')


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
			category = row['group']
			source_rig_file = os.path.sep.join([source_path, row['rig-name']])
			target_rig_file = os.path.join(target_path, category, row['file-rig'])
			source_anim_file = os.path.sep.join([source_path, row['animation-name']])
			target_anim_file = os.path.join(target_path, category, row['file-animation'])

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
	outname = 'labels.csv'
	labels, commonpath = distill('/home/yen/data/gern/egg_human/original/**/*.egg')
	csvify(labels, commonpath, outname)

	copyfile('labels.csv', commonpath, '/home/yen/data/gern/egg_human')	
	copytexture('/home/yen/data/gern/egg_human/original/**/tex', '/home/yen/data/gern/egg_human/train')
	copytexture('/home/yen/data/gern/egg_human/original/**/tex', '/home/yen/data/gern/egg_human/test')
