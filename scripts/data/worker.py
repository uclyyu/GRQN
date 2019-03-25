import multiprocessing as mp
import queue, sys, argparse, json, scene, os, random, shutil, glob, time
from loguru import logger


# def mp_collect(worker, mode, serial, bjson, sjson, wsize, scene_ext, model_ext, output_base):
def mp_collect(worker, serialq, args):
	import blender
	# --- blender: json information 
	bp_main = os.path.abspath(args.blender_proto_file)
	bp_flor = os.path.abspath(args.blender_texdir_floor)
	bp_wall = os.path.abspath(args.blender_texdir_wall)
	bp_blnd = os.path.abspath(args.blender_texdir_blind)
	bp_expo = os.path.abspath(args.savedir_3dscene)

	# --- scene: json information
	sp_actr = os.path.abspath(args.searchdir_actor)
	sp_anim = os.path.abspath(args.searchdir_animation)
	sp_pose = args.openpose

	# --- for blender exports
	use_bam = True if args.fileext_3dscene == '.bam' else False

	# --- List of actors
	if args.generation_mode == 'sample':
		actors = glob.glob(os.path.sep.join([sp_actr, 'L??.R????{}'.format(args.fileext_actor)]))

	# --- blender: make a new copy of blender proto file and load export utility
	savedir_proto = os.path.join(args.savedir_sample, '../blender')
	proto_name = os.path.basename(bp_main).replace('.blend', '{}.blend'.format(worker))
	if not os.path.isdir(savedir_proto):
		os.mkdir(savedir_proto)
	bp_proto = os.path.join(savedir_proto, proto_name)
	shutil.copy(bp_main, bp_proto)
	blender.addon_utils.enable('io_scene_egg')
	blender.bpy.ops.wm.open_mainfile(filepath=bp_proto)

	# --- A new instance of scene manager
	if args.generation_mode == 'sample':
		smgr = scene.SceneManager('collect', sp_pose, size=args.render_size)

	# --- Main procedures
	while True:
		try:
			job = serialq.get(timeout=3)
		except queue.Empty:
			logger.info('Worker {worker}: empty queue.', worker=worker)
			return
		else:
			if job % 2 == 0:
				logger.info('Worker {worker}: handling job {job:,} without a blind.', worker=worker, job=job)
				use_blind = False
			else:
				logger.info('Worker {worker}: handling job {job:,} using a blind.', worker=worker, job=job)
				use_blind = True

			# Sample and generate .egg scene
			if args.generation_mode == '3dscene':
				blender.sample_environment(
					job, args.generation_phase, bp_flor, bp_wall, bp_blnd, bp_expo, 
					use_blind=use_blind, use_bam=use_bam)
			elif args.generation_mode == 'sample':
				scene_file = os.path.join(bp_expo, 'scene_{:08d}{}'.format(job, args.scene_fileext))
				# Sample actor and animation
				actor, animation = _sample_actors(actors, sp_anim, args.fileext_actor)
				# Update scene manager
				job_savedir_sample = os.path.join(args.savedir_sample, '{:08d}'.format(job))
				smgr.swapScene(scene_file)
				smgr.swapActor(actor, animation)
				smgr.rebase(job, job_savedir_sample)
				smgr.step()
			else:
				logger.exception('Unknown mode to proceed: {mode}.', mode=args.generation_mode)


def _sample_actors(actors, anim_search_path, extension):
	actor, = random.sample(actors, 1)
	search_pattern = actor.split(os.path.sep)[-1].replace(extension, '.A????' + extension)
	search_path = os.path.join(anim_search_path, search_pattern)
	animations = glob.glob(search_path)
	animation, = random.sample(animations, 1)
	return actor, animation


def _update_main_dict(main, args):
	# Deprecated
	if args['wjson'] is None:
		output_dict = dict()
		for k, v in args.items():
			if k != 'wjson' and k is None:
				raise

			if k == 'bjson':
				with open(v, 'r') as jr:
					main.__dict__.update({k: json.load(jr)})
			elif k == 'sjson':
				with open(v, 'r') as jr:
					main.__dict__.update({k: json.load(jr)})
			elif k == 'wsize':
				main.__dict__.update({k: tuple(map(int, v.split('x')))})
			else:
				main.__dict__.update({k: v})

			output_dict.update({k: main.__dict__[k]})
		return output_dict
	else:
		with open(args['wjson'], 'r') as jr:
			args_json = json.load(jr)
			main.__dict__.update(args_json)
		return None


def _unpack_openpose_args(args):
	openpose_attr = dict()
	for key, val in args.__dict__.items():
		if key.startswith('openpose_'):
			new_key = key.replace('openpose_', '')
			openpose_attr.update({new_key: val})
	setattr(args, 'openpose', openpose_attr)


class UndefinedString(str):
	def __str__(self):
		return '<UNDEFINED>'
	def __repr__(self):
		return 'UndefinedString'


if __name__ == '__main__':
	main = sys.modules['__main__']

	parser = argparse.ArgumentParser(description='Multi-worker data generator.')
	parser.add_argument('--generation-phase', type=str, choices=['train', 'test'], help='Genearte {train|test} dataset.')
	parser.add_argument('--generation-mode', type=str, choices=['3dscene', 'sample'])
	parser.add_argument('--save-default-arguments', type=str, default=UndefinedString())
	parser.add_argument('--use-json', type=str, default=UndefinedString())
	parser.add_argument('--savedir-sample', type=str, default=UndefinedString(), help='Root path for generated samples.')
	parser.add_argument('--num-workers', type=int, default=1, help='Number of multiprocessing workers.')
	parser.add_argument('--from-sample', type=int, default=0, help='Genearte from sample N.')
	parser.add_argument('--chunk-size', type=int, default=1000, help='')
	parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to draw.')
	parser.add_argument('--fileext-actor', type=str,  default='.bam', choices=['.egg', '.bam'],	help='Panda3D model extension.')
	parser.add_argument('--fileext-3dscene', type=str, default='.egg', choices=['.egg', '.bam'], help='Panda3D scene extension.')
	parser.add_argument('--render-size', type=int, nargs=2, default=[256, 256], help='Render window height and width.')
	parser.add_argument('--blender-texdir-floor', type=str, default=UndefinedString())
	parser.add_argument('--blender-texdir-wall', type=str, default=UndefinedString())
	parser.add_argument('--blender-texdir-blind', type=str, default=UndefinedString())
	parser.add_argument('--blender-proto-file', type=str, default=UndefinedString())
	parser.add_argument('--savedir-3dscene', type=str, default=UndefinedString())
	parser.add_argument('--searchdir-actor', type=str, default=UndefinedString())
	parser.add_argument('--searchdir-animation', type=str, default=UndefinedString())
	parser.add_argument('--openpose-model-folder', type=str, default=UndefinedString())
	parser.add_argument('--openpose-num-gpu', type=int, default=2)
	parser.add_argument('--openpose-num-gpu-start', type=int, default=0)
	parser.add_argument('--openpose-net-resolution', type=str, default='256x256')
	parser.add_argument('--openpose-heatmaps-add-bkg', type=str, default='true')
	parser.add_argument('--openpose-heatmaps-scale', type=int, default=1)
	parser.add_argument('--openpose-disable-blending', type=str, default='true')

	# Add optional openpose arguments
	parsed, unknown = parser.parse_known_args()
	for arg in unknown:
		if arg.startswith(('--openpose',)):
			parser.add_argument(arg)

	# Parse again
	args = parser.parse_args()

	# Save arguments and return
	if os.path.isdir(args.save_default_arguments):
		arg_json = os.path.join(args.save_default_arguments, 'default_worker_arguments.json')
		delattr(args, 'save_default_arguments')
		delattr(args, 'use_json')
		with open(arg_json, 'w') as jw:
			json.dump(vars(args), jw)
			logger.info('Saved default arguments: {file}', file=arg_json)
			sys.exit(0)

	# Update arguments from json file
	if os.path.isfile(args.use_json):
		logger.info('Loading arguments from json: {file}', file=args.use_json)
		with open(args.use_json, 'r') as jr:
			json_data = json.load(jr)
			args.__dict__.update(json_data)

	# Unpack openpose arguments
	_unpack_openpose_args(args)

	# Some sanity checks
	assert not isinstance(args.savedir_sample, 		  UndefinedString)
	assert not isinstance(args.blender_texdir_floor,  UndefinedString)
	assert not isinstance(args.blender_texdir_wall,   UndefinedString)
	assert not isinstance(args.blender_texdir_blind,  UndefinedString)
	assert not isinstance(args.blender_proto_file,	  UndefinedString)
	assert not isinstance(args.savedir_3dscene, 	  UndefinedString)
	assert not isinstance(args.searchdir_actor,		  UndefinedString)
	assert not isinstance(args.searchdir_animation,   UndefinedString)
	assert not isinstance(args.openpose_model_folder, UndefinedString)

	if not os.path.isdir(args.savedir_sample):
		os.mkdir(args.savedir_sample)
		logger.info('New directory at {path}.', path=args.savedir_sample)

	if not os.path.isdir(args.savedir_3dscene):
		os.mkdir(args.savedir_3dscene)
		logger.info('New directory at {path}.', path=args.savedir_3dscene)

	# Main
	if args.num_workers >= 1:
		mp.set_start_method('spawn')

		iter_start = args.from_sample
		iter_end = args.from_sample + args.num_samples
		iter_step = args.chunk_size
		for chunk_start in range(iter_start, iter_end, iter_step):

			# Set up a queue of serial numbers
			serialq = mp.Queue()
			for i in range(chunk_start, min(iter_end, chunk_start + args.chunk_size)):
				serialq.put(i)

			# Spawn workers
			workers = []
			for worker_id in range(args.num_workers):
				# args = (nw, gmode, serials, bjson, sjson, wsize, scene_ext, model_ext, outpath)
				p = mp.Process(target=mp_collect, args=(worker_id, serialq, args))
				workers.append(p)

			# Start workers
			for w in workers:
				w.start()

			for w in workers:
				w.join()

			serialq.close()
			serialq.join_thread()

	else:
		raise NotImplementedError
