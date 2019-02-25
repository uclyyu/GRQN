import multiprocessing as mp
import queue, sys, argparse, json, scene, os, random, shutil, glob


def mp_collect(worker, mode, serial, bjson, sjson, wsize, extension, output_base):
	import blender
	# --- blender: json information 
	bp_main = os.path.abspath(bjson['blender_main_file'])
	bp_flor = os.path.abspath(bjson['texture_path_floor'])
	bp_wall = os.path.abspath(bjson['texture_path_wall'])
	bp_blnd = os.path.abspath(bjson['texture_path_blind'])
	bp_expo = os.path.abspath(bjson['export_path'])

	# --- scene: json information
	sp_actr = os.path.abspath(sjson['actor_search_path'])
	sp_anim = os.path.abspath(sjson['animation_search_path'])
	sp_pose = os.path.abspath(sjson['openpose_config'])

	# --- List of actors
	actors = glob.glob(os.path.sep.join([sp_actr, 'L??.R????{}'.format(extension)]))
	# animations = glob.glob(os.path.sep.join([sp_anim, 'L??.R????.A????_*{}'.format(extension)]))

	# --- blender: load export utility
	bp_new = os.path.join(
		output_base, '../blender', 
		bp_main.split(os.path.sep)[-1].replace('.blend', '{}.blend'.format(worker)))
	shutil.copy(bp_main, bp_new)
	blender.addon_utils.enable('io_scene_egg')
	blender.bpy.ops.wm.open_mainfile(filepath=bp_new)

	# --- A new instance of scene manager
	smgr = scene.SceneManager('collect', sp_pose, size=wsize)

	# --- Main procedures
	while True:
		try:
			job = serial.get(timeout=3)
		except queue.Empty:
			print('Worker ({}): empty queue.'.format(worker))
			break
		else:
			print('Worker ({}): processing job {:,}.'.format(worker, job))
			if random.uniform(0, 1) < .5:
				use_blind = False
			else:
				use_blind = True

			# Sample and generate .egg scene
			blender.sample_environment(worker, mode, bp_flor, bp_wall, bp_blnd, bp_expo, use_blind=use_blind)
			scene_file = os.path.sep.join([bp_expo, 'scene_{:08d}{}'.format(worker, extension)])

			# Sample actor and animation
			actor, animation = _sample_actors(actors, sp_anim, extension)

			# Update scene manager
			output_path = os.path.sep.join([output_base, '{:08d}'.format(job)])
			smgr.swapScene(scene_file)
			smgr.swapActor(actor, animation)
			smgr.rebase(output_path)
			smgr.step()


def _sample_actors(actors, anim_search_path, extension):
	actor, = random.sample(actors, 1)
	search_pattern = actor.split(os.path.sep)[-1].replace(extension, '.A????' + extension)
	search_path = os.path.sep.join([anim_search_path, search_pattern])
	animations = glob.glob(search_path)
	animation, = random.sample(animations, 1)
	return actor, animation


def _update_main_dict(main, args):
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


if __name__ == '__main__':
	main = sys.modules['__main__']

	parser = argparse.ArgumentParser(description='Multi-worker data generator.')
	parser.add_argument('--mode',           type=str, dest='gmode',     metavar='MODE',                    help='Genearte {train|test} dataset.')
	parser.add_argument('--num-worker', 	type=int, dest='nworker', 	metavar='N', 					   help='Number of multiprocessing workers.')
	parser.add_argument('--num-sample', 	type=int, dest='nsample', 	metavar='N', 					   help='Number of samples to draw.')
	parser.add_argument('--from-sample',    type=int, dest='fsample',   metavar='N',                       help='Genearte from sample N.')
	parser.add_argument('--file-extension', type=str, dest='extension', metavar='.EXT', default='.bam',	   help='Panda3D model extension.')
	parser.add_argument('--win-size', 	  	type=str, dest='wsize', 	metavar='WxH',  default='128x128', help='Render window size.')
	parser.add_argument('--out-base',	  	type=str, dest='outpath', 	metavar='PATH', 				   help='Root path for generated samples.')
	parser.add_argument('--blender-json', 	type=str, dest='bjson', 	metavar='blender.json', 		   help='blender.py config file.')
	parser.add_argument('--scene-json',   	type=str, dest='sjson', 	metavar='scene.json',  			   help='scene.py config file.')
	parser.add_argument('--worker-json',  	type=str, dest='wjson', 	metavar='worker.json',			   help='JSON file to override command line arguments.')

	arg_dict = vars(parser.parse_args())
	arg_dict_processed = _update_main_dict(main, arg_dict)

	# Keep a copy of the arguments
	configpath = os.path.sep.join([outpath, '../config'])
	if not os.path.exists(configpath):
		os.mkdir(configpath)
	if arg_dict_processed:
		with open(os.path.sep.join([configpath, 'worker_arguments.json']), 'w') as jw:
			json.dump(arg_dict_processed, jw)

	# Main
	if nworker > 1:
		mp.set_start_method('spawn')

		# Set up a queue of serial numbers
		serials = mp.Queue()
		for i in range(fsample, fsample + nsample):
			serials.put(i)

		# Spawn workers
		workers = []
		for nw in range(nworker):
			args = (nw, gmode, serials, bjson, sjson, wsize, extension, outpath)
			p = mp.Process(target=mp_collect, args=args)
			workers.append(p)

		# Start workers
		for w in workers:
			w.start()

		for w in workers:
			w.join()

	else:
		raise NotImplementedError
