import multiprocessing as mp
import queue, sys, argparse, json, scene, os, random


def mp_collect(worker, serial, bjson, sjson, wsize, extension, output_base):
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

	# --- List of actors and animations
	actors = glob.glob(os.path.sep.join([sp_actr, 'rigg_*{}'.format(extension)]))
	animations = glob.glob(os.path.sep.join([sp_anim, 'anim_*{}'.format(extension)]))

	# --- blender: load export utility
	blender.addon_utils.enable('io_scene_egg')
	blender.bpy.ops.wm.open_mainfile(filepath=bp_main)

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
			# Sample and generate .egg scene
			blender.sample_environment(job, bp_flor, bp_wall, bp_blnd, bp_expo)
			scene = os.path.sep.join([bp_expo, 'scene_{:08d}.egg'.format(job)])

			# Sample actor and animation
			actor, animation = _sample_actors(actors, animations)

			# Update scene manager
			output_path = os.path.sep.join(output_base, '{:08d}'.format(job))
			smgr.swapScene(scene)
			smgr.swapActor(actor, animation)
			smgr.rebase(output_path)
			smgr.step()


def _sample_actors(actors, animations):
	actor, = random.sample(actors, 1)
	animation, = random.sample(animations, 1)
	return actor, animation

def _update_main_dict(main, args):
	for k, v in args.items():
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


if __name__ == '__main__':
	main = sys.modules['__main__']

	parser = argparse.ArgumentParser(description='Multi-worker data generator.')
	parser.add_argument('--num-worker', type=int, dest='nworker', metavar='N', required=True, help='(integer) Number of multiprocessing workers.')
	parser.add_argument('--num-sample', type=int, dest='nsample', metavar='M', required=True, help='(integer) Number of samples to draw.')
	parser.add_argument('--file-extension', type=str, dest='extension', metavar='.EXT', required=True, default='.bam', help='')
	parser.add_argument('--win-size', type=str, dest='wsize', metavar='WxH', required=True, default='128x128')
	parser.add_argument('--out-base', type=str, dest='outpath', metavar='PATH', required=True, help='Root path for generated samples.')
	parser.add_argument('--blender-json', type=str, dest='bjson', metavar='B', required=True, help='')
	parser.add_argument('--scene-json', type=str, dest='sjson', metavar='S', required=True, help='')

	arg_dict = vars(parser.parse_args())
	_update_main_dict(main, arg_dict)

	# Keep a copy of the arguments
	with open(os.path.sep.join(outpath, 'worker_arguments.json'), 'w') as jw:
		json.dump(arg_dict, jw)
	
	if nworker > 1:
		mp.set_start_method('spawn')

		# Set up a queue of serial numbers
		serials = mp.Queue()
		for i in range(nsample):
			serials.put(i)

		# Spawn workers
		workers = []
		for j in range(nworker):
			args = (worker, serial, bjson, sjons, wsize, extension, outpath)
			p = mp.Process(target=mp_collect, args=args)
			workers.append(p)

		# Start workers
		for w in workers:
			w.start()

		for w in workers:
			w.join()

	else:
		raise NotImplementedError




