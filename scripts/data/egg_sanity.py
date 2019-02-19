import scene, os
from glob import glob




if __name__ == '__main__':
	smgr = scene.SceneManager('view')

	scene = os.path.abspath('../../resources/examples/scenes/scene_00000001.egg')
	riggs = glob('/home/yen/data/gern/egg_human/train/rigg*.egg')
	
	smgr.swapScene(scene)
	for rigg in riggs:
		anim = rigg.replace('rigg_', 'anim_').replace('.egg', '_0000.egg')
		smgr.swapActor(rigg, anim, loop=True)
		smgr.step()