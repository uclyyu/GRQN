import bpy, random, math, addon_utils, os
import multiprocessing as mp
from glob import glob
from collections import namedtuple


Vec3 = namedtuple('Vec3', ['x', 'y', 'z'])


class BadSampleException(Exception):
	pass


def _get_seeds(mode):
	if mode == 'train':
		return (0, 29999)
	elif mode == 'test':
		return (30000, 39999)
	else:
		raise ValueError('Unknwon seeding mode: {}'.format(mode))


def _deselect_all():
	for obj in bpy.context.selected_objects:
		obj.select = False


def _remove_object_by_name(name):
	if bpy.data.objects.find(name) >= 0:
		obj = bpy.data.objects[name]
		bpy.data.objects.remove(obj)


def _duplicate_object(name_proto, tag):
	bpy.data.objects[name_proto].select = True
	bpy.ops.object.duplicate(linked=False)
	dup = bpy.context.selected_objects[0]
	dup.name = '.'.join(['obj', tag, 'dup'])
	dup.data.name = '.'.join(['dat', tag, 'dup'])
	dup.hide = False
	bpy.context.scene.objects.active = dup

	return dup


def _apply_array_modifier(obj, axis, count):
	displace = [0, 0, 0]
	if axis == 'x':
		displace[0] = 1
	elif axis == 'y':
		displace[1] = 1
	elif axis == 'z':
		displace[2] = 1

	tag = 'mod.array'
	bpy.context.scene.objects.active = obj
	mod_array = obj.modifiers.new(tag, type='ARRAY')    
	mod_array.relative_offset_displace = displace
	mod_array.count = count
	bpy.ops.object.modifier_apply(apply_as='DATA', modifier=tag)


def _apply_solidify_modifier(obj, thickness):
	tag = 'mod.solidify'
	bpy.context.scene.objects.active = obj
	mod_solid = obj.modifiers.new(tag, type='SOLIDIFY')
	mod_solid.thickness = thickness
	bpy.ops.object.modifier_apply(apply_as='DATA', modifier=tag)


def _remove_random_faces(obj, percent, seed):
	bpy.context.scene.objects.active = obj
	bpy.ops.object.mode_set(mode='EDIT')
	bpy.ops.mesh.select_all(action='DESELECT')
	bpy.ops.mesh.select_mode(type='FACE', action='ENABLE')
	bpy.ops.mesh.select_random(percent=percent, seed=seed)
	bpy.ops.mesh.delete(type='FACE')
	bpy.ops.object.mode_set(mode='OBJECT')
	if len(obj.data.polygons) == 0:
		# A fail-safe section in case of all faces being deleted
		raise BadSampleException
		return True
	return False


def _nudge_random_faces(obj, percent, seed, N=100):
	bpy.context.scene.objects.active = obj
	bpy.ops.object.mode_set(mode='EDIT')
	bpy.ops.mesh.select_mode(type='VERT', action='ENABLE')
	for _ in range(N):
		bpy.ops.mesh.select_all(action='DESELECT')
		bpy.ops.mesh.select_random(percent=percent, seed=seed)
		bpy.ops.mesh.select_linked()
		i = random.randint(0, 2)
		if i == 0:
			bpy.ops.transform.translate(value=(0, random.uniform(-.01, .01), 0))
		elif i == 1:
			r = math.radians(1)
			bpy.ops.transform.rotate(value=random.uniform(-.02, .02), axis=(0, 0, 1))
		elif i == 2:
			r = math.radians(1)
			bpy.ops.transform.rotate(value=random.uniform(-.02, .02), axis=(1, 0, 0))
	bpy.ops.object.mode_set(mode='OBJECT')


def _texture_object(obj, tag, image_group):
	## Randomly sample a texture image and apply to object 'obj'.

	# Make sure material exists
	mtag = '.'.join(['mat', tag, 'dup'])
	if bpy.data.materials.find(mtag) < 0:
		mat = bpy.data.materials.new(mtag)
	else:
		mat = bpy.data.materials[mtag] 
		mat.user_clear()

	# Make sure texure exists
	ttag = '.'.join(['tex', tag, 'dup'])
	if bpy.data.textures.find(ttag) < 0:
		tex = bpy.data.textures.new(tex, type='IMAGE')
	else:
		tex = bpy.data.textures[ttag]
		tex.user_clear()

	# Assign texture to material
	if mat.texture_slots[0] is None:
		tex_slot = mat.texture_slots.add()
		tex_slot.texture = tex
	else:
		mat.texture_slots[0].texture = tex
		
	# Assign material to object
	obj.data.materials.append(mat)

	# Sample texture image and assign to texture
	imag_path, = random.sample(image_group, 1)
	imag_name = imag_path.split('/')[-1]
	if bpy.data.images.find(imag_name) < 0:
		tex.image = bpy.data.images.load(imag_path)
	else:
		tex.image = bpy.data.images[imag_name]


def sample_blind (texture_path, sample_mode='train'):
	tag = 'blind'
	name_proto, = random.sample(['.'.join(['obj', tag, 'proto', X]) for X in ['A', 'B']], 1)
	name_dup = '.'.join(['obj', tag, 'dup'])
	imag_group = glob('/'.join([texture_path, 'img.*']))
	array_count = Vec3(x=None, y=None, z=random.randint(8, 18))
	seed_range = _get_seeds(sample_mode)

	# find dup object and remove
	_remove_object_by_name(name_dup)
	
	# Deselct all objects
	_deselect_all()
	
	# Create and ready object:	
	dup = _duplicate_object(name_proto, tag)	

	# Apply random ARRAY modifier
	_apply_array_modifier(dup, 'z', array_count.z)

	# Object specific adjustments
	if name_proto.split('.')[-1] == 'A':
		# Adjust Z location, Z rotation, and X/Y scale
		dup.location[2] = random.uniform(0, 0.2)
		dup.scale[0] = random.uniform(0.8, 1.8)
		dup.scale[1] = random.uniform(0.8, 1.8)
		dup.rotation_euler[2] = random.uniform(-math.pi, math.pi)
	elif name_proto.split('.')[-1] == 'B':
		bpy.context.scene.cursor_location = [0, 0, 0]
		dup.location[1] = - 0.1 * array_count.z / 2
		bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
		dup.scale[0] = random.uniform(0.8, 1.6)
		dup.scale[1] = random.uniform(1.8, 2.0)
		dup.rotation_euler[2] = random.uniform(-math.pi, math.pi)

	# Select and delete random faces
	if _remove_random_faces(dup, random.uniform(45, 55), random.randint(*seed_range)):
		raise BadSampleException

	# Solidify object
	_apply_solidify_modifier(dup, random.uniform(0.01, 0.08))

	# Finally, deal with textures
	_texture_object(dup, tag, imag_group)


def sample_wall(texture_path, sample_mode='train'):
	tag = 'wall'
	name_proto, = random.sample(['.'.join(['obj', tag, 'proto', X]) for X in ['A']], 1) 
	name_dup_ = '.'.join(['obj', tag, 'dup', '_'])
	imag_group = glob('/'.join([texture_path, 'img.*']))
	array_count = Vec3(x=random.randint(40, 50), y=None, z=random.randint(10, 12))
	seed_range = _get_seeds(sample_mode)

	for wall in ['N', 'W', 'S', 'E']:
		name_dup = name_dup_.replace('_', wall)

		# find dup object and remove
		_remove_object_by_name(name_dup)

		# Deselct all objects
		_deselect_all()
			
		# Create and ready object:
		dup = _duplicate_object(name_proto, tag)
		dup.name = name_dup
		dup.data.name = name_dup.replace('obj', 'dat')

		# Apply random ARRAY modifier
		_apply_array_modifier(dup, 'z', array_count.z)
		_apply_array_modifier(dup, 'x', array_count.x)

		# Shuffle random mesh
		_nudge_random_faces(dup, percent=random.uniform(3, 4), seed=random.randint(*seed_range), N=10)

		# Deal with textures
		_texture_object(dup, tag, imag_group)

		# Object specific adjustments
		if wall == 'N':
			dup.location[0] = -0.2 * array_count.x / 2
			dup.location[1] = 0.2 * array_count.x / 2
		elif wall == 'W':
			dup.location[0] = -0.2 * array_count.x / 2
			dup.location[1] = -0.2 *  array_count.x / 2
			dup.rotation_euler[2] = math.radians(90)
		elif wall == 'S':
			dup.location[0] = 0.2 * array_count.x / 2
			dup.location[1] = -0.2 *  array_count.x / 2
			dup.rotation_euler[2] = math.radians(180)
		elif wall == 'E':
			dup.location[0] = 0.2 * array_count.x / 2
			dup.location[1] = 0.2 * array_count.x / 2	
			dup.rotation_euler[2] = math.radians(-90)



def sample_floor(texture_path):
	tag = 'floor'
	name_proto = '.'.join(['obj', tag])
	imag_group = glob('/'.join([texture_path, 'img.*']))

	obj = bpy.data.objects[name_proto]
	_texture_object(obj, tag, imag_group)


def sample_environment(job, texture_path_floor, texture_path_wall, texture_path_blind, export_path):
	sample_floor(texture_path_floor)
	sample_wall(texture_path_wall)
	sample_blind(texture_path_blind)

	_deselect_all()

	bpy.data.objects['obj.floor'].select = True
	bpy.data.objects['obj.blind.dup'].select = True
	bpy.data.objects['obj.wall.dup.N'].select = True
	bpy.data.objects['obj.wall.dup.W'].select = True
	bpy.data.objects['obj.wall.dup.S'].select = True
	bpy.data.objects['obj.wall.dup.E'].select = True

	export_name = 'scene_{:08d}.egg'.format(job)
	bpy.data.scenes['Scene'].yabee_settings.opt_tps_proc = 'PANDA'
	bpy.ops.export.panda3d_egg(filepath='/'.join([export_path, export_name]))


if __name__ == '__main__':
	blf = os.path.abspath('../../resources/blender/env_proto.blend')
	tpf = os.path.abspath('../../resources/textures/floor/train')
	tpw = os.path.abspath('../../resources/textures/wall/train')
	tpb = os.path.abspath('../../resources/textures/blind/train')
	exp = os.path.abspath('../../../../data/gern/egg_scene')

	addon_utils.enable('io_scene_egg')
	bpy.ops.wm.open_mainfile(filepath=blf)

	sample_environment(1, tpf, tpw, tpb, exp)