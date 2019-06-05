import bpy, bmesh, os, math, random, csv
from bpy_extras.view3d_utils import location_3d_to_region_2d
from itertools import product, chain

# Reference: https://blender.stackexchange.com/questions/40823/alternative-to-redraw-timer/40826#40826

def find_first_3dview():
	for area in bpy.context.window.screen.areas:
		if area.type == 'VIEW_3D':
			sv3d = area.spaces[0]  # bpy.types.SpaceView3D
			rv3d = sv3d.region_3d  # bpy.types.RegionsView3D

			for region in area.regions:
				if region.type == 'WINDOW':
					return region, rv3d, sv3d
	return None, None, None


def view3d_camera_border(scene, cam, region, rv3d):
	# Retuens 4 points for the camera frame in pixel space
	camdat = cam.data

	frame = camdat.view_frame(scene)  # 4 points for the camera frame

	# move into object space
	frame = [cam.matrix_world * v for v in frame]

	# move into pixel space
	frame_px = [location_3d_to_region_2d(region, rv3d, v) for v in frame]

	return frame_px


def detect_ball_visibility(ball, region, rv3d, camframe_px, scene, bmball):
	for vert in bmball.verts:
		xy = location_3d_to_region_2d(region, rv3d, ball.matrix_world * vert.co)

		if ((camframe_px[2].x <= xy.x <= camframe_px[0].x) and 
			(camframe_px[1].y <= xy.y <= camframe_px[0].y)):
			return True
	return False


def ray_cast(scene, orig, verts, target_obj):
	for v in verts:
		dest = target_obj.matrix_world * v.co
		dirc = dest - orig
		dirc.normalize()

		hit, loc, norm, fidx, hit_obj, mat = scene.ray_cast(orig, dirc)
		if hit_obj == target_obj:
			return True
	return False


def get_colour(code):
	if code.upper() == 'R':
		return [1., 0., 0.]
	elif code.upper() == 'G':
		return [0., 1., 0.]
	elif code.upper() == 'B':
		return [0., 0., 1.]
	elif code.upper() == 'X':
		return list(map(lambda _: random.random(), range(3)))
	else:
		raise NotImplementedError


def initialise():
	obj = {
		'ball': bpy.data.objects['Ball'],
		'tube': bpy.data.objects['Tube']}
	mat = {
		'ball': obj['ball'].material_slots['mat.ball'].material,
		'tube': obj['tube'].material_slots['mat.tube'].material}
	cam = bpy.data.objects['Camera']

	scene = bpy.context.scene
	wregion, rv3d, sv3d = find_first_3dview()

	# --- Settings ---
	# set ball to active
	scene.objects.active = obj['ball']

	# set mode to edit
	bpy.ops.object.mode_set(mode='EDIT')

	# for unobstructed, visible vertices
	sv3d.use_occlude_geometry = True  

	# turn on camera perspective
	rv3d.view_perspective = 'CAMERA'

	# rendering parameters
	scene.render.image_settings.file_format = 'JPEG'
	scene.render.image_settings.quality = 100
	scene.render.resolution_x = 128
	scene.render.resolution_y = 128

	# for video manifest
	csv_cols = ['ball-colour', 'ball-initial-position', 'camera-pose-index', 'camera-pose', 'frame-start', 'visible-frames']

	return scene, obj, mat, cam, wregion, rv3d, csv_cols


def enumerate_inital_ball_posiiton(r=2):
	xyzs = [[r, 0], [0, r], [-r, 0], [0, -r]]
	for xy in xyzs:
		xy.append(6.5 + random.random() * 1.5)

	return xyzs


def rotate_tube(tube, deg=None):
	if deg is None:
		z_rot = math.radians(random.random() * 10. - 5.)
	else:
		z_rot = math.radians(deg)

	tube.rotation_euler[2] = z_rot

	return z_rot


def position_ball(ball, tube_rot, pos):
	z = (random.random() - 0.5) / 3
	r = sum(pos[:2])
	ball.location = pos
	ball.location[0] = r * math.cos(tube_rot + z)
	ball.location[1] = r * math.sin(tube_rot + z)


def rotate_camera(cam, phase_deg, r=8, margin_deg=10):
	margin = random.random() * margin_deg * 2 - margin_deg
	phase_rad = math.radians(phase_deg + margin)
	phase_deg = math.degrees(phase_rad)

	x = r * math.cos(phase_rad)
	y = r * math.sin(phase_rad)
	z = 2.5 + random.random() * 1.5

	pitch = 95 + random.random() * 10
	yaw = 90 + phase_deg + random.random() * 2 - 1.

	cam.location = [x, y, z]
	cam.rotation_euler.zero()
	cam.rotation_euler[0] = math.radians(pitch)
	cam.rotation_euler[2] = math.radians(yaw)

	pose = [
		x, y, z,
		math.sin(cam.rotation_euler[0]), math.cos(cam.rotation_euler[2]),
		math.sin(cam.rotation_euler[0]), math.cos(cam.rotation_euler[2])]

	return pose


def make_dataset(runs, phase, scene, obj, mat, cam, wregion, rv3d, csvhead):
	if phase == 'train':
		_colour_ball = ['R', 'G', 'B']
		_colour_tube = ['R', 'G', 'B', 'X']
	elif phase == 'test':
		_colour_ball = ['R', 'G', 'B']
		_colour_tube = ['X']
	else:
		raise NotImplementedError

	bm_ball = bmesh.from_edit_mesh(obj['ball'].data)

	scene_count = 0
	# repeat given number of runs
	for run in range(runs):

		# loop over ball colour/tube colour/initial ball position combinations
		for cb, ct, bp in product(_colour_ball, _colour_tube, enumerate_inital_ball_posiiton()):
			# set object colour
			colour_ball = get_colour(cb)
			colour_tube = get_colour(ct)
			colour_label = colour_ball.index(1)
			mat['ball'].diffuse_color = colour_ball
			mat['tube'].diffuse_color = colour_tube

			# set output folder
			fp = '/Users/hikoyu/Downloads/BallTube/{}/{:08d}'.format(phase, scene_count)
			if not os.path.isdir(fp):
				os.mkdir(fp)

			# adjust tube rotation
			deg = 45. if phase == 'test' else None
			tube_rot = rotate_tube(obj['tube'], deg)
			position_ball(obj['ball'], tube_rot, bp)

			# looping over camera poses and video frames
			frame_count = 0

			with open(os.path.join(fp, 'manifest.csv'), 'w') as csv_manifest:
				csvwriter = csv.DictWriter(csv_manifest, fieldnames=csvhead)
				csvwriter.writeheader()

				for cpi, phase_deg in enumerate(range(0, 360, 45)):
					# rotate camera and get the camera pose
					pose = rotate_camera(cam, phase_deg)
					# update camera frame in pixel space
					# camframe_px = view3d_camera_border(scene, cam, wregion, rv3d)

					# camera coordinate in object space
					cam_co = cam.location

					visible_frames = []
					data = {'frame-start': frame_count}
					for frame in range(scene.frame_start, scene.frame_end + 1):
						scene.frame_current = frame

						# write image and advance counter
						scene.render.filepath = os.path.join(fp, 'F{:08d}'.format(frame_count))
						bpy.ops.render.render(write_still=True)
						frame_count += 1

						# if detect_ball_visibility(obj['ball'], wregion, rv3d, camframe_px, scene, bm_ball):
						# 	visible_frames.append(frame_count)
						if ray_cast(scene, cam_co, bm_ball.verts, obj['ball']):
							visible_frames.append(frame_count)

					data.update({
						'ball-colour': colour_label,  # {0, 1, 2}
						'ball-initial-position': bp,  # [x, y, z]
						'camera-pose-index': cpi, 
						'camera-pose': pose, 
						'visible-frames': visible_frames})
					csvwriter.writerow(data)
			scene_count += 1


def main(runs):
	scene, obj, mat, cam, wregion, rv3d, csvhead = initialise()
	make_dataset(runs, 'train', scene, obj, mat, cam, wregion, rv3d, csvhead)
	make_dataset(runs, 'test', scene, obj, mat, cam, wregion, rv3d, csvhead)


def reset(scene, ball, tube, cam):
	scene.frame_current = scene.frame_start
	ball.location = [2., 0., 6.5]
	tube.rotation_euler = [0., 0., 0.]
	cam.location = [8., 0., 2.5]


if __name__ == '__main__':
	# for testing:
	# reset(bpy.context.scene, bpy.data.objects['Ball'], bpy.data.objects['Tube'], bpy.data.objects['Camera'])
	# z_rot = rotate_tube(bpy.data.objects['Tube'])
	# position_ball(bpy.data.objects['Ball'], z_rot, [-2., 0., 7.0])
	# rotate_camera(bpy.data.objects['Camera'], 45 * 3)

	# for data collection:
	reset(bpy.context.scene, bpy.data.objects['Ball'], bpy.data.objects['Tube'], bpy.data.objects['Camera'])
	main(runs=1)
	