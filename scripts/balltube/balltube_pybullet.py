import pybullet as b
import numpy as np
import pybullet_data, time, cv2, pkgutil, os, queue
import pandas as pd
import multiprocessing as mp
from itertools import chain, product
from loguru import logger

def get_tube_color_space():
	slc = slice(0.3, 1, 6j)
	grid = np.mgrid[slc, slc, slc].reshape(3, -1)
	mask = np.logical_not(np.logical_and(grid[0, :] == grid[1, :], grid[1, :] == grid[2, :]))
	grid = grid[:, mask]
	grid = np.vstack([grid, np.ones((1, grid.shape[1]))]).T 
	return grid

def get_ball_color_space():
	return [[1., 0., 0., 1.], [0., 1., 0, 1.], [0., 0., 1., 1.]]

def get_tube_yaws():
	return list(range(0, 360, 3))

class Generator(object):
	def __init__(self, client):
		self.client = client
		self.projection_matrix = [
			0.7499999403953552, 0., 0., 0., 
			0., 1., 0., 0., 
			0., 0., -1.0000200271606445, -1., 
			0., 0., -0.02000020071864128, 0.]
		# [
		# 	1.0825318098068237, 0.0, 0.0, 0.0, 
		# 	0.0, 1.732050895690918, 0.0, 0.0, 
		# 	0.0, 0.0, -1.0002000331878662, -1.0, 
		# 	0.0, 0.0, -0.020002000033855438, 0.0]

		self.cam_distance = 5.0
		self.cam_pitch = 0.0
		self.cam_roll = 0.0
		self.cam_yaw = None
		self.cam_ztarget = 1.7
		self.clip_length = 16

	def reset(self, tube_color, ball_color, tube_yaw):
		tube_orientation = b.getQuaternionFromEuler([0., 0., tube_yaw])

		self.tube_visual = b.createVisualShape(
			b.GEOM_MESH,
			fileName='../../resources/blender/tube.obj',
			meshScale=[1., 1., 1.],
			rgbaColor=tube_color,
			visualFrameOrientation=tube_orientation,
			physicsClientId=self.client)

		self.ball_visual = b.createVisualShape(
			b.GEOM_MESH,
			fileName='../../resources/blender/ball.obj',
			meshScale=[2., 2., 2.],
			rgbaColor=ball_color,
			physicsClientId=self.client)

		self.ball_collision = b.createCollisionShape(
			b.GEOM_MESH,
			fileName='../../resources/blender/ball.obj',
			meshScale=[2., 2., 2.],
			physicsClientId=self.client)

		self.tube_body = b.createMultiBody(
			baseMass=0,
			baseCollisionShapeIndex=-1,
			baseVisualShapeIndex=self.tube_visual,
			physicsClientId=self.client)

		self.ball_body = b.createMultiBody(
			baseMass=1.,
			baseCollisionShapeIndex=self.ball_collision,
			baseVisualShapeIndex=self.ball_visual,
			basePosition=[0., 0., 3],
			physicsClientId=self.client)

		self.floor = b.loadURDF('plane.urdf', physicsClientId=self.client)
		self.ball_color = ball_color.index(1)
		# self.light_direction = [
		# 	np.random.uniform(-2, 2), 
		# 	np.random.uniform(-2, 2), 
		# 	np.random.uniform(3, 5)]
		b.setGravity(0., 0., -1.)

	def get_camview(self, view_matrix):
		camview = b.getCameraImage(
			256, 256, 
			viewMatrix=view_matrix, projectionMatrix=self.projection_matrix,
			shadow=1, physicsClientId=self.client)

		return camview[2]

	def get_campose(self):
		distance = self.cam_distance
		target = [0., 0., self.cam_ztarget]
		radian_pitch = np.deg2rad(self.cam_pitch)
		radian_yaw = np.deg2rad(self.cam_yaw - 90)
		sin_pitch = np.sin(radian_pitch)
		cos_pitch = np.cos(radian_pitch)
		sin_yaw = np.sin(radian_yaw)
		cos_yaw = np.cos(radian_yaw)
		z = - sin_pitch * distance + target[2]
		x = cos_pitch * cos_yaw * distance + target[0]
		y = cos_pitch * sin_yaw * distance + target[1]

		return [x, y, z, sin_pitch, cos_pitch, sin_yaw, cos_yaw]

	def simulate(self):
		b.setTimeStep(40./240., physicsClientId=self.client)
		b.setPhysicsEngineParameter(numSubSteps=40, physicsClientId=self.client)

		manifest = pd.DataFrame(columns=['ball-color', 'camera-pose', 'start-frame', 'length'])

		yaws = list(range(0, 360, 45))
		cam_captures = [[] for _ in range(8)]
		view_matrices = []

		target = [0., 0., self.cam_ztarget]
		for i, yaw in enumerate(yaws):
			self.cam_yaw = yaw
			view_matrix = b.computeViewMatrixFromYawPitchRoll(target, self.cam_distance, self.cam_yaw, self.cam_pitch, self.cam_roll, 2, physicsClientId=self.client)
			view_matrices.append(view_matrix)
			manifest.loc[len(manifest)] = [self.ball_color, self.get_campose(), self.clip_length * i, self.clip_length]

		for t in range(self.clip_length):
			for i, view_matrix in enumerate(view_matrices):
				view = self.get_camview(view_matrix)
				cam_captures[i].append(view)
			b.stepSimulation(physicsClientId=self.client)

		return manifest, cam_captures

def main(worker, argque, savedir_root):
	client = b.connect(b.DIRECT)
	b.setAdditionalSearchPath(pybullet_data.getDataPath())	
	gen = Generator(client)

	while True:
		try:
			counter, tcolor, tyaw, bcolor = argque.get(timeout=3)
		except queue.Empty:
			return
		else:
			t0 = time.time()
			b.resetSimulation(client)
			gen.reset(tcolor, bcolor, tyaw)
			manifest, cam_captures = gen.simulate()

			if (counter % 1260) in [0, 1, 2]:
				phase = 'test'
			else:
				phase = 'train'

			savedir = os.path.join(savedir_root[phase], '{:08d}'.format(counter))
			if not os.path.isdir(savedir):
				os.makedirs(savedir)

			fp_vid = os.path.join(savedir, 'video.avi')
			fp_csv = os.path.join(savedir, 'manifest.csv')
			vidout = cv2.VideoWriter(fp_vid, cv2.VideoWriter_fourcc(*'XVID'), 1, (64, 64))
			for frame in chain(*cam_captures):
				vidout.write(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR), (64, 64), cv2.INTER_CUBIC))

			vidout.release()
			manifest.to_csv(fp_csv)
			t1 = time.time()

			logger.info('Worker: {worker} :: {counter:08d} ({second}s)', worker=worker, counter=counter, second=t1 - t0)


if __name__ == '__main__':
	mp.set_start_method('spawn')

	tube_colors = get_tube_color_space()
	ball_colors = get_ball_color_space()
	tube_yaws = get_tube_yaws()

	savedir_root = {
		'train': '/home/yen/data/balltube/train',
		'test':  '/home/yen/data/balltube/test'}

	argque = mp.Queue()
	for i, (tcolor, tyaw, bcolor) in enumerate(product(tube_colors, tube_yaws, ball_colors)):
		argque.put((i, tcolor, tyaw, bcolor))

	workers = []
	for worker_id in range(8):
		p = mp.Process(target=main, args=(worker_id, argque, savedir_root))
		workers.append(p)

	for worker in workers:
		worker.start()

	for worker in workers:
		worker.join()

	argque.close()
	argque.join_thread()
