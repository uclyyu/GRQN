import math, random, sys, os, json, numpy, shutil, cv2
from panda3d.core import NodePath, PerspectiveLens, AmbientLight, Spotlight, VBase4, ClockObject, loadPrcFileData
loadPrcFileData('', 'window-type offscreen')   # Spawn an offscreen buffer
loadPrcFileData('', 'audio-library-name null') # Avoid ALSA errors https://discourse.panda3d.org/t/solved-render-only-one-frame-and-other-questions/12899/5
from direct.actor.Actor import Actor 
from direct.showbase.ShowBase import ShowBase, WindowProperties, FrameBufferProperties, GraphicsPipe, GraphicsPipeSelection
from direct.gui.OnscreenText import TextNode, OnscreenText
from collections import OrderedDict
from PIL import Image
import pandas as pd
import numpy as np
from openpose import pyopenpose as openpose
_PY_OPENPOSE_AVAIL_ = True


class SceneOpError(Exception):
	pass


class SceneManager(ShowBase):
	"""The SceneManager will import two models: one for the static 
	environment layout, the other for an animated human animation. """
	def __init__(self, mode, op_config=None, 
				 scene=None, actor=None, animation=None,
				 step_phase_deg=3, extremal=None,
				 size=(256, 256), zNear=0.1, zFar=1000.0, fov=70.0, showPosition=False):
		super(SceneManager, self).__init__()

		self.__dict__.update(size=size, zNear=zNear, zFar=zFar, fov=fov, showPosition=showPosition)

		self.time = 0
		self.global_clock = ClockObject.getGlobalClock()
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
	
		# --- Configure onscreen properties
		# wp = WindowProperties()
		# wp.setSize(size[0], size[1])
		# wp.setTitle("Viewer")
		# wp.setCursorHidden(True)
		# self.win.requestProperties(wp)

		# --- Containers
		self.loader_manifest = OrderedDict([
			('label', None),
			('scene', None),
			('root', '.'),
			('serial', 0),
			('actor', ''),
			('animation', ''),
			('frame.size', size),
			('step.phase.deg', step_phase_deg)])
		self.pov_state = OrderedDict([
			('radius',    2.5), 
			('phase',     0), 
			('yaw',       0), 
			('pitch',     0), 
			('elevation', 0)])
		self.pov_reading = OrderedDict([
			('x',  None), 
			('y',  None), 
			('z',  None), 
			('h1', None), 
			('h2', None), 
			('p1', None), 
			('p2', None)])
		self.pov_state_extremal = OrderedDict([
			('radius',    (2.3, 3.5)), 
			('phase',     (-math.pi, math.pi)), 
			('yaw',       (-math.pi, math.pi)), 
			('pitch',     (-.05, .05)), 
			('elevation', (1.2, 1.4))])
		if type(extremal) is dict:
			self.pov_state_extremal.update(extremal)

		# --- Essential node paths
		self.scene = None
		self.actor = None
		self.dummy = NodePath('dummy')
		if scene is not None:
			self.scene = self.loader.loadModel(scene)
		if actor is not None and animation is not None:
			self.swapActor(actor, animation)

		# --- Initialise
		self._initialiseScene()

		# --- Entering resepctive modes
		if mode == 'collect':
			if op_config is None:
				raise
				
			self._initialiseOpenpose(op_config)
			# --- Mode for data collection
			self.taskMgr.add(self._modeCollect, 'mode-collect')

		elif mode == 'navigate':
			# --- Mode for manual scene navigation
			# sensitivity settings
			self.movSens = 2
			self.movSensFast = self.movSens * 5
			self.sensX = self.sensY = 0.2

			# key controls
			self.forward = False
			self.backward = False
			self.fast = 1.0
			self.left = False
			self.right = False
			self.up = False
			self.down = False
			self.up = False
			self.down = False
			self.stop = False

			self._setupNavigateMode()
			self.taskMgr.add(self._modeNavigate, 'mode-navigate')
			
		elif mode == 'view':
			# --- POV follows a circular motion, focusing on the actor
			self._setupViewMode()
			self.taskMgr.add(self._modeView, 'mode-view')

		else:
			ShowBase.destroy(self)
			raise ValueError('Unknow mode: {}'.format(mode))

	def _addDefaultLighting(self):
		self._lightnp_ambient = self.render.attachNewNode(AmbientLight('lightAmbient'))
		self._lightnp_ambient.node().setColor(VBase4(.4, .4, .4, 1.))
		self._lightnp_ambient.setPos(0, 0, 0)
		self.render.setLight(self._lightnp_ambient)

		self._lightnp_spot = self.render.attachNewNode(Spotlight('lightSpot'))
		self._lightnp_spot.node().setColor(VBase4(.7, .7, .7, 1.))
		self._lightnp_spot.node().setLens(PerspectiveLens())
		self._lightnp_spot.node().setShadowCaster(True, 2048, 2048)
		self._lightnp_spot.node().getLens().setFov(90)
		self._lightnp_spot.node().getLens().setNearFar(.1, 40)
		self._lightnp_spot.node().getLens().setFilmSize(2048, 2048)
		self._lightnp_spot.setPos(0, 0, 5)
		self._lightnp_spot.lookAt(self.dummy)
		self.render.setLight(self._lightnp_spot)
		self.render.setShaderAuto()

	def _actorSetYaw(self, rad):
		deg = math.degrees(rad)
		self.actor.setH(deg)

	def _actorRandomiseYaw(self):
		rad = random.uniform(-math.pi, math.pi)
		self._actorSetYaw(rad)

	def cameraCopyPOV(self):
		x = self.pov_reading['x']
		y = self.pov_reading['y']
		z = self.pov_reading['z']
		h = math.degrees(self.pov_state['yaw'])
		p = math.degrees(self.pov_state['pitch'])
		self.camera.setPos(x, y, z)
		self.camera.setH(h)
		self.camera.setP(p)

	def _initialiseScene(self):
		# self.setBackgroundColor(.5294, .8078, .9804)
		lens = self.cam.node().getLens()
		lens.setFov(self.fov)
		lens.setNear(self.zNear)
		lens.setFar(self.zFar)

		self.dummy.reparentTo(self.render)
		self.dummy.setPos(0, 0, 0)

		self._addDefaultLighting()
		self._renderFrame()

	def _initialiseOpenpose(self, config):
		self.op_datum = None
		self.op_wrapper = None
		if _PY_OPENPOSE_AVAIL_:
			self.op_datum = openpose.Datum()
			self.op_wrapper = openpose.WrapperPython()

			if type(config) is str:
				with open(config, 'r') as jc:
					self.op_wrapper.configure(json.load(jc))
			elif type(config) is dict:
				self.op_wrapper.configure(config)

			self.op_wrapper.start()

	def _modeNavigate(self, task):
		pass
		# dt = self.globalClock.getDt()
		# dt = task.time - self.time

		# if self.interactive:

		# 	# handle mouse look
		# 	md = self.win.getPointer(0)
		# 	x = md.getX()
		# 	y = md.getY()

		# 	if self.win.movePointer(0, int(self.centX), int(self.centY)):
		# 		self.cam.setH(self.cam, self.cam.getH(
		# 			self.cam) - (x - self.centX) * self.sensX)
		# 		self.cam.setP(self.cam, self.cam.getP(
		# 			self.cam) - (y - self.centY) * self.sensY)
		# 		self.cam.setR(0)

		# 	# handle keys:
		# 	if self.forward:
		# 		self.cam.setY(self.cam, self.cam.getY(
		# 			self.cam) + self.movSens * self.fast * dt)
		# 	if self.backward:
		# 		self.cam.setY(self.cam, self.cam.getY(
		# 			self.cam) - self.movSens * self.fast * dt)
		# 	if self.left:
		# 		self.cam.setX(self.cam, self.cam.getX(
		# 			self.cam) - self.movSens * self.fast * dt)
		# 	if self.right:
		# 		self.cam.setX(self.cam, self.cam.getX(
		# 			self.cam) + self.movSens * self.fast * dt)
		# 	if self.up:
		# 		self.cam.setZ(self.cam, self.cam.getZ(
		# 			self.cam) + self.movSens * self.fast * dt)
		# 	if self.down:
		# 		self.cam.setZ(self.cam, self.cam.getZ(
		# 			self.cam) - self.movSens * self.fast * dt)

		# if self.showPosition:
		# 	position = self.cam.getNetTransform().getPos()
		# 	hpr = self.cam.getNetTransform().getHpr()
		# 	self.positionText.setText(
		# 		'Position: (x = %4.2f, y = %4.2f, z = %4.2f)' % (position.x, position.y, position.z))
		# 	self.orientationText.setText(
		# 		'Orientation: (h = %4.2f, p = %4.2f, r = %4.2f)' % (hpr.x, hpr.y, hpr.z))

		# self.time = task.time

		# # Simulate physics
		# if 'physics' in self.scene.worlds:
		# 	self.scene.worlds['physics'].step(dt)

		# # Rendering
		# if 'render' in self.scene.worlds:
		# 	self.scene.worlds['render'].step(dt)

		# # Simulate acoustics
		# if 'acoustics' in self.scene.worlds:
		# 	self.scene.worlds['acoustics'].step(dt)

		# # Simulate semantics
		# # if 'render-semantics' in self.scene.worlds:
		# #     self.scene.worlds['render-semantics'].step(dt)

		# return task.cont

	def _modeView(self, task):
		for _ in range(120):
			self._povAdvancePhase(math.radians(3.))
			self._povTurnToActor()
			self.cameraCopyPOV()
			self._renderFrame()
		return task.cont

	def _modeCollect(self, task):
		manifest = pd.DataFrame(columns=[
			'activity', 'camera-pose-index', 'camera-pose', 
			'length', 'fps', 'start-frame', 
			'camv-filename', 'skel-filename'])
		step_phase_deg = self.loader_manifest['step.phase.deg']
		avail_phase_rad = np.deg2rad(np.arange(0, 360, step_phase_deg))
		T = self.actor.getNumFrames('act')

		# cv2.VideoWriter parameters
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		fps = 0.1
		render_size = tuple(self.loader_manifest['frame.size'])

		# Video filenames
		vfname_camv = 'camv.avi'
		vfname_skel = 'skel.avi'
		vfp_camv = os.path.join(self.loader_manifest['root'], vfname_camv)
		vfp_skel = os.path.join(self.loader_manifest['root'], vfname_skel)

		# Writer targets
		vw_camv = cv2.VideoWriter(vfp_camv, fourcc, fps, render_size)
		vw_skel = cv2.VideoWriter(vfp_skel, fourcc, fps, render_size)

		dfrows = 0
		# Loop over each available camera phase
		for cpi, phase_rad in enumerate(avail_phase_rad):
			# Determine full camera pose
			self._povRandomiseRadius()
			self._povRandomiseElevation()
			self._povSetState(phase=phase_rad)
			self._povTurnToActor(jitter=3)
			self.cameraCopyPOV()
			camera_pose = list(self.pov_reading.values())
			activity = self.loader_manifest['label']

			# Loop over each actor frame and save
			for timestamp in range(T):
				self.actor.pose('act', timestamp)
				self.graphicsEngine.renderFrame()

				ffname_camv = 'camv_P{:04d}.T{:04d}.jpg'.format(cpi, timestamp)
				ffname_skel = 'skel_P{:04d}.T{:04d}.jpg'.format(cpi, timestamp)
				# filename_heat = 'heat_P{:04d}.T{:04d}.jpg'.format(cpi, timestamp)
				ffp_camv = os.path.join(self.loader_manifest['root'], ffname_camv)
				ffp_skel = os.path.join(self.loader_manifest['root'], ffname_skel)
				# fp_heat = os.path.join(self.loader_manifest['root'], filename_heat)

				self._taskWriteVisual(ffp_camv, container=vw_camv)
				self._taskInvokeOpenPose(ffp_camv)
				self._taskWriteSkeleton(ffp_skel, container=vw_skel)
				# self._taskWriteHeatmap(fp_heat)

				os.remove(ffp_camv)

			# update dataframe
			manifest.loc[dfrows] = [
				activity, cpi, camera_pose, 
				T, fps, T * cpi,
				vfname_camv, vfname_skel]
			dfrows += 1

		vw_camv.release()
		vw_skel.release()
		fp_manifest = os.path.join(self.loader_manifest['root'], 'manifest.csv')
		manifest.to_csv(fp_manifest)

		return task.cont

	def _modeCollectPending(self, task):
		if self.global_clock.getFrameTime() < 5:
			self._povAdvancePhase(math.radians(1.))
			self._povTurnToActor()
			self.cameraCopyPOV()
			return task.cont
		else:
			self.taskMgr.add(self._modeCollect, 'mode-collect')	
			return task.done

	def povRandomiseState(self, phase_quadrant='all', to_actor=False):
		self._povRandomisePhase(phase_quadrant)
		self._povRandomiseRadius()
		self._povRandomiseYaw()
		self._povRandomisePitch()
		self._povRandomiseElevation()
		if to_actor:
			self._povTurnToActor(jitter=math.radians(random.uniform(0, 1)))

	def _povSetState(self, radius=None, phase=None, yaw=None, pitch=None, elevation=None):
		if radius:
			phase = self.pov_state['phase']
			x, y = radius * math.cos(phase), radius * math.sin(phase)
			self.pov_state.update({'radius': radius})
			self.pov_reading.update({'x': x, 'y': y})

		if phase:
			radius = self.pov_state['radius']
			x, y = radius * math.cos(phase), radius * math.sin(phase)
			self.pov_state.update({'phase': phase})
			self.pov_reading.update({'x': x, 'y': y})

		if yaw:
			self.pov_state.update({'yaw': yaw})
			self.pov_reading.update({'h1': math.sin(yaw), 'h2': math.cos(yaw)})

		if pitch:
			self.pov_state.update({'pitch': pitch})
			self.pov_reading.update({'p1': math.sin(pitch), 'p2': math.cos(pitch)})

		if elevation:
			self.pov_state.update({'elevation': elevation})
			self.pov_reading.update({'z': elevation})

	def _povAdvancePhase(self, by_rad):
		phase = self.pov_state['phase']
		phase = phase + by_rad
		phase = (phase + math.pi) % (2 * math.pi)
		phase = phase - math.pi
		self._povSetState(phase=phase)

	def _povAdvanceRadius(self, by):
		radius = self.pov_state['radius']
		_min, _max = self.pov_state_extremal['radius']
		radius = min(max(radius + by, _min), _max)
		self._povSetState(radius=radius)

	def _povAdvanceYaw(self, by_rad):
		yaw = self.pov_state['yaw']
		yaw = yaw + by_rad
		yaw = (yaw + math.pi) % (2 * math.pi)
		yaw = yaw - math.pi
		self._povSetState(phase=phase)

	def _povAdvancePitch(self, by):
		pit = self.pov_state['pitch']
		_min, _max = self.pov_state_extremal['pitch']
		pit = min(max(pit + by, _min), _max)
		self._povSetState(pitch=pit)

	def _povAdvanceElevation(self, by):
		ele = self.pov_state['elevation']
		_min, _max = self.pov_state_extremal['elevation']
		ele = min(max(ele + by, _min), _max)
		self._povSetState(elevation=ele)

	def _povRandomisePhase(self, quadrant):
		if quadrant == 0:
			phase = random.uniform(0, math.pi / 2)
		elif quadrant == 1:
			phase = random.uniform(math.pi / 2, math.pi)
		elif quadrant == 2:
			phase = random.uniform(-math.pi, -math.pi/2)
		elif quadrant == 3:
			phase = random.uniform(-math.pi/2, 0)
		elif quadrant == 'all':
			phase = random.uniform(-math.pi, math.pi)
		self._povSetState(phase=phase)

	def _povRandomiseRadius(self):
		radius = random.uniform(*self.pov_state_extremal['radius'])
		self._povSetState(radius=radius)

	def _povRandomiseYaw(self):
		yaw = random.uniform(*self.pov_state_extremal['yaw'])
		self._povSetState(yaw=yaw)

	def _povRandomisePitch(self):
		pit = random.uniform(*self.pov_state_extremal['pitch'])
		self._povSetState(pitch=pit)		

	def _povRandomiseElevation(self):
		ele = random.uniform(*self.pov_state_extremal['elevation'])
		self._povSetState(elevation=ele)

	def _povTurnToActor(self, jitter=0):
		x = self.pov_reading['x']
		y = self.pov_reading['y']
		z = self.pov_reading['z']
		self.camera.setPos(x, y, z)
		self.camera.lookAt(self.actor)
		deg = self.camera.getH() + random.uniform(-jitter, jitter)
		rad = math.radians(deg)
		self._povSetState(yaw=rad)

	def _povCopyReadings(self, container):
		reading = list(self.pov_reading.values())
		container.append(reading)

	def rebase(self, job, path, mkdir=True, clean=False):
		apath = os.path.abspath(path)
		if mkdir:
			try:
				os.mkdir(apath)
			except PermissionError:
				print('Cannot make new directory!')
				raise SceneOpError
			except FileExistsError:
				if clean:
					print('Removing old directory! ', apath)
					shutil.rmtree(apath)
					os.mkdir(apath)
				else:
					print('Directory alreay exists!')
					raise SceneOpError
		self.loader_manifest.update({'serial': job})			
		self.loader_manifest.update({'root': path})

	def _renderFrame(self):
		# render twice because of double buffering?
		# self.graphicsEngine.render_frame()
		self.graphicsEngine.render_frame()

	def _resampleSpotlight(self):
		z = self._lightnp_spot.getZ()
		x = random.uniform(-5, 5)
		y = random.uniform(-5, 5)
		self._lightnp_spot.setPos(x, y, z)
		self._lightnp_spot.lookAt(self.dummy)

	def step(self):
		self.taskMgr.step()

	def swapActor(self, actor, animation, loop=False):
		# Note: Frames are numbered beginning at 0
		if isinstance(self.actor, NodePath):
			self.actor.detachNode()
			self.actor.cleanup()
			# self.actor.removeNode()
		self.actor = Actor(actor, {'act': animation})
		self.actor.reparentTo(self.render)
		self.actor.setScale(0.085, 0.085, 0.085)

		label = int(actor.split(os.path.sep)[-1].split('.')[0].split('L')[-1])
		self.loader_manifest.update(
			{'label': label,
			 'actor': actor, 
			 'animation': animation})

		self._actorRandomiseYaw()
		self.povRandomiseState(to_actor=True)

		if loop:
			self.actor.loop('act')

		if hasattr(self, 'textRiggedModel'):
			self.textRiggedModel.setText(actor.split(os.path.sep)[-1])
		if hasattr(self, 'textAnimatedModel'):
			self.textAnimatedModel.setText(animation.split(os.path.sep)[-1])

	def swapScene(self, scene, resample_light=True):
		if isinstance(self.scene, NodePath):
			self.scene.detachNode()
			# self.scene.clean_up()
			self.scene.removeNode()
		self.scene = self.loader.loadModel(scene)
		self.scene.reparentTo(self.render)
		self.loader_manifest['scene'] = scene
		if resample_light:
			self._resampleSpotlight()

	def _setupCollecMode(self):
		pass

	def _setupNavigateMode(self):
		self.centX = self.win.getProperties().getXSize() / 2
		self.centY = self.win.getProperties().getYSize() / 2

		# reset mouse to start position:
		self.win.movePointer(0, int(self.centX), int(self.centY))

		self.escapeEventText = OnscreenText(text="ESC: Quit",
											style=1, fg=(1, 1, 1, 1), pos=(-1.3, 0.95),
											align=TextNode.ALeft, scale=.05)

		if self.showPosition:
			self.positionText = OnscreenText(text="Position: ",
											 style=1, fg=(1, 1, 1, 1), pos=(-1.3, 0.85),
											 align=TextNode.ALeft, scale=.05)

			self.orientationText = OnscreenText(text="Orientation: ",
												style=1, fg=(1, 1, 1, 1), pos=(-1.3, 0.80),
												align=TextNode.ALeft, scale=.05)

		# Set up the key input
		self.accept('escape', sys.exit)
		self.accept('0', setattr, [self, "stop", True])
		self.accept("w", setattr, [self, "forward", True])
		self.accept("shift-w", setattr, [self, "forward", True])
		self.accept("w-up", setattr, [self, "forward", False])
		self.accept("s", setattr, [self, "backward", True])
		self.accept("shift-s", setattr, [self, "backward", True])
		self.accept("s-up", setattr, [self, "backward", False])
		self.accept("a", setattr, [self, "left", True])
		self.accept("shift-a", setattr, [self, "left", True])
		self.accept("a-up", setattr, [self, "left", False])
		self.accept("d", setattr, [self, "right", True])
		self.accept("shift-d", setattr, [self, "right", True])
		self.accept("d-up", setattr, [self, "right", False])
		self.accept("r", setattr, [self, "up", True])
		self.accept("shift-r", setattr, [self, "up", True])
		self.accept("r-up", setattr, [self, "up", False])
		self.accept("f", setattr, [self, "down", True])
		self.accept("shift-f", setattr, [self, "down", True])
		self.accept("f-up", setattr, [self, "down", False])
		self.accept("shift", setattr, [self, "fast", 10.0])
		self.accept("shift-up", setattr, [self, "fast", 1.0])

	def _setupViewMode(self):
		self.accept('escape', sys.exit)
		self._resampleSpotlight()
		self.textRiggedModel = OnscreenText(text="",
											style=1, fg=(1, 1, 1, 1), pos=(0, 0.85),
											align=TextNode.ARight, scale=.05, mayChange=True)
		self.textAnimatedModel = OnscreenText(text="",
										  	  style=1, fg=(1, 1, 1, 1), pos=(0, 0.80),
											  align=TextNode.ARight, scale=.05, mayChange=True)

	def _slicePoses(self, start, length):
		return self.actor_valid_poses[slice(start, start + length)]

	def _samplePoseSlice(self, length):
		start = random.randint(0, len(self.actor_valid_poses) - length - 1)
		return self._slicePoses(start, length)

	def _taskInvokeOpenPose(self, name_visual):
		self.op_datum.cvInputData = cv2.imread(name_visual)
		self.op_wrapper.emplaceAndPop([self.op_datum])

	def _taskWriteVisual(self, name_visual, container=None, check=False):
		self.graphicsEngine.renderFrame()
		self.screenshot(name_visual, defaultFilename=False, source=self.win)
		# Perform checks
		if check:
			visual = Image.open(name_visual)
			assert visual.size[0] == self.size[0] and visual.size[1] == self.size[1], \
				"Visual size is {} but expected {}.".format(visual.size, self.size)
		if isinstance(container, list):
			container.append(name_visual)
		elif isinstance(container, cv2.VideoWriter):
			container.write(cv2.imread(name_visual))
		else:
			pass

	def _taskWriteHeatmap(self, name_heatmap, container=None):
		hm = self.op_datum.poseHeatMaps.copy()[0]
		hm = (hm * 255).astype('uint8')
		if isinstance(container, list):
			cv2.imwrite(name_heatmap, hm)
			container.append(name_heatmap)
		elif isinstance(container, cv2.VideoWriter):
			container.write(hm)
		else:
			cv2.imwrite(name_heatmap, hm)

	def _taskWriteSkeleton(self, name_skeleton, container=None):
		sk = self.op_datum.cvOutputData.copy()
		if isinstance(container, list):
			cv2.imwrite(name_skeleton, sk)	
			container.append(name_skeleton)
		elif isinstance(container, cv2.VideoWriter):
			container.write(sk)
		else:
			cv2.imwrite(name_skeleton, sk)	


if __name__ == '__main__':
	scene = os.path.abspath('../../resources/examples/scenes/scene_00000001.egg')
	actor = os.path.abspath('../../resources/examples/models/modl_001.egg')
	animation = os.path.abspath('../../resources/examples/models/anim_001.egg')
	config = os.path.abspath('../../resources/openpose/default_config.json')
	mode = 'collect'

	mgr = SceneManager(mode, config)
	mgr.swapScene(scene)
	mgr.swapActor(actor, animation)
	mgr.rebase('/tmp/tmp-scene', clean=True)
	mgr.step()
