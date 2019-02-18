import math, random, sys, os, json, numpy
from panda3d.core import NodePath, PerspectiveLens, AmbientLight, Spotlight, VBase4, ClockObject, loadPrcFileData
loadPrcFileData('', 'window-type offscreen')   # Spawn an offscreen buffer
loadPrcFileData('', 'audio-library-name null') # Avoid ALSA errors https://discourse.panda3d.org/t/solved-render-only-one-frame-and-other-questions/12899/5
from direct.actor.Actor import Actor 
from direct.showbase.ShowBase import ShowBase, WindowProperties
from direct.gui.OnscreenText import TextNode, OnscreenText

from collections import OrderedDict
try:
	import cv2
	from openpose import openpose
	_PY_OPENPOSE_AVAIL_ = True
except ImportError:
	_PY_OPENPOSE_AVAIL_ = False


class SceneManager(ShowBase):
	"""The SceneManager will import two models: one for the static 
	environment layout, the other for an animated human animation. """
	def __init__(self, scene, actor, animation, mode, op_config, pose_gap=4, size=(768, 768), zNear=0.1, zFar=1000.0, fov=70.0, showPosition=False, extremal=None):
		super(SceneManager, self).__init__()

		self.__dict__.update(size=size, zNear=zNear, zFar=zFar, fov=fov, showPosition=showPosition)

		self.time = 0
		self.global_clock = ClockObject.getGlobalClock()
		self.disableMouse()
	
		# --- Change window size (on-screen)
		# wp = WindowProperties()
		# wp.setSize(size[0], size[1])
		# wp.setTitle("Viewer")
		# wp.setCursorHidden(True)
		# self.win.requestProperties(wp)

		# --- Change window size (off-screen)
		self.win.setSize(size[0], size[1])

		# --- Containers
		self.loader_manifest = OrderedDict([
			('root', '.')
			('serial', 0),
			('actor', ''),
			('animation', ''),
			('num.pre', None),
			('num.pose', None),
			('num.rewind', None),
			('poses.pre', None),
			('poses.post', None),
			('poses.rewind', None),
			('visuals.rewind', []),
			('visuals.condition', []),
			('heatmaps.rewind', []),
			('heatmaps.condition', []),
			('skeletons.rewind', []),
			('skeletons.condition', []),
			('pose.gap.size', pose_gap),
			('pov.readings.rewind', []),
			('pov.readings.condition', [])])
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
		self.scene = self.loader.loadModel(scene)
		self.dummy = NodePath('dummy')
		self.swapActor(actor, animation)

		# --- Initialise
		self._initialiseScene()
		self._initialiseOpenpose(op_config)

		# --- Entering resepctive modes
		if mode == 'collect':
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

		self.scene.reparentTo(self.render)
		self.actor.reparentTo(self.render)
		self.dummy.reparentTo(self.render)

		self.actor.setScale(0.085, 0.085, 0.085)
		self.dummy.setPos(0, 0, 0)

		self._addDefaultLighting()

		self.povRandomiseState(to_actor=True)

	def _initialiseOpenpose(self, config):
		self.op_datum = None
		self.op_wrapper = None
		if _PY_OPENPOSE_AVAIL_:
			self.op_datum = openpose.Datum()
			self.op_wrapper = openpose.WrapperPython()

			with open(config, 'r') as jc:
				self.op_wrapper.configure(json.load(jc))

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
		self._povAdvancePhase(math.radians(1.))
		self._povTurnToActor()
		self.cameraCopyPOV()
		return task.cont

	def _modeCollect(self, task):
		num_pose_pre = random.randint(1, 8)
		num_pose_post = random.randint(16, 32)
		num_pose_rewind = random.randint(1, int(num_pose_post / 2))
		pose_slice = self._samplePoseSlice(num_pose_pre + num_pose_post)
		pose_slice_pre = pose_slice[:num_pose_pre]
		pose_slice_post = pose_slice[num_pose_pre:]
		pose_slice_rewind = list(reversed(pose_slice_post[-num_pose_rewind:]))

		self.loader_manifest.update({
			'num.pre': num_pose_pre,
			'num.post': num_pose_post,
			'num.rewind': num_pose_rewind,
			'poses.pre': pose_slice_pre,
			'poses.post': pose_slice_post,
			'poses.rewind': pose_slice_rewind
			})

		# --- Prepare conditional inputs
		# Allow up to two samples per quadrant
		for i, q in enumerate(pose_slice_pre):
			self.actor.pose('act', q)
			if i < 4:
				# The first centers on the actor		
				self.povRandomiseState(phase_quadrant='all', to_actor=True)
			else:
				# The second has a randomised view
				self.povRandomiseState(phase_quadrant='all', to_actor=False)

			# Record camera states
			self.cameraCopyPOV()
			self._povCopyReadings(self.loader_manifest['pov.readings.condition'])

			# Screenshots, heatmaps, and skeletons
			name_visual = 'visual-cond-{:03d}.jpg'.format(i)
			name_heatmap = 'heatmap-cond-{:03d}.jpg'.format(i)
			name_skeleton = 'skeleton-cond-{:03d}.jpg'.format(i)
			self._renderFrame()
			self._taskWriteVisual(name_visual, self.loader_manifest['visuals.condition'])
			if _PY_OPENPOSE_AVAIL_:
				self._taskInvokeOpenPose(name_visual)
				self._taskWriteHeatmap(name_heatmap, self.loader_manifest['heatmaps.condition'])
				self._taskWriteSkeleton(name_skeleton, self.loader_manifest['skeletons.condition'])

		# Next, sample temporal data from a random position
		self.povRandomiseState(phase_quadrant='all', to_actor=True)
		step = math.copysign(math.radians(1), random.uniform(-1, 1))
		for i, q in enumerate(pose_slice_post):
			k = i + num_pose_pre
			self.actor.pose('act', q)
			# Move POV
			self._povAdvancePhase(step)
			self._povTurnToActor()
			# Record camera states
			self.cameraCopyPOV()
			self._povCopyReadings(self.loader_manifest['pov.readings.condition'])

			# Screenshots, heatmaps, and skeletons
			name_visual = 'visual-cond-{:03d}.jpg'.format(k)
			name_heatmap = 'heatmap-cond-{:03d}.jpg'.format(k)
			name_skeleton = 'skeleton-cond-{:03d}.jpg'.format(k)
			self._renderFrame()
			self._taskWriteVisual(name_visual, self.loader_manifest['visuals.condition'])
			if _PY_OPENPOSE_AVAIL_:
				self._taskInvokeOpenPose(name_visual)
				self._taskWriteHeatmap(name_heatmap, self.loader_manifest['heatmaps.condition'])
				self._taskWriteSkeleton(name_skeleton, self.loader_manifest['skeletons.condition'])

		# --- Prepare query outputs
		self.povRandomiseState(phase_quadrant='all', to_actor=True)
		if random.uniform(-1, 1) < 0:
			# Random sample
			for i, q in enumerate(pose_slice_rewind):
				self.actor.pose('act', q)
				self._povTurnToActor(jitter=math.radians(random.uniform(-3, 3)))
				self.cameraCopyPOV()
				self._povCopyReadings(self.loader_manifest['pov.readings.rewind'])

				name_visual = 'visual-rewd-{:03d}.jpg'.format(i)
				name_heatmap = 'heatmap-rewd-{:03d}.jpg'.format(i)
				name_skeleton = 'skeleton-rewd-{:03d}.jpg'.format(i)
				self._renderFrame()
				self._taskWriteVisual(name_visual, self.loader_manifest['visuals.rewind'])
				if _PY_OPENPOSE_AVAIL_:
					self._taskInvokeOpenPose(name_visual)
					self._taskWriteHeatmap(name_heatmap, self.loader_manifest['heatmaps.rewind'])
					self._taskWriteSkeleton(name_skeleton, self.loader_manifest['skeletons.rewind'])
				self.povRandomiseState(phase_quadrant='all', to_actor=True)

		else:
			# Circular sample
			step = math.copysign(math.radians(1), random.uniform(-1, 1))
			for i, q in enumerate(pose_slice_rewind):
				print('r: ', i)
				self.actor.pose('act', q)
				self._povAdvancePhase(step)
				self._povTurnToActor()
				self.cameraCopyPOV()
				self._povCopyReadings(self.loader_manifest['pov.readings.rewind'])

				name_visual = 'visual-rewd-{:03d}.jpg'.format(i)
				name_heatmap = 'heatmap-rewd-{:03d}.jpg'.format(i)
				name_skeleton = 'skeleton-rewd-{:03d}.jpg'.format(i)
				self._renderFrame()
				self._taskWriteVisual(name_visual, self.loader_manifest['visuals.rewind'])
				if _PY_OPENPOSE_AVAIL_:
					self._taskInvokeOpenPose(name_visual)
					self._taskWriteHeatmap(name_heatmap, self.loader_manifest['heatmaps.rewind'])
					self._taskWriteSkeleton(name_skeleton, self.loader_manifest['skeletons.rewind'])

		# --- Finally, output manifest
		with open('manifest.json', 'w') as jf:
			json.dump(self.loader_manifest, jf)


		# --- Revert back to pending state
		# self.taskMgr.add(self._modeCollectPending, 'mode-collect-pending')
		print('Done!')
		return task.done

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

	def rebase(self, path, mkdir=True):
		apath = os.path.abspath(path)
		if mkdir:
			os.mkdir(apath)
		self.loader_manifest.update('root': path)

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

	def swapActor(self, actor, animation):
		if hasattr(self, 'actor'):
			self.actor.detachNode()
			self.actor.destroy()
		self.actor = Actor(actor, {'act': animation})
		self.actor.reparentTo(self.render)

		total_frames = self.actor.getNumFrames('act')
		gap = self.loader_manifest['pose.gap.size']
		self.actor_valid_poses = numpy.arange(0, total_frames, gap).tolist()

		self.loader_manifest.update(
			{'actor': actor, 
			 'animation': animation,
			 'poses.pre': None,
			 'poses.post': None,
			 'poses.rewind': None})
		self.loader_manifest['visuals.rewind'].clear()
		self.loader_manifest['visuals.condition'].clear()
		self.loader_manifest['heatmaps.rewind'].clear()
		self.loader_manifest['heatmaps.condition'].clear()
		self.loader_manifest['skeletons.rewind'].clear()
		self.loader_manifest['skeletons.condition'].clear()
		self.loader_manifest['pov.readings.rewind'].clear()
		self.loader_manifest['pov.readings.condition'].clear()

		self._actorRandomiseYaw()

	def swapScene(self, scene, resample_light=True):
		self.scene.detachNode()
		self.scene.destroy()
		self.scene = self.loader.loadMode(scene)
		self.scene.reparentTo(self.render)

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
		self.actor.loop('act')

	def _slicePoses(self, start, length):
		return self.actor_valid_poses[slice(start, start + length)]

	def _samplePoseSlice(self, length):
		start = random.randint(0, len(self.actor_valid_poses) - length - 1)
		return self._slicePoses(start, length)

	def _taskInvokeOpenPose(self, name_visual):
		self.op_datum.cvInputData = cv2.imread(name_visual)
		self.op_wrapper.emplaceAndPop([self.op_datum])

	def _taskWriteVisual(self, name_visual, container):
		self.graphicsEngine.renderFrame()
		self.screenshot(name_visual, defaultFilename=False, source=self.win)
		container.append(name_visual)

	def _taskWriteHeatmap(self, name_heatmap, container):
		hm = self.op_datum.poseHeatMaps.copy()[0]
		hm = (hm * 255).astype('uint8')
		cv2.imwrite(name_heatmap, hm)
		container.append(name_heatmap)

	def _taskWriteSkeleton(self, name_skeleton, container):
		sk = self.op_datum.cvOutputData.copy()
		cv2.imwrite(name_skeleton, sk)
		container.append(name_skeleton)


if __name__ == '__main__':
	scene = os.path.abspath('../../resources/examples/scenes/scene_00000001.egg')
	actor = os.path.abspath('../../resources/examples/models/modl_001.egg')
	animation = os.path.abspath('../../resources/examples/models/anim_001.egg')
	config = os.path.abspath('../../resources/openpose/default_config.json')
	mode = 'collect'

	mgr = SceneManager(scene, actor, animation, mode, config)
	mgr.step()
