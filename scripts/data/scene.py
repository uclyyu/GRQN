import math, random, sys, os
from direct.actor.Actor import Actor 
from direct.showbase.ShowBase import ShowBase, WindowProperties
from direct.gui.OnscreenText import TextNode, OnscreenText
from panda3d.core import NodePath, PerspectiveLens, AmbientLight, Spotlight, VBase4, ClockObject
from collections import OrderedDict


class SceneManager(ShowBase):
	"""The SceneManager will import two models: one for the static 
	environment layout, the other for an animated human animation. """
	def __init__(self, scene, actor, animation, mode, size=(768, 768), zNear=0.1, zFar=1000.0, fov=70.0, showPosition=False, extremal=None):
		super(SceneManager, self).__init__()

		self.__dict__.update(size=size, zNear=zNear, zFar=zFar, fov=fov, showPosition=showPosition)

		self.time = 0
		self.global_clock = ClockObject.getGlobalClock()
		self.disableMouse()
	
		# Change window size
		wp = WindowProperties()
		wp.setSize(size[0], size[1])
		wp.setTitle("Viewer")
		wp.setCursorHidden(True)
		self.win.requestProperties(wp)

		# essential node paths
		self.scene = self.loader.loadModel(scene)
		self.actor = Actor(actor, {'act': animation})
		self.dummy = NodePath('dummy')

		# containers
		self.pov_state = OrderedDict([
			('radius',    0), 
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

		self._initialiseScene()

		if mode == 'collect':
			# --- Mode for data collection
			self.taskMgr.add(self._updateCollectTask, 'update-collect')

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

			self._setupViewEvent()
			self.taskMgr.add(self.updateViewTask, 'update-navigate')
			
		elif mode == 'view':
			# --- POV follows a circular motion, focusing on the actor
			self._setupViewEvents()
			self.taskMgr.add(self._updateViewTask, 'update-view')

		else:
			ShowBase.destroy(self)
			raise ValueError('Unknow mode: {}'.format(mode))

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

	def _setupCollecEvents(self):
		pass

	def _setupNavigateEvents(self):
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

	def _setupViewEvents(self):
		self.accept('escape', sys.exit)
		self.actor.loop('act')

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

	def _resampleSpotlight(self):
		z = self._lightnp_spot.getZ()
		x = random.uniform(-5, 5)
		y = random.uniform(-5, 5)
		self._lightnp_spot.setPos(x, y, z)
		self._lightnp_spot.lookAt(self.dummy)

	def _actorSetYaw(self, rad):
		deg = math.degrees(rad)
		self.actor.setH(deg)

	def _actorRandomiseYaw(self):
		rad = random.uniform(low=-math.pi, high=math.pi)
		self._actorSetYaw(rad)

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
		# if phase < math.pi:
		# 	phase = phase % math.pi
		# elif phase > math.pi:
		# 	phase = (phase % math.pi) - math.pi
		self._povSetState(phase=phase)

	def _povAdvanceRadius(self, by):
		radius = self.pov_state['radius']
		_min, _max = self.pov_state_extremal['radius']
		radius = min(max(radius + by, _min), _max)
		self._povSetState(radius=radius)

	def _povAdvanceYaw(self, by_rad):
		yaw = self.pov_state['yaw']
		yaw = yaw + by_rad
		if yaw < math.pi:
			yaw = yaw % math.pi
		elif yaw > math.pi:
			yaw = (yaw % math.pi) - math.pi
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

	def _povTurnToActor(self):
		self.camera.lookAt(self.actor)
		deg = self.camera.getH()
		rad = math.radians(deg)
		self._povSetState(yaw=rad)

	def povRandomiseState(self, phase_quardrant='all', to_actor=False):
		self._povRandomisePhase(phase_quardrant)
		self._povRandomiseRadius()
		self._povRandomiseYaw()
		self._povRandomisePitch()
		self._povRandomiseElevation()
		if to_actor:
			self._povTurnToActor()

	def cameraCopyPOV(self):
		x = self.pov_reading['x']
		y = self.pov_reading['y']
		z = self.pov_reading['z']
		h = math.degrees(self.pov_state['yaw'])
		p = math.degrees(self.pov_state['pitch'])
		self.camera.setPos(x, y, z)
		self.camera.setH(h)
		self.camera.setP(p)

	def swapActor(self, actor, animation):
		self.actor.detachNode()
		self.actor.destroy()
		self.actor = Actor(actor, {'act': animation})
		self.actor.reparentTo(self.actor)

	def swapScene(self, scene):
		self.scene.detachNode()
		self.scene.destroy()
		self.scene = self.loader.loadMode(scene)
		self.scene.reparentTo(self.render)

	def _updateNavigateTask(self, task):
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

	def _updateViewTask(self, task):
		self._povAdvancePhase(.01)
		self._povTurnToActor()
		self.cameraCopyPOV()
		return task.cont

	def _updateCollectTask(self, task):
		phase = -0.5 * numpy.pi
		self._setCamera(phase=phase, hdeg=0., pdeg=0., z=1.1)
		self._setObstacle(0, phase=phase, hdeg=180)
		return task.cont

	def step(self):
		self.taskMgr.step()


if __name__ == '__main__':
	scene = os.path.abspath('../../../../data/gern/egg_scene/scene_00000001.egg')
	actor = os.path.abspath('../../../../data/gern/egg_human/models/man0/man0pm2.egg')
	animation = os.path.abspath('../../../../data/gern/egg_human/models/man0/man0pm2-pm21.egg')

	mgr = SceneManager(scene, actor, animation, 'view')
	mgr.run()