import math, random, numpy, sys
from direct.actor.Actor import Actor 
from direct.showbase.ShowBase import ShowBase, WindowProperties
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import NodePath, PerspectiveLens, AmbientLight, Spotlight, VBase4, ClockObject
from collections import namedtuple


POVState = namedtuple('POVState', ['radius', 'elevation', 'phase', 'yaw'])
POVReading = namedtuple('POVReading', [])


class SceneManager(ShowBase):
	"""The SceneManager will import two models: one for the static 
	environment layout, the other for an animated human animation. """
	def __init__(self, scene, actor, animation, mode, size=(768, 768), zNear=0.1, zFar=1000.0, fov=70.0, showPosition=False):
		super(SceneManager, self).__init__()

		self.__dict__.update(size=size, zNear=zNear, zFar=zFar, fov=fov, showPosition=showPosition)

		self.time = 0
		self.globalClock = ClockObject.getGlobalClock()

		self.disableMouse()
		# self.setBackgroundColor(.5294, .8078, .9804)

		lens = self.cam.node().getLens()
		lens.setFov(self.fov)
		lens.setNear(self.zNear)
		lens.setFar(self.zFar)

		# Change window size
		wp = WindowProperties()
		wp.setSize(size[0], size[1])
		wp.setTitle("Viewer")
		wp.setCursorHidden(True)
		self.win.requestProperties(wp)

		self.centX = self.win.getProperties().getXSize() / 2
		self.centY = self.win.getProperties().getYSize() / 2

		# reset mouse to start position:
		self.win.movePointer(0, int(self.centX), int(self.centY))

		# essential node paths
		self.scene = self.loader.loadModel(scene)
		self.actor = Actor(actor, {'act': animation})
		self.dummy = NodePath('dummy')

		self.scene.reparentTo(self.render)
		self.actor.reparentTo(self.render)
		self.dummy.reparentTo(self.render)

		self.pov_state = POVState(None, None, None, None)

		self._initialiseScene()

		if mode == 'collect':
			self.taskMgr.add(self.updateCollectTask, 'update-collect')
		elif mode == 'navigate':
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
			
		elif mode == 'demo':
			self._setupDemoEvents()
			self.taskMgr.add(self.updateDemoTask, 'update-demo')

		elif mode == 'calibrate':
			self._setupCalibrateEvents()
			self.taskMgr.add(self.updateCalibrateTask, 'update-calibrate')
		else:
			ShowBase.destroy(self)
			raise ValueError()

	def _initialiseScene(self):
		self.scene.reparentTo(self.render)
		self.actor.reparentTo(self.render)
		self.dummy.reparentTo(self.render)

		self.actor.setScale(0.085, 0.085, 0.085)

	def _setupCollecEvents(self):
		pass

	def _setupNavigateEvents(self):

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

	def _setupDemoEvents(self):
		self.accept('escape', sys.exit)

	def _setupCalibrateEvents(self):
		self.accept('escape', sys.exit)

	def _setupDefaultScene(self):
		self.scene.reparentTo(self.render)
		self.dummy.reparentTo(self.render)

		self._addDefaultLighting()
		self._resampleActor()
		self._randomiseCameraPosition()
		self._pointCameraAtActor()
		self._resampleObstacle()
		self._randomiseObstaclePosition()

		self.dummy.setPos(0, 0, 0)

	def _addDefaultLighting(self):
		self.lightAmbientNp = self.render.attachNewNode(AmbientLight('lightAmbient'))
		self.lightAmbientNp.node().setColor(VBase4(.4, .4, .4, 1.))
		self.lightAmbientNp.setPos(0, 0, 0)
		self.render.setLight(self.lightAmbientNp)

		self.lightSpotNp = self.render.attachNewNode(Spotlight('lightSpot'))
		self.lightSpotNp.node().setColor(VBase4(.7, .7, .7, 1.))
		self.lightSpotNp.node().setLens(PerspectiveLens())
		self.lightSpotNp.node().setShadowCaster(True, 2048, 2048)
		self.lightSpotNp.node().getLens().setFov(70)
		self.lightSpotNp.node().getLens().setNearFar(.1, 40)
		self.lightSpotNp.node().getLens().setFilmSize(2048, 2048)
		self.lightSpotNp.setPos(0, 0, 8)
		self.lightSpotNp.lookAt(self.dummy)
		self.render.setLight(self.lightSpotNp)
		self.render.setShaderAuto()

	def _resampleSpotlight(self):
		z = self.lightSpotNp.getZ()
		x, y = numpy.random.uniform(-4, 4, 2)
		self.lightSpotNp.setPos(x, y, z)

		x, y = numpy.random.uniform(-8, 8, 2)
		self.dummy.setPos(x, y, 0)

		self.lightSpotNp.lookAt(self.dummy)


	def _resampleActor(self):
		# detach current actor from scene graph and 
		# sample a new obstacle and place it under the scene graph
		N = len(self.actorList)
		
		i = self.actorState['id']
		self.actorList[i].stop()
		self.actorList[i].detachNode()  
		
		j = random.randint(0, N - 1)
		self.actorList[j].setScale(0.085, 0.085, 0.085)
		self.actorList[j].reparentTo(self.render)
		self.actorList[j].loop('act')
		self.actorState.update({'id': j})

	def _actorSetYaw(self, rad):
		deg = math.degrees(rad)
		self.actor.setH(deg)

	def _actorRandomiseYaw(self):
		rad = random.uniform(low=-math.pi, high=math.pi)
		self._actorSetYaw(rad)

	def _povSetState(self, radius=None, phase=None, yaw=None, elevation=None):

		pass

	def _povRandomiseState(self):
		radius = random.uniform(2.3, 3.5)
		phase = random.uniform(-math.pi, math.pi)
		yaw = random.uniform(-math.pi, math.pi)
		elevation = random.uniform(1.3, 1.5)

		self._povSetState(radius, phase, yaw, elevation)

	def _povLookAtActor(self):
		i = self.actorState['id']
		self.camera.lookAt(self.actorList[i])
		pdeg = self.cameraState['pdeg']
		hdeg = self.camera.getH()
		hrad = math.radians(hdeg)
		self.camera.setP(pdeg)

		self.cameraState.update({'hdeg': hdeg, 'hrad': hrad})


	def _setCamera(self, phase, hdeg, pdeg, z=1.3, radius=2):
		x, y = radius * numpy.cos(phase), radius * numpy.sin(phase)
		hrad = math.radians(hdeg)
		prad = math.radians(pdeg)
		self.camera.setPos(x, y, z)
		self.camera.setH(hdeg)
		self.camera.setP(pdeg)

		self.cameraState.update({'phase': phase, 'radius': radius, 'x': x, 'y': y, 'z': z, 'hdeg': hdeg, 'pdeg': pdeg, 'hrad': hrad, 'prad': prad})

	def _randomiseCameraPosition(self):
		u = self.cameraState['radius']
		v = numpy.random.uniform(low=-numpy.pi, high=numpy.pi)
		x, y = u * numpy.cos(v), u * numpy.sin(v)
		z = self.cameraState['z']

		hdeg = numpy.random.uniform(low=-180, high=180)
		pdeg = numpy.random.uniform(low=-10, high=3)
		hrad = math.radians(hdeg)
		prad = math.radians(pdeg)

		self.camera.setPos(x, y, z)
		self.camera.setP(pdeg)
		self.camera.setH(hdeg)

		self.cameraState.update({'hdeg': hdeg, 'pdeg': pdeg, 'hrad': hrad, 'prad': prad, 'phase': v, 'x': x, 'y': y, 'z': z})

	def _moveCameraByDegree(self, deg):
		vdeg = (math.degrees(self.cameraState['phase']) + deg) % 360
		vrad = math.radians(vdeg)
		u = 2.0
		x, y = u * numpy.cos(vrad), u * numpy.sin(vrad)
		z = self.cameraState['z']

		self.camera.setPos(x, y, z)
		self.cameraState.update({'x': x, 'y': y, 'phase': vrad})
		self._pointCameraAtActor()

	

	def updateViewTask(self, task):

		# dt = self.globalClock.getDt()
		dt = task.time - self.time

		if self.interactive:

			# handle mouse look
			md = self.win.getPointer(0)
			x = md.getX()
			y = md.getY()

			if self.win.movePointer(0, int(self.centX), int(self.centY)):
				self.cam.setH(self.cam, self.cam.getH(
					self.cam) - (x - self.centX) * self.sensX)
				self.cam.setP(self.cam, self.cam.getP(
					self.cam) - (y - self.centY) * self.sensY)
				self.cam.setR(0)

			# handle keys:
			if self.forward:
				self.cam.setY(self.cam, self.cam.getY(
					self.cam) + self.movSens * self.fast * dt)
			if self.backward:
				self.cam.setY(self.cam, self.cam.getY(
					self.cam) - self.movSens * self.fast * dt)
			if self.left:
				self.cam.setX(self.cam, self.cam.getX(
					self.cam) - self.movSens * self.fast * dt)
			if self.right:
				self.cam.setX(self.cam, self.cam.getX(
					self.cam) + self.movSens * self.fast * dt)
			if self.up:
				self.cam.setZ(self.cam, self.cam.getZ(
					self.cam) + self.movSens * self.fast * dt)
			if self.down:
				self.cam.setZ(self.cam, self.cam.getZ(
					self.cam) - self.movSens * self.fast * dt)

		if self.showPosition:
			position = self.cam.getNetTransform().getPos()
			hpr = self.cam.getNetTransform().getHpr()
			self.positionText.setText(
				'Position: (x = %4.2f, y = %4.2f, z = %4.2f)' % (position.x, position.y, position.z))
			self.orientationText.setText(
				'Orientation: (h = %4.2f, p = %4.2f, r = %4.2f)' % (hpr.x, hpr.y, hpr.z))

		self.time = task.time

		# Simulate physics
		if 'physics' in self.scene.worlds:
			self.scene.worlds['physics'].step(dt)

		# Rendering
		if 'render' in self.scene.worlds:
			self.scene.worlds['render'].step(dt)

		# Simulate acoustics
		if 'acoustics' in self.scene.worlds:
			self.scene.worlds['acoustics'].step(dt)

		# Simulate semantics
		# if 'render-semantics' in self.scene.worlds:
		#     self.scene.worlds['render-semantics'].step(dt)

		return task.cont

	def updateDemoTask(self, task):
		self._moveCameraByDegree(1)
		self._moveObstacleByDegree(-1.5)
		return task.cont

	def updateCalibrateTask(self, task):
		phase = -0.5 * numpy.pi
		self._setCamera(phase=phase, hdeg=0., pdeg=0., z=1.1)
		self._setObstacle(0, phase=phase, hdeg=180)
		return task.cont

	def step(self):
		self.taskMgr.step()


if __name__ == '__main__':
	model = 'egg/room.egg'
	actors = [['egg/prepare_meal-I.egg', {'act': 'egg/prepare_meal-I-PM1.egg'}]]
	obstacles = ['egg/obstacle-01.egg', 'egg/obstacle-02.egg', 'egg/obstacle-03.egg']
	rlsm = RLSceneManager(model, actors, obstacles, 'calibrate', fov=70)
	rlsm.run()