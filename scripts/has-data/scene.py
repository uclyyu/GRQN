import pybullet as p
import cv2
import numpy as np
from numpy import random

TAU = 2 * np.pi


class Scene:
    def __init__(self, client, meshfile, map_spawn, coords_world, value=255):
        self.client = client
        self.mesh_vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=meshfile,
            meshScale=[1., 1., 1.],
            physicsClientId=client)
        self.mesh_col = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=meshfile,
            meshScale=[1., 1., 1.],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            physicsClientId=client)
        self.mesh_bdy = p.createMultiBody(
            baseMass=0.,
            baseVisualShapeIndex=self.mesh_vis,
            baseCollisionShapeIndex=self.mesh_col,
            physicsClientId=client)

        if isinstance(map_spawn, str):
            self.map_spawn = cv2.imread(map_spawn, cv2.IMREAD_GRAYSCALE)
        elif isinstance(map_spawn, np.ndarray):
            self.map_spwan = map_spawn
        else:
            raise ValueError("map_spawn should be a string or numpy.ndarray")

        if isinstance(coords_world, str):
            self.coords_world = np.load(coords_world)
        elif isinstance(coords_world, np.ndarray):
            self.coords_world = coords_world
        else:
            raise ValueError(
                "coords_world should be a string or numpy.ndarray")

        assert self.map_spawn.size == len(self.coords_world)

        self.world_spawnables = self.coords_world[np.where(
            self.map_spawn.ravel() == value)]

    def sample_position(self, at=None, z=None, map_spawn=None, value=255):
        if map_spawn is None:
            spawnables = self.world_spawnables
        else:
            if isinstance(map_spawn, str):
                map_spawn = cv2.imread(map_spawn, cv2.IMREAD_GRAYSCALE)
            elif isinstance(map_spawn, np.ndarray):
                map_spawn = map_spawn.astype(np.float)
                map_spawn = (map_spawn - map_spawn.max()) / \
                    (map_spawn.max() - map_spwan.min())
            else:
                raise ValueError(
                    "map_spawn should be a string or numpy.ndarray")

            spawnables = self.coords_world[np.where(
                map_spawn.ravel() == value)]

        if at is None:
            index = random.randint(0, len(spawnables))
            px, py, pz = spawnables[index]
            pz = pz if z is None else z
            pos = [px, py, pz]
        else:
            at = np.add(at, [0.1 * random.randn(), 0.1 * random.randn(), 0.])
            d = at - self.spawnables
            d = (d ** 2).sum(axis=1)
            px, py, pz = spawnables[d.argmin()]
            pz = pz if z is None else z
            pos = [px, py, pz]

        return pos

    def sample_orientation(self):
        return [0., 0., random.rand() * TAU]

    def get_camera_image(self, width, height, position, heading, hfov, near, far, up=[0, 0, 1]):
        aspect = width / height
        target = [np.cos(heading), -np.sin(heading), 0.]
        target = np.add(position, target)

        mat_v = p.computeViewMatrix(position, target, up, self.client)
        mat_p = p.computeProjectionMatrixFOV(
            hfov, aspect, near, far, self.client)

        _, _, rgb, dep, _ = p.getCameraImage(
            width, height, mat_v, mat_p,
            physicsClientId=self.client)

        return rgb, dep
