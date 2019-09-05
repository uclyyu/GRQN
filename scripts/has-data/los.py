import numpy as np
import cv2
import pybullet as p
from functools import reduce


class LineOfSight:
    def __init__(self, x_range, y_range, map_size, hfov=58., vfov=45., depth=10., nzrays=4, param_morphology=dict()):
        """
        Args:
            x_range (tuple of 2 floats): World X range; 1 unit = 1 metre
            y_range (tuple of 2 floats): World Y range; 1 unit = 1 metre
            map_size (tuple of 2 floats): Size of the grid world (x, y)
            hfov (float, optional): Camera horizontal FOV in degrees
            depth (float, optional): Camera depth in metre
        """
        self.x_range = x_range
        self.y_range = y_range
        self.width = x_range[1] - x_range[0]
        self.height = y_range[1] - y_range[0]
        self.hfov = hfov
        self.vfov = vfov
        self.depth = depth
        self.map_size = map_size
        self.z_rays = nzrays

        grid = np.meshgrid(np.linspace(x_range[0], x_range[1], map_size[0]),
                           np.linspace(y_range[1], y_range[0], map_size[1]))
        self.grid = np.stack(grid, axis=2)
        self.param_morphology = {
            'erode': {'kernel': np.ones((8, 8), dtype=np.int), 'iterations': 1},
            'dilate': {'kernel': np.ones((8, 8), dtype=np.int), 'iterations': 2},
        }
        self.param_morphology.update(param_morphology)

    def rays(self, position, orientation, num):
        """Returns ray origins and destinations for RayTestBatch
        Args:
            position (list of 3 floats): Camera position in (X, Y, Z)
            orientation (list of 3 floats): Camera orientation in euler angles (radians)
            num (int): Number of rays
        """
        pos_x, pos_y, pos_z = position
        orr_x, orr_y, orr_z = orientation
        fov_2 = np.radians(self.hfov / 2)

        radians = np.linspace(-fov_2, fov_2, num)
        affine_z = np.array([
            [np.cos(orr_z), -np.sin(orr_z), 0],
            [np.sin(orr_z), np.cos(orr_z), 0],
            [0, 0, 1]])

        x = [self.depth]
        y = self.depth * np.tan(radians)
        z = np.linspace(pos_z, .05, self.z_rays)

        x, y, z = np.meshgrid(x, y, z)
        xyz = np.stack([x, y, z], axis=3).reshape(-1, 3).dot(affine_z)

        dest = xyz + [pos_x, pos_y, 0]
        orig = np.tile([pos_x, pos_y, pos_z], (num * self.z_rays, 1))

        return orig, dest

    def hit(self, rtb_output, destination):
        """Returns hit positions in world coordinates.
        Args:
            rtb_output (list): A listed returned by pybullet.rayTestBatch
            destination (list of floats): Ray destination
        Returns:
            Array-like: True ray destinations
        """
        hid_id, link_id, hit_frac, hit_pos, hit_norm = zip(*rtb_output)

        hit_frac = np.array(hit_frac)
        hit_pos = np.array(hit_pos)

        hit_pos[:, 0] = np.where(
            hit_frac == 1, destination[:, 0], hit_pos[:, 0])
        hit_pos[:, 1] = np.where(
            hit_frac == 1, destination[:, 1], hit_pos[:, 1])

        return hit_pos

    def map_index(self, xy):
        """Returns coordinates as map 2D indices.
        Args:
            xy (tuple of 2 floats or a list of tuple of 2 floats): x and y world coordinate
        Returns:
            A list of tuples of 2 floats: row and column indices to map
        """
        if np.ndim(xy) == 1:
            xy = [xy]

        index = map(lambda _xy: tuple(np.unravel_index(
            ((self.grid - _xy) ** 2).sum(axis=2).argmin(), self.map_size)[-1::-1]), xy)

        return list(index)

    def rays2map(self, position, orientation, num, client):
        spx, spy, _ = position
        ray_orig, ray_dest = self.rays(position, orientation, num)

        rtb_output = []
        for i in range(0, len(ray_orig), 1024):
            slc = slice(i, i + 1024)
            rtb_output.extend(p.rayTestBatch(
                ray_orig[slc], ray_dest[slc], physicsClientId=client))

        true_dest = self.hit(rtb_output, ray_dest)

        index_orig, = self.map_index((spx, spy))
        index_dest = self.map_index(true_dest[:, :2])
        index_dest = np.reshape(index_dest, [-1, self.z_rays, 2])

        img = np.zeros([self.z_rays] + self.map_size)
        for j in range(self.z_rays):
            for pt in index_dest[:, j]:
                img[j] = cv2.line(img[j], index_orig, tuple(
                    pt), (255, 255, 255), 1, cv2.LINE_8)

        img = reduce(np.logical_and, img) * 255.

        return img

    def orays2map(self):
        pass

    def morphology(self, img):
        img = cv2.dilate(
            img, self.param_morphology['dilate']['kernel'],
            iterations=self.param_morphology['dilate']['iterations'])
        img = cv2.erode(
            img, self.param_morphology['erode']['kernel'],
            iterations=self.param_morphology['erode']['iterations'])

        return img
