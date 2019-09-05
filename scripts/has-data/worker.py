import pybullet as p
import numpy as np
import cv2
import os
import json
import argparse
import multiprocessing as mp
import pandas as pd
import los
import scene
import queue
import pkgutil

LineOfSight = los.LineOfSight
Scene = scene.Scene

TAU = 2 * np.pi
PI_2 = np.pi / 2
PI = np.pi


def task(basedir, mapsize, nrays, Z, camwh, scene, los):
    ctx_pos = None
    ctx_jlos = np.zeros(mapsize)
    df_ctx = pd.DataFrame(columns=['pos_x', 'pos_y', 'pos_z',
                                   'eul_x', 'eul_y', 'eul_z',
                                   'file_pxl', 'file_dep', 'file_los', 'file_jlos'])
    df_qry = pd.DataFrame(columns=['pos_x', 'pos_y', 'pos_z',
                                   'eul_x', 'eul_y', 'eul_z',
                                   'file_pxl', 'file_dep', 'file_los'])
    width, height = camwh

    for i in range(5):
        #
        # Contextual samples
        ctx_pos = scene.sample_position(ctx_pos, z=Z)
        ctx_orn = scene.sample_orientation()

        ctx_los = los.rays2map(ctx_pos, ctx_orn, nrays, scene.client)
        ctx_jlos = np.logical_or(ctx_jlos, ctx_los)
        _ctx_jlos = los.morphology(ctx_jlos * 1.0)
        _ctx_jlos = cv2.normalize(_ctx_jlos, None, 0, 255, cv2.NORM_MINMAX)

        img_pxl, img_dep = scene.get_camera_image(
            width, height, ctx_pos, ctx_orn[-1], los.hfov, .01, los.depth)

        img_pxl = cv2.cvtColor(img_pxl, cv2.COLOR_RGB2BGR)
        img_dep = cv2.normalize(img_dep, None, 0, 255, cv2.NORM_MINMAX)

        file_pxl = 'ctx-pxl-{:02d}.png'.format(i)
        file_dep = 'ctx-dep-{:02d}.png'.format(i)
        file_los = 'ctx-los-{:02d}.png'.format(i)
        file_jlos = 'ctx-jlos-{:02d}.png'.format(i)

        cv2.imwrite(os.path.join(basedir, file_pxl), img_pxl)
        cv2.imwrite(os.path.join(basedir, file_dep), img_dep)
        cv2.imwrite(os.path.join(basedir, file_los), ctx_los)
        cv2.imwrite(os.path.join(basedir, file_jlos), _ctx_jlos)

        df_ctx.loc[i] = [*ctx_pos, *ctx_orn,
                         file_pxl, file_dep, file_los, file_jlos]

        #
        # Query samples
        jlos_edge = cv2.Laplacian(ctx_jlos * 1.0, cv2.CV_64F)
        qry_spawn = np.where(jlos_edge != 0., 1., 0.)
        qry_spawn = cv2.dilate(qry_spawn, np.ones((4, 4)), iterations=1)
        qry_spawn *= ctx_jlos
        qry_spawn = np.logical_and(scene.map_spawn, qry_spawn) * 1.0

        if (qry_spawn == 0).all():
            qry_pos = ctx_pos
        else:
            qry_pos = scene.sample_position(at=None, map_spawn=qry_spawn, z=Z)

        qry_orn = scene.sample_orientation()

        qry_los = los.rays2map(qry_pos, qry_orn, nrays, scene.client)
        qry_los = los.morphology(qry_los)
        qry_los = np.logical_and(qry_los, np.logical_not(ctx_jlos)) * 1.0
        qry_los = cv2.normalize(qry_los, None, 0, 255, cv2.NORM_MINMAX)

        img_pxl, img_dep = scene.get_camera_image(
            width, height, qry_pos, qry_orn[-1], los.hfov, .01, los.depth)

        img_pxl = cv2.cvtColor(img_pxl, cv2.COLOR_RGB2BGR)
        img_dep = cv2.normalize(img_dep, None, 0, 255, cv2.NORM_MINMAX)

        file_pxl = 'qry-pxl-{:02d}.png'.format(i)
        file_dep = 'qry-dep-{:02d}.png'.format(i)
        file_los = 'qry-los-{:02d}.png'.format(i)

        cv2.imwrite(os.path.join(basedir, file_pxl), img_pxl)
        cv2.imwrite(os.path.join(basedir, file_dep), img_dep)
        cv2.imwrite(os.path.join(basedir, file_los), qry_los)

        df_qry.loc[i] = [*qry_pos, *qry_orn, file_pxl, file_dep, file_los]

    df_ctx.to_csv(os.path.join(basedir, 'ctx-data.csv'))
    df_qry.to_csv(os.path.join(basedir, 'qry-data.csv'))


def main(job, serque, args):
    client = p.connect(p.DIRECT)
    if args.enable_egl:
        print('Loading EGL plugin.')
        egl = pkgutil.get_loader('eglRenderer')
        if egl:
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
            p.loadPlugin("eglRendererPlugin")

    scene = Scene(client, args.file_mesh,
                  args.file_spawn_map, args.file_world_coords)
    los = LineOfSight(args.range_x, args.range_y, args.map_size,
                      args.camera_hfov, args.camera_vfov, args.camera_depth, args.num_zrays)

    while True:
        try:
            serial = serque.get(block=True, timeout=3)
            basedir = os.path.join(args.base_dir, '{:08d}'.format(serial))
            os.mkdir(basedir)
        except queue.Empty:
            p.disconnect(client)
            return
        except FileExistsError:
            print("'{}'' already exists.".format(basedir))
            continue
        else:
            n_rays = int(args.rays_per_deg * args.camera_hfov)
            task(basedir, args.map_size, n_rays,
                 args.camera_height, args.camera_dimension, scene, los)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--enable-egl', type=bool, default=False)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--from-sample', type=int, default=0)
    parser.add_argument('--until-sample', type=int, default=20000)  # exclusive
    parser.add_argument('--chunk-size', type=int, default=0)
    parser.add_argument('--base-dir', type=str, default='')
    parser.add_argument('--file-mesh', type=str, default='')
    parser.add_argument('--file-spawn-map', type=str, default='')
    parser.add_argument('--file-world-coords', type=str, default='')
    parser.add_argument('--range-x', type=float, nargs=2, default=[-3.5, 4.0])
    parser.add_argument('--range-y', type=float, nargs=2, default=[-3.5, 4.0])
    parser.add_argument('--map-size', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--camera-hfov', type=float, default=58.0)
    parser.add_argument('--camera-vfov', type=float, default=45.0)
    parser.add_argument('--camera-depth', type=float, default=10.0)
    parser.add_argument('--num-zrays', type=int, default=5)
    parser.add_argument('--rays-per-deg', type=int, default=7)
    parser.add_argument('--use-json', type=str, default='')

    args = parser.parse_args()
    if os.path.isfile(args.use_json):
        with open(args.use_json, 'r') as jr:
            json_data = json.load(jr)
            args.__dict__.update(json_data)

    if not os.path.isdir(args.base_dir):
        raise FileNotFoundError(
            "--base-dir: '{}' is not a valid path.".format(args.base_dir))

    if not os.path.isfile(args.file_mesh):
        raise FileNotFoundError(
            "--file-mesh: '{}' is not a valid file.".format(args.file_mesh))

    if not os.path.isfile(args.file_spawn_map):
        raise FileNotFoundError(
            "--file-spawn-map: '{}' is not a valid file.".format(args.file_spawn_map))

    if not os.path.isfile(args.file_world_coords):
        raise FileNotFoundError(
            "--file-world-coords: '{}' is not a valid file.".format(args.file_world_coords))

    scene.random.seed(args.seed)

    iter_start = args.from_sample
    iter_end = args.until_sample - args.from_sample
    iter_step = args.chunk_size if args.chunk_size > 0 else iter_end
    for chunk_start in range(iter_start, iter_end, iter_step):

        # Set up a queue of serial numbers
        serque = mp.Queue()
        for i in range(chunk_start, min(iter_end, chunk_start + iter_step)):
            serque.put(i)

        # Spawn workers
        workers = []
        for worker_id in range(args.num_workers):
            p = mp.Process(target=main,
                           args=(worker_id, serque, args))
            workers.append(p)

        # Start workers
        for w in workers:
            w.start()

        for w in workers:
            w.join()

        serque.close()
        serque.join_thread()
