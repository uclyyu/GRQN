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
        print('ctx')
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
        qry_spawn = np.where(jlos_edge < 0., 1., 0.)
        qry_spawn = cv2.dilate(qry_spawn, np.ones((8, 8)), iterations=1)
        qry_spawn *= ctx_jlos
        qry_spawn = np.logical_and(scene.map_spawn, qry_spawn)

        print('qry')
        pos = scene.sample_position(at=None, map_spawn=qry_spawn, z=Z)
        orn = scene.sample_orientation()

        qry_los = los.rays2map(pos, orn, nrays, scene.client)
        qry_los = los.morphology(qry_los)
        qry_los = np.logical_and(qry_los, np.logical_not(ctx_jlos)) * 1.
        qry_los = cv2.normalize(qry_los, None, 0, 255, cv2.NORM_MINMAX)

        img_pxl, img_dep = scene.get_camera_image(
            width, height, pos, orn[-1], los.hfov, .01, los.depth)

        img_pxl = cv2.cvtColor(img_pxl, cv2.COLOR_RGB2BGR)
        img_dep = cv2.normalize(img_dep, None, 0, 255, cv2.NORM_MINMAX)

        file_pxl = 'qry-pxl-{:02d}.png'.format(i)
        file_dep = 'qry-dep-{:02d}.png'.format(i)
        file_los = 'qry-los-{:02d}.png'.format(i)

        cv2.imwrite(os.path.join(basedir, file_pxl), img_pxl)
        cv2.imwrite(os.path.join(basedir, file_dep), img_dep)
        cv2.imwrite(os.path.join(basedir, file_los), qry_los)

        df_qry.loc[i] = [*pos, *orn, file_pxl, file_dep, file_los]

    df_ctx.to_csv(os.path.join(basedir, 'ctx-data.csv'))
    df_qry.to_csv(os.path.join(basedir, 'qry-data.csv'))


def main(job, serque, args):
    client = p.connect(p.DIRECT)
    scene = Scene(client, args.file_mesh,
                  args.file_spawn_map, args.file_world_coords)
    los = LineOfSight(args.range_x, args.range_y, args.map_size,
                      args.camera_hfov, args.camera_vfov, args.camera_depth, args.num_zrays)

    while True:
        try:
            serial = serque.get(block=True, timeout=3)
        except queue.Empty:
            p.disconnect(client)
            return
        else:
            n_rays = int(args.rays_per_deg * args.camera_hfov)
            task(args.base_dir, args.map_size, n_rays,
                 args.camera_height, args.camera_dimension, scene, los)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--from-sample', type=int)
    parser.add_argument('--until-sample', type=int)  # exclusive
    parser.add_argument('--chunk-size', type=int)
    parser.add_argument('--base-dir', type=str)
    parser.add_argument('--file-mesh', type=str)
    parser.add_argument('--file-spawn-map', type=str)
    parser.add_argument('--file-world-coords', type=str)
    parser.add_argument('--range-x', type=int, nargs=2)
    parser.add_argument('--range-y', type=int, nargs=2)
    parser.add_argument('--map-size', type=int, nargs=2)
    parser.add_argument('--camera-hfov', type=float)
    parser.add_argument('--camera-vfov', type=float)
    parser.add_argument('--camera-depth', type=lambda x: max(0, float(x)))
    parser.add_argument('--num-zrays', type=lambda x: max(1, int(x)))
    parser.add_argument('--rays-per-deg', type=int)
    parser.add_argument('--use-json', type=str)

    args = parser.parse_args()
    if os.path.isfile(args.use_json):
        with open(args.use_json, 'r') as jr:
            json_data = json.load(jr)
            args.__dict__.update(json_data)

    iter_start = args.from_sample
    iter_end = args.until_sample - args.from_sample
    iter_step = args.chunk_size
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
