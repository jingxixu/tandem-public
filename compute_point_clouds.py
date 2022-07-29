from floating_finger_env import FloatingFingerEnv
import pybullet_utils as pu
import pybullet as p
import math
import random
import time
import itertools
import numpy as np
import argparse
import misc_utils as mu
import os
from distutils.util import strtobool


""" 
The assumption here is that the workspace is big enough to hold the object and have sufficient margins. This 
file generates the point clouds for each polygon assuming they are at a particular angle. 

It generates under the same dataset folder
    dataset_folder/
        point_clouds/
            point_clouds.npy - numpy array object type
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--angle', type=float, default=0, help='angle of the polygons to generate the point clouds')
    parser.add_argument('--dataset', type=str, default='extruded_polygons_r_0.1_s_8_h_0.05')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    save_dir = os.path.join('assets', 'datasets', args.dataset, 'point_clouds')
    env = FloatingFingerEnv(dataset=args.dataset)

    grids = np.zeros((env.num_classes, env.max_x_idx, env.max_y_idx), dtype=np.uint8)
    grids_border = np.zeros((env.num_classes, env.max_x_idx, env.max_y_idx), dtype=np.uint8)
    border_neighbors_arr = np.full(env.num_classes, {})
    for i in range(10):
        # loop through objects
        border_neighbors = {}
        env.reset(polygon_id=i, angle=args.angle)

        old_loc = (0, 0)
        pre_collision = False
        # go in the x-axis direction
        for x in range(env.max_x_idx):
            for y in range(env.max_y_idx):
                new_loc = (x, y)
                new_pose = [env.get_position_from_loc(new_loc), env.finger_initial_quaternion]
                env.finger.set_pose_no_control(new_pose)
                now_collision = env.check_collision()
                if now_collision:
                    grids[i][new_loc] = mu.white
                # entering object
                if now_collision and not pre_collision:
                    grids_border[i][new_loc] = mu.white
                    if new_loc not in border_neighbors.keys():
                        border_neighbors[new_loc] = []
                    border_neighbors[new_loc].append(old_loc)
                # leaving object
                if not now_collision and pre_collision:
                    grids_border[i][old_loc] = mu.white
                    if old_loc not in border_neighbors.keys():
                        border_neighbors[old_loc] = []
                    border_neighbors[old_loc].append(new_loc)
                pre_collision = now_collision
                old_loc = new_loc

        old_loc = (0, 0)
        pre_collision = False
        # go in the y-axis direction
        for y in range(env.max_y_idx):
            for x in range(env.max_x_idx):
                new_loc = (x, y)
                new_pose = [env.get_position_from_loc(new_loc), env.finger_initial_quaternion]
                env.finger.set_pose_no_control(new_pose)
                now_collision = env.check_collision()
                if now_collision:
                    grids[i][new_loc] = mu.white
                # entering object
                if now_collision and not pre_collision:
                    grids_border[i][new_loc] = mu.white
                    if new_loc not in border_neighbors.keys():
                        border_neighbors[new_loc] = []
                    border_neighbors[new_loc].append(old_loc)
                # leaving object
                if not now_collision and pre_collision:
                    grids_border[i][old_loc] = mu.white
                    if old_loc not in border_neighbors.keys():
                        border_neighbors[old_loc] = []
                    border_neighbors[old_loc].append(new_loc)
                pre_collision = now_collision
                old_loc = new_loc

        # mu.show_img(grids[i])
        # mu.show_img(grids_border[i])
        print(f'object {i} finishes!')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    point_clouds = []
    for i, g in enumerate(grids_border):
        point_clouds.append(mu.convert_grid_2_pc(g))
    # np.save(os.path.join(save_dir, 'grids.npy'), grids)
    # np.save(os.path.join(save_dir, 'grids_border.npy'), grids_border)
    # np.save(os.path.join(save_dir, 'border_neighbors_arr.npy'), border_neighbors_arr)
    np.save(os.path.join(save_dir, 'point_clouds.npy'), np.array(point_clouds, dtype=object))
