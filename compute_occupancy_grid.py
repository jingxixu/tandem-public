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
file generates the ground truth grids numpy file for easy sampling of locations on the border.

It generates
    save_dir/
        grids.npy - occupied being white, otherwise black
        grids_border.npy - border being white, otherwise black
        rgb.npy - rgb array for visualization border and neighbors together
        border_neighbors_arr.npy - np array of dict of list of neighbor location tuples
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_orientations', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='extruded_primitives')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    save_dir = os.path.join('assets', 'datasets', args.dataset, 'new_ori_{}'.format(args.num_orientations))
    env = FloatingFingerEnv(num_orientations=args.num_orientations)
    gap = int(360 / args.num_orientations)
    angles = [0 + i * gap for i in range(args.num_orientations)]
    # angles = [0, 36, 72, 108, 144, 180, 216, 252, 288, 324]

    grids = np.zeros((env.num_classes, len(angles), env.max_x_idx, env.max_y_idx), dtype=np.uint8)
    grids_border = np.zeros((env.num_classes, len(angles), env.max_x_idx, env.max_y_idx), dtype=np.uint8)
    rgb = np.zeros((env.num_classes, len(angles), env.max_x_idx, env.max_y_idx, 3), dtype=np.uint8)
    border_neighbors_arr = np.full((env.num_classes, len(angles)), {})
    for i in range(10):
        # loop through objects
        for a_i, a in enumerate(angles):
            # loop through different angles
            border_neighbors = {}
            env.reset(polygon_id=i, angle=a)

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
                        grids[i][a_i][new_loc] = mu.white
                    # entering object
                    if now_collision and not pre_collision:
                        grids_border[i][a_i][new_loc] = mu.white
                        if new_loc not in border_neighbors.keys():
                            border_neighbors[new_loc] = []
                        border_neighbors[new_loc].append(old_loc)
                    # leaving object
                    if not now_collision and pre_collision:
                        grids_border[i][a_i][old_loc] = mu.white
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
                        grids[i][a_i][new_loc] = mu.white
                    # entering object
                    if now_collision and not pre_collision:
                        grids_border[i][a_i][new_loc] = mu.white
                        if new_loc not in border_neighbors.keys():
                            border_neighbors[new_loc] = []
                        border_neighbors[new_loc].append(old_loc)
                    # leaving object
                    if not now_collision and pre_collision:
                        grids_border[i][a_i][old_loc] = mu.white
                        if old_loc not in border_neighbors.keys():
                            border_neighbors[old_loc] = []
                        border_neighbors[old_loc].append(new_loc)
                    pre_collision = now_collision
                    old_loc = new_loc

            # verify the border neighbors
            rgb_ = np.tile(grids_border[i][a_i], (3, 1, 1))
            rgb_ = np.transpose(rgb_, [1, 2, 0])
            for loc, neighbors in border_neighbors.items():
                for n in neighbors:
                    rgb_[n] = [255, 0, 0]
            rgb[i][a_i] = rgb_
            border_neighbors_arr[i][a_i] = border_neighbors

            # mu.show_img(rgb[i][a_i])
            # mu.show_img(grids[i][a_i])
            # mu.show_img(grids_border[i][a_i])
            mu.save_rgb(rgb[i][a_i], os.path.join(save_dir, 'info', f'{i}_{a}.png'))
        print(f'object {i} finishes!')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'grids.npy'), grids)
    np.save(os.path.join(save_dir, 'grids_border.npy'), grids_border)
    np.save(os.path.join(save_dir, 'rgb.npy'), rgb)
    np.save(os.path.join(save_dir, 'border_neighbors_arr.npy'), border_neighbors_arr)
