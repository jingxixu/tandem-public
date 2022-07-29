import rtde_control
import rtde_receive
import sys
sys.path.append("../")
import real_robot_utils as ru
from real_floating_finger_env import RealFloatingFingerEnv
from enjoy_floating_finger import enjoy_floating_finger
import numpy as np
import time
import rospy
import misc_utils as mu
import gym
import torch
import argparse
from distutils.util import strtobool
import pprint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_ob', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--seed', type=int, default=10)

    # env related
    parser.add_argument('--terminal_confidence', type=float, default=0.98)
    parser.add_argument('--polygon_id', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=1)

    # discriminator
    parser.add_argument('--discriminator', type=str, default='learned',
                        help='one of "dummy", "gt", or "learned"')
    parser.add_argument('--discriminator_path', type=str,
                        help='path to the learned discriminator model checkpoint or to the gt discriminator grids path')
    parser.add_argument('--dataset', type=str, default='extruded_polygons_r_0.1_s_8_h_0.05')

    # explorer
    parser.add_argument('--explorer', type=str, default='ppo')
    parser.add_argument('--explorer_path', type=str)

    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    return args


if __name__ == "__main__":
    args = get_args()
    INITIAL_Q = [-1.12, -1.34, 1.36, -1.60, -1.61, 0.11]
    ORIGINAL_XY = [-0.478, 0.128]
    # ORIGINAL_XY = [-0.478, 0.138]
    max_ep_len = 100

    env = RealFloatingFingerEnv(
        max_ep_len=max_ep_len,
        max_x=0.3,
        max_y=0.3,
        step_size=0.005,    # 5mm
        robot_ip='192.168.0.166',
        finger_height=0.0158,    # to match simulation tip to floor distance 4cm
        initial_q=INITIAL_Q,
        original_xy=ORIGINAL_XY,
        render_ob=True
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    rospy.init_node('demo')

    mu.seed_env(env, args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)   # some of the explorers use np random, so needs to seed them as well.

    res = enjoy_floating_finger(env,
                                args.discriminator,
                                args.discriminator_path,
                                args.dataset,
                                args.explorer,
                                args.explorer_path,
                                args.terminal_confidence,
                                save_npy=True,
                                num_episodes=args.num_episodes,
                                exp_name=f'polygon_{args.polygon_id}_{args.explorer}_{args.discriminator}_{args.timestr}',
                                polygon_id=args.polygon_id)
    print()
    pprint.pprint(res, indent=4)
    print()
    env.close()
