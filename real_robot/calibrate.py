import rtde_control
import rtde_receive
import sys
sys.path.append("../")
import real_robot_utils as ru
from real_floating_finger_env import RealFloatingFingerEnv
import numpy as np
import time
import rospy
import misc_utils as mu
import gym
import torch
import argparse
from distutils.util import strtobool
import pprint


""" Move the robot along the four edges of the workspace """


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_ob', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    args = parser.parse_args()
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
        finger_height=0.0158,    # to match simulation
        initial_q=INITIAL_Q,
        original_xy=ORIGINAL_XY,
        render_ob=True
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    rospy.init_node('demo')

    env.reset(do_nothing=True, polygon_id=0)
    initial_pos = env.env.rtde_r.getActualTCPPose()[:3]
    for move in [mu.down, mu.right, mu.up, mu.left]:
        action = dict()
        action['move'] = move
        action['prediction'] = 0
        action['done'] = False
        action['max_prob'] = 0.1
        action['probs'] = [0.1] * 10
        # the grid is 60 by 60, the largest index is 59. From index 0 to index 59, there is only 59 * step size.
        # So the finger can never reach (30cm , 30cm)
        for i in range(60-1):
            env.step(action)
    env.close()