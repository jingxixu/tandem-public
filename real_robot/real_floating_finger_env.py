import time

import rtde_control
import rtde_receive
import sys
sys.path.append('./')
import copy
import math
import numpy as np
import gym
import misc_utils as mu
import cv2
import rospkg
import rospy
from std_msgs.msg import Bool


class RealFloatingFingerEnv(gym.Env):
    def __init__(self,
                 max_ep_len=2000,
                 max_x=0.3,
                 max_y=0.3,
                 step_size=0.005,
                 robot_ip='192.168.0.166',
                 finger_height=0.0158,
                 initial_q=[-1.12, -1.34, 1.36, -1.60, -1.61, 0.11],
                 original_xy=[-0.488, 0.128],
                 render_ob=True,
                 ):
        self.max_ep_len = max_ep_len
        self.max_x = max_x
        self.max_y = max_y
        self.render_ob = render_ob
        self.step_size = step_size
        # discretize the workspace
        self.max_x_idx = round(self.max_x / self.step_size)
        self.max_y_idx = round(self.max_y / self.step_size)
        self.robot_ip = robot_ip
        self.finger_height = finger_height
        self.tcp_orientation = [-np.pi, 0, 0]
        self.initial_q = initial_q
        # this is where matches the location (0, 0)
        # top left of the image
        self.original_xy = original_xy
        self.starting_pose = self.original_xy + [self.finger_height] + self.tcp_orientation
        self.pre_starting_pose = self.original_xy + [self.finger_height+0.1] + self.tcp_orientation
        # define the interfaces
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.collision_sub = rospy.Subscriber("/contact/collision", Bool, self.collision_callback)
        self.collision = False

        self.move_dim = len(mu.move_map)
        self.num_classes = 10
        self.action_space = gym.spaces.Dict({"move": gym.spaces.Discrete(self.move_dim),
                                             "prediction": gym.spaces.Discrete(self.num_classes),
                                             "probs": gym.spaces.Box(low=0, high=1, shape=(self.num_classes, )),
                                             "max_prob": gym.spaces.Box(low=0, high=1, shape=(1, )),
                                             "done": gym.spaces.Discrete(2)})
        self.observation_space = gym.spaces.Box(low=np.zeros((1, self.max_x_idx, self.max_y_idx)),
                                                high=np.full((1, self.max_x_idx, self.max_y_idx), 255), dtype=np.uint8)

        # step related info
        self.rendered_occupancy = False
        self.current_loc = (0, 0)
        self.current_step = 0
        self.success = False
        self.done = False
        self.occupancy_grid = np.full((1, self.max_x_idx, self.max_y_idx), mu.unexplored, dtype=np.uint8)
        self.initial_explored_pixel = 0
        self.polygon_id = None
        self.angle = None

    def reset(self, polygon_id=None, angle=None, do_nothing=False):
        assert polygon_id is not None
        self.done = False
        self.info = None
        self.success = False
        self.current_step = 0
        self.occupancy_grid = np.full((1, self.max_x_idx, self.max_y_idx), mu.unexplored, dtype=np.uint8)
        # self.polygon_id = self.np_random.randint(low=0, high=10) if polygon_id is None else polygon_id
        self.polygon_id = polygon_id
        self.current_loc = (0, 0)

        self.cartesian_control('z', 0.1)
        # self.rtde_c.moveJ(self.initial_q)
        self.rtde_c.moveL(self.pre_starting_pose)
        self.rtde_c.moveL(self.starting_pose)
        import ipdb; ipdb.set_trace()

        if do_nothing:
            return

        # always starts on boarder
        moves = self.generate_heuristic_moves()
        num_moves = 0
        collision = False
        while not collision:
            # manually introduce sensor failure for testing
            override_collision = False
            # if num_moves == 20:
            #     override_collision = True
            collision = self.move(moves[num_moves], collision=override_collision)
            num_moves += 1
            if self.render_ob:
                self.render_grid()

        self.ob = copy.deepcopy(self.occupancy_grid)
        if self.render_ob:
            self.render_grid()
        self.initial_explored_pixel = np.count_nonzero(self.occupancy_grid != mu.unexplored)
        # print(f'env id: {self.env_id}\t client id: {self.client_id}\t polygon id: {self.polygon_id}')
        # import ipdb; ipdb.set_trace()
        return self.ob

    def move(self, move, collision=False):
        """ also return collision or not """
        """ This function will also set current_loc and occupancy grid """
        goal_loc = self.compute_next_loc(move)
        traj_poses = self.get_traj_poses(self.current_loc, goal_loc)
        for pose in traj_poses:
            self.rtde_c.moveL(pose, speed=0.005, acceleration=0.01)
            time.sleep(0.01)
            # collision provides an option to overwrite the actual collision signal
            if self.collision or collision:
                self.rtde_c.moveL(traj_poses[0], speed=0.005, acceleration=0.01)
                self.occupancy_grid[0][goal_loc] = mu.white
                return True
        self.occupancy_grid[0][self.current_loc] = mu.black
        self.current_loc = goal_loc
        self.occupancy_grid[0][self.current_loc] = mu.current_black
        return False

    def get_traj_poses(self, start_loc, goal_loc, step=5):
        start_position = self.get_position_from_loc(start_loc)
        goal_position = self.get_position_from_loc(goal_loc)
        traj_positions = np.linspace(start_position, goal_position, num=step+1)
        traj_poses = []
        for pos in traj_positions:
            traj_poses.append(list(pos) + self.tcp_orientation)
        return traj_poses

    def cartesian_control(self, axis, distance):
        current_position = self.rtde_r.getActualTCPPose()[:3]
        # computer new pose
        new_position = copy.deepcopy(current_position)
        if axis == 'x':
            new_position[0] += distance
        elif axis == 'y':
            new_position[1] += distance
        elif axis == 'z':
            new_position[2] += distance
        else:
            raise TypeError('unrecognized axis')
        new_pose = new_position + self.tcp_orientation
        self.rtde_c.moveL(new_pose)

    def close(self):
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()

    def compute_next_loc(self, move):
        return mu.compute_next_loc(self.current_loc, move, self.max_x_idx, self.max_y_idx)

    def get_position_from_loc(self, loc):
        # This is where magic happens!
        return [self.original_xy[0] - loc[0] * self.step_size, self.original_xy[1] - loc[1] * self.step_size, self.finger_height]

    def step(self, action):
        move = action['move']
        prediction = action['prediction']
        max_prob = action['max_prob']
        probs = action['probs']
        done = action['done']

        # The move function is very cool. Should use this in simulation as well.
        self.move(move)
        self.current_step += 1
        self.success = self.check_success(prediction)
        self.done = done or self.current_step >= self.max_ep_len

        self.ob = copy.deepcopy(self.occupancy_grid)
        self.info = {'num_explored_pixels': np.count_nonzero(self.occupancy_grid != mu.unexplored) - self.initial_explored_pixel,  # the first pixel from
                     # reset does not count
                     'num_gt': self.polygon_id,
                     'prediction': prediction,
                     'success': self.success}
        if self.render_ob:
            self.render_grid()
        return self.ob, 0, self.done, self.info

    def check_success(self, prediction):
        return prediction == self.polygon_id

    def render_grid(self, mode='human'):
        if not self.rendered_occupancy:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 300, 300)
            self.rendered_occupancy = True
        if mode == 'rgb_array':
            return self.occupancy_grid  # return RGB frame suitable for video
        elif mode == 'human':
            # pop up a window for visualization
            cv2.imshow('image', self.occupancy_grid[0])
            cv2.waitKey(1)
            return self.occupancy_grid
        else:
            super(RealFloatingFingerEnv, self).render(mode=mode)  # just raise an exception

    def check_collision(self):
        return False

    def collision_callback(self, msg):
        self.collision = msg.data

    def generate_heuristic_trajectory(self):
        trajectory = [(0, 0)]
        current = (0, 0)
        while current != (self.max_x_idx - 1, self.max_y_idx - 1):
            # move down
            current = mu.compute_next_loc(current, 2, self.max_x_idx, self.max_y_idx)
            trajectory.append(current)
            # move right
            current = mu.compute_next_loc(current, 1, self.max_x_idx, self.max_y_idx)
            trajectory.append(current)
        while current != (0, self.max_y_idx - 1):
            # move up
            current = mu.compute_next_loc(current, 0, self.max_x_idx, self.max_y_idx)
            trajectory.append(current)
        while current != (self.max_x_idx - 1, 0):
            # move down
            current = mu.compute_next_loc(current, 2, self.max_x_idx, self.max_y_idx)
            trajectory.append(current)
            # move left
            current = mu.compute_next_loc(current, 3, self.max_x_idx, self.max_y_idx)
            trajectory.append(current)
        return trajectory

    def generate_heuristic_moves(self):
        moves = []
        current = (0, 0)
        while current != (self.max_x_idx - 1, self.max_y_idx - 1):
            # move down
            current = mu.compute_next_loc(current, mu.down, self.max_x_idx, self.max_y_idx)
            moves.append(mu.down)
            # move right
            current = mu.compute_next_loc(current, mu.right, self.max_x_idx, self.max_y_idx)
            moves.append(mu.right)
        while current != (0, self.max_y_idx - 1):
            # move up
            current = mu.compute_next_loc(current, mu.up, self.max_x_idx, self.max_y_idx)
            moves.append(mu.up)
        while current != (self.max_x_idx - 1, 0):
            # move down
            current = mu.compute_next_loc(current, mu.down, self.max_x_idx, self.max_y_idx)
            moves.append(mu.down)
            # move left
            current = mu.compute_next_loc(current, mu.left, self.max_x_idx, self.max_y_idx)
            moves.append(mu.left)
        return moves