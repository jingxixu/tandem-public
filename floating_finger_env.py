import time

import gym
import numpy as np
import cv2
import copy
import misc_utils as mu
import pybullet as p
import pybullet_utils as pu
import math
import os
from math import radians
import itertools
from gym.utils import seeding


class FloatingFingerEnv(gym.Env):
    def __init__(self,
                 max_ep_len=2000,
                 max_x=0.3,
                 max_y=0.3,
                 finger_urdf_path='assets/simplified_finger_urdf/urdf/roam_distal.urdf',
                 dataset='extruded_polygons_r_0.1_s_8_h_0.05',
                 step_size=0.005,
                 object_scale=1.0,
                 finger_height=0.05 + 0.05 + 0.0185 * 0.25,
                 start_on_border=True,
                 reward_type='sparse',
                 reward_scale=1.0,
                 num_orientations=20,
                 translate=False,
                 translate_range=0.01,
                 render_pybullet=False,
                 render_ob=False,
                 debug=False,
                 use_correctness=False,
                 exp_knob=None,
                 threshold=0.98,
                 sensor_noise=0,
                 env_id=0):
        self.seed()
        # the height of the finger is ~0.095
        self.finger_height = finger_height
        self.max_ep_len = max_ep_len
        self.max_x = max_x
        self.max_y = max_y
        self.step_size = step_size
        # discretize the workspace
        self.max_x_idx = round(self.max_x / self.step_size)
        self.max_y_idx = round(self.max_y / self.step_size)
        self.finger_urdf_path = finger_urdf_path
        self.env_id = env_id
        self.object_scale = object_scale
        self.use_correctness = use_correctness
        self.num_classes = 10
        self.move_dim = len(mu.move_map)
        # self.action_space = gym.spaces.Discrete(self.action_dim)
        self.action_space = gym.spaces.Dict({"move": gym.spaces.Discrete(self.move_dim),
                                             "prediction": gym.spaces.Discrete(self.num_classes),
                                             "probs": gym.spaces.Box(low=0, high=1, shape=(self.num_classes,)),
                                             "max_prob": gym.spaces.Box(low=0, high=1, shape=(1,)),
                                             "done": gym.spaces.Discrete(2)})
        self.observation_space = gym.spaces.Box(low=np.zeros((1, self.max_x_idx, self.max_y_idx)),
                                                high=np.full((1, self.max_x_idx, self.max_y_idx), 255), dtype=np.uint8)
        self.start_on_border = start_on_border
        self.reward_type = reward_type
        self.reward_scale = reward_scale
        self.exp_knob = exp_knob
        self.prob_mapping_function = mu.get_prob_mapping_function(reward_type, 0.1, threshold, exp_knob)
        self.num_orientations = num_orientations
        self.translate = translate
        self.translate_range = translate_range
        self.render_ob = render_ob
        self.render_pybullet = render_pybullet
        self.debug = debug
        self.finger_initial_position = [0, 0, self.finger_height]
        self.finger_initial_quaternion = pu.quaternion_from_euler([math.pi, 0, 0])
        self.polygon_initial_quaternion = [0, 0, 0, 1]
        self.waitlist_position = [-1, -1, 0]
        self.sensor_noise = sensor_noise

        # step related info
        self.gt_grids = None
        self.rendered_occupancy = False
        self.current_step = 0
        self.current_loc = None
        self.polygon_id = None
        self.angle = None
        self.polygon_initial_position = None
        # y is the world y axis (in simulation), y is the width of the image, the second axis of the numpy array
        # x is the world x axis (in simulation), x is the height of the image, the first axis of the numpy array
        self.occupancy_grid = np.full((1, self.max_x_idx, self.max_y_idx), mu.unexplored, dtype=np.uint8)
        self.ob = None
        self.done = None
        self.info = None
        self.reward = 0
        self.success = None
        self.discover = None
        self.max_prob = 0.1
        self.initial_explored_pixel = None

        self.dataset_path = os.path.join('assets', 'datasets', dataset)
        self.object_urdf_folder = os.path.join(self.dataset_path, 'vhacd_urdfs')

        self.client_id = pu.configure_pybullet(rendering=render_pybullet, debug=self.debug,
                                               target=(max_x / 2, max_y / 2, 0.05), dist=0.6)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=0,
                                     cameraTargetPosition=(max_x / 2, max_y / 2, 0.05))
        time.sleep(2)
        # if this part goes to reset, it becomes 100X slower!
        p.resetSimulation(physicsClientId=self.client_id)
        self.finger = FloatingFingerController([self.finger_initial_position, self.finger_initial_quaternion],
                                               self.finger_urdf_path,
                                               self.client_id)
        # load all the polygons at a waiting location
        self.polygons = []  # list of pybullet object ids
        for i in range(self.num_classes):
            object_urdf_path = os.path.join(self.object_urdf_folder, f'{i}.urdf')
            object = p.loadURDF(object_urdf_path,
                                basePosition=self.waitlist_position,
                                baseOrientation=[0, 0, 0, 1],
                                globalScaling=self.object_scale,
                                useFixedBase=True,
                                physicsClientId=self.client_id)
            self.polygons.append(object)
        # for measuring the distance from the finger tip to the plane
        # self.plane = p.loadURDF("plane.urdf")
        # closest_point = pu.get_closest_potins(self.finger.id, self.plane, 1.0, -1, -1)[0]
        # print(closest_point.contactDistance)

    def check_collision(self, object_id=None):
        flip = False
        if self.sensor_noise > 0:
            random_n = self.random_nums[self.collision_cnt]
            self.collision_cnt += 1
            if random_n >= 1 - self.sensor_noise:
                flip = True
            else:
                flip = False
        object_id = object_id if object_id is not None else self.polygons[self.polygon_id]
        closest_points = pu.get_closest_potins(self.finger.id, object_id, 0, -1, -1, client=self.client_id)
        if len(closest_points) > 0:
            return True if not flip else False
        else:
            return False if not flip else True

    def step(self, action):
        move = action['move']
        prediction = action['prediction']
        max_prob = action['max_prob']
        probs = action['probs']
        done = action['done']

        # print(f'{self.client_id}: {action}, current loc: {self.current_loc}')

        num_explored = np.count_nonzero(self.occupancy_grid != mu.unexplored)
        new_loc = self.compute_next_loc(move)

        if new_loc != self.current_loc:
            new_pose = [self.get_position_from_loc(new_loc), self.finger_initial_quaternion]
            self.finger.set_pose_no_control(new_pose)
            if self.check_collision():
                # contact happens
                self.occupancy_grid[0][new_loc] = mu.white
                new_loc = self.current_loc
                new_pixel = mu.white
            else:
                # change the pixel at current location assuming the agent has left
                self.occupancy_grid[0][self.current_loc] = mu.black if self.occupancy_grid[0][self.current_loc] \
                                                                       == mu.current_black else mu.white
                # reveal pixel at new location, assuming the agent is on the new location
                self.occupancy_grid[0][new_loc] = mu.current_black
                new_pixel = mu.black

        self.discover = True if np.count_nonzero(self.occupancy_grid != mu.unexplored) > num_explored else False
        self.current_loc = new_loc
        self.current_step += 1

        self.success = self.check_success(prediction)
        self.done = done or self.current_step >= self.max_ep_len

        if (not self.use_correctness and max_prob > self.max_prob) or \
                (self.use_correctness and max_prob > self.max_prob and prediction == self.polygon_id):
            old_mapped_prob = self.prob_mapping_function(self.max_prob)
            mapped_prob = self.prob_mapping_function(max_prob)
            self.reward = mapped_prob - old_mapped_prob
            self.max_prob = max_prob
        else:
            self.reward = 0
        self.reward = self.reward * self.reward_scale

        self.ob = copy.deepcopy(self.occupancy_grid)
        self.info = {'discover': self.discover,
                     'ob': self.ob,
                     'num_explored_pixels': np.count_nonzero(
                         self.occupancy_grid != mu.unexplored) - self.initial_explored_pixel,
                     'num_gt': self.polygon_id,
                     'prediction': action['prediction'],
                     'success': self.success,
                     'angle': self.angle}
        if self.render_ob:
            self.render_grid()
        return self.ob, self.reward, self.done, self.info

    def reset(self, polygon_id=None, angle=None):
        if self.polygon_id is not None:
            pu.set_point(self.polygons[self.polygon_id], self.waitlist_position, client=self.client_id)
        mu.draw_workspace_xy([0, 0], [self.max_x, self.max_y], z=0)
        self.done = False
        self.info = None
        self.reward = 0
        self.success = False
        self.discover = True
        self.current_step = 0
        self.max_prob = 0.1
        self.discover = True
        self.occupancy_grid = np.full((1, self.max_x_idx, self.max_y_idx), mu.unexplored, dtype=np.uint8)
        self.polygon_id = self.np_random.randint(low=0, high=10) if polygon_id is None else polygon_id
        self.finger.set_pose_no_control([self.finger_initial_position, self.finger_initial_quaternion])
        # different methods have different number of actions, calling the check_collision function different number of times,
        # then calling random choice different number of times
        # doing it this way make sure the sensor error happens at the same time (not the same location) for each episode across different models
        self.random_nums = self.np_random.uniform(size=3000)
        self.collision_cnt = 0

        # object orientation
        if angle is not None:
            self.angle = angle
        else:
            if self.num_orientations == -1:
                self.angle = self.np_random.uniform(low=0, high=360)
            else:
                gap = int(360 / self.num_orientations)
                angles = [0 + i * gap for i in range(self.num_orientations)]
                angle_i = self.np_random.choice(range(self.num_orientations))
                self.angle = angles[angle_i]
        euler = (0, 0, radians(self.angle))
        self.polygon_initial_quaternion = pu.quaternion_from_euler(euler)
        # object position
        if self.translate:
            self.polygon_initial_position = self.sample_polygon_position()
        else:
            self.polygon_initial_position = [self.max_x / 2, self.max_y / 2, 0]
        pu.set_pose(self.polygons[self.polygon_id], (self.polygon_initial_position, self.polygon_initial_quaternion), client=self.client_id)

        # the finger always starts at location (0, 0), this location is guaranteed to be collision-free
        old_loc = (0, 0)
        new_loc = (0, 0)
        new_pose = [self.get_position_from_loc(new_loc), self.finger_initial_quaternion]
        self.finger.set_pose_no_control(new_pose)

        if self.start_on_border:
            # always starts on boarder
            trajectory = self.generate_heuristic_trajectory()
            num_moves = 1
            collision = False
            while not collision:
                old_loc = new_loc
                self.occupancy_grid[0][old_loc] = mu.black
                new_loc = trajectory[num_moves]
                new_pose = [self.get_position_from_loc(new_loc), self.finger_initial_quaternion]
                self.finger.set_pose_no_control(new_pose)
                collision = self.check_collision()
                num_moves += 1
        self.occupancy_grid[0][new_loc] = mu.white
        self.occupancy_grid[0][old_loc] = mu.current_black
        self.current_loc = old_loc
        self.occupancy_grid[0][self.current_loc] = mu.current_black

        self.finger.set_pose_no_control([self.get_position_from_loc(self.current_loc), self.finger_initial_quaternion])
        self.ob = copy.deepcopy(self.occupancy_grid)
        self.initial_explored_pixel = np.count_nonzero(self.occupancy_grid != mu.unexplored)
        if self.render_ob:
            self.render_grid()

        # print(f'env id: {self.env_id}\t client id: {self.client_id}\t polygon id: {self.polygon_id}\t angle: {self.angle}')
        return self.ob

    def generate_gt_grids(self):
        # use the sampled initial position and orientation to obtain the ground truth grids
        grids = np.zeros((self.num_classes, self.max_x_idx, self.max_y_idx), dtype=np.uint8)
        for i in range(self.num_classes):
            object_urdf_path = os.path.join(self.object_urdf_folder, f'{i}.urdf')
            object = p.loadURDF(object_urdf_path,
                                basePosition=self.polygon_initial_position,
                                baseOrientation=self.polygon_initial_quaternion,
                                globalScaling=self.object_scale,
                                useFixedBase=True,
                                physicsClientId=self.client_id)
            for loc in list(itertools.product(range(self.max_x_idx), range(self.max_y_idx))):
                new_pose = [self.get_position_from_loc(loc), self.finger_initial_quaternion]
                self.finger.set_pose_no_control(new_pose)
                grids[i][loc] = mu.white if self.check_collision(object_id=object) else mu.black
            # mu.show_gray(grids[i])
            pu.remove_body(object, client=self.client_id)
        return grids

    def compute_occupancy_grid(self):
        # save current finger loc
        current_loc = self.current_loc

        # white if occupied, black otherwise
        grid = np.zeros((self.max_x_idx, self.max_y_idx), dtype=np.uint8)
        # white if border, black otherwise
        grid_border = np.zeros((self.max_x_idx, self.max_y_idx), dtype=np.uint8)
        # white if border, red if neighbor to border, otherwise black
        rgb = np.zeros((self.max_x_idx, self.max_y_idx, 3), dtype=np.uint8)
        # key: border locations, item: outter neighbors to the this location
        border_neighbors = {}

        old_loc = (0, 0)
        pre_collision = False
        # go in the x-axis direction
        for x in range(self.max_x_idx):
            for y in range(self.max_y_idx):
                new_loc = (x, y)
                new_pose = [self.get_position_from_loc(new_loc), self.finger_initial_quaternion]
                self.finger.set_pose_no_control(new_pose)
                now_collision = self.check_collision()
                if now_collision:
                    grid[new_loc] = mu.white
                # entering object
                if now_collision and not pre_collision:
                    grid_border[new_loc] = mu.white
                    if new_loc not in border_neighbors.keys():
                        border_neighbors[new_loc] = []
                    border_neighbors[new_loc].append(old_loc)
                # leaving object
                if not now_collision and pre_collision:
                    grid_border[old_loc] = mu.white
                    if old_loc not in border_neighbors.keys():
                        border_neighbors[old_loc] = []
                    border_neighbors[old_loc].append(new_loc)
                pre_collision = now_collision
                old_loc = new_loc

        old_loc = (0, 0)
        pre_collision = False
        # go in the y-axis direction
        for y in range(self.max_y_idx):
            for x in range(self.max_x_idx):
                new_loc = (x, y)
                new_pose = [self.get_position_from_loc(new_loc), self.finger_initial_quaternion]
                self.finger.set_pose_no_control(new_pose)
                now_collision = self.check_collision()
                if now_collision:
                    grid[new_loc] = mu.white
                # entering object
                if now_collision and not pre_collision:
                    grid_border[new_loc] = mu.white
                    if new_loc not in border_neighbors.keys():
                        border_neighbors[new_loc] = []
                    border_neighbors[new_loc].append(old_loc)
                # leaving object
                if not now_collision and pre_collision:
                    grid_border[old_loc] = mu.white
                    if old_loc not in border_neighbors.keys():
                        border_neighbors[old_loc] = []
                    border_neighbors[old_loc].append(new_loc)
                pre_collision = now_collision
                old_loc = new_loc

        # verify the border neighbors
        rgb = np.tile(grid_border, (3, 1, 1))
        rgb = np.transpose(rgb, [1, 2, 0])
        for loc, neighbors in border_neighbors.items():
            for n in neighbors:
                rgb[n] = [255, 0, 0]

        # mu.show_img(rgb)
        # mu.show_img(grid)
        # mu.show_img(grid_border)

        # recover finger location
        pose = [self.get_position_from_loc(current_loc), self.finger_initial_quaternion]
        self.finger.set_pose_no_control(pose)
        return grid, grid_border, rgb, border_neighbors

    def check_done(self, move):
        return move == mu.done

    def check_success(self, prediction):
        return prediction == self.polygon_id

    def get_position_from_loc(self, loc):
        return [loc[0] * self.step_size, loc[1] * self.step_size, self.finger_height]

    def calculate_new_pose(self, move):
        current_position, current_orn = self.finger.get_pose()
        current_orn = self.finger_initial_quaternion
        if move == 0:
            # y + 1
            current_position[1] = current_position[1] + self.step_size \
                if current_position[1] + self.step_size <= self.max_y else current_position[1]
        elif move == 1:
            # x - 1
            current_position[0] = current_position[0] - self.step_size \
                if current_position[0] - self.step_size >= 0 else current_position[0]
        elif move == 2:
            # y - 1
            current_position[1] = current_position[1] - self.step_size \
                if current_position[1] - self.step_size >= 0 else current_position[1]
        elif move == 3:
            # x + 1
            current_position[0] = current_position[0] + self.step_size \
                if current_position[0] + self.step_size <= self.max_x else current_position[0]
        else:
            raise ValueError('unrecognized move')
        return [current_position, current_orn]

    def compute_next_loc(self, move):
        return mu.compute_next_loc(self.current_loc, move, self.max_x_idx, self.max_y_idx)

    def render_grid(self, mode='human'):
        if not self.rendered_occupancy:
            cv2.namedWindow('image' + str(self.env_id), cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image' + str(self.env_id), 300, 300)
            self.rendered_occupancy = True
        if mode == 'rgb_array':
            return self.occupancy_grid  # return RGB frame suitable for video
        elif mode == 'human':
            # pop up a window for visualization
            cv2.imshow('image' + str(self.env_id), self.occupancy_grid[0])
            cv2.waitKey(1)
            return self.occupancy_grid
        else:
            super(FloatingFingerEnv, self).render(mode=mode)  # just raise an exception

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_polygon_position(self):
        x = self.np_random.uniform(low=self.max_x / 2 - self.translate_range / 2,
                                   high=self.max_x / 2 + self.translate_range / 2)
        y = self.np_random.uniform(low=self.max_y / 2 - self.translate_range / 2,
                                   high=self.max_y / 2 + self.translate_range / 2)
        return [x, y, 0]

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


class FloatingFingerController:
    def __init__(self, initial_pose, urdf_path, client_id):
        self.initial_pose = initial_pose
        self.urdf_path = urdf_path
        self.client_id = client_id
        self.id = p.loadURDF(self.urdf_path, initial_pose[0], initial_pose[1], physicsClientId=self.client_id)

        self.cid = p.createConstraint(parentBodyUniqueId=self.id, parentLinkIndex=-1, childBodyUniqueId=-1,
                                      childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                      parentFramePosition=[0, 0, 0], childFramePosition=initial_pose[0],
                                      childFrameOrientation=initial_pose[1],
                                      physicsClientId=self.client_id)

    def set_pose_no_control(self, pose):
        pu.set_pose(self.id, pose, client=self.client_id)

    def set_pose(self, pose):
        pu.set_pose(self.id, pose, client=self.client_id)
        self.control_pose(pose)

    def get_pose(self):
        return pu.get_body_pose(self.id, client=self.client_id)

    def control_pose(self, pose):
        p.changeConstraint(self.cid, jointChildPivot=pose[0], jointChildFrameOrientation=pose[1],
                           physicsClientId=self.client_id)
