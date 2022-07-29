from scipy.stats import entropy
import copy
import misc_utils as mu
import cv2
cv2.ocl.setUseOpenCL(False)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from ppo_discrete import Scale, layer_init, Agent


""" Explorer takes in observations and decides where to move next. it does not control termination """


class Explorer:
    def get_move(self, obs):
        """
        obs is of shape (1, height, width) or (n, 1, height, weight).
        obs is of numpy array.
        return
            move, int or (n, ) np array
        """
        raise NotImplementedError

    def reset(self, obs):
        # reset the explorer using the initial observations
        pass


class RandomExplorer(Explorer):
    pattern = 'random'

    def __init__(self, move_dim):
        self.move_dim = move_dim

    def get_move(self, obs):
        if obs.ndim == 3:
            move = np.random.choice(range(self.move_dim))
        elif obs.ndim == 4:
            n = obs.shape[0]
            move = np.random.choice(range(self.move_dim), n)
        else:
            raise TypeError
        return move


class PPOExplorer(Explorer, nn.Module):
    pattern = 'PPO'

    def __init__(self, action_dim, device, model_path=None, frames=1, img_size=50):
        super(PPOExplorer, self).__init__()
        self.img_size = img_size
        self.agent = Agent(action_dim=action_dim, device=device, frames=frames, img_size=img_size)
        self.model_path = model_path
        if self.model_path is not None:
            self.agent.load_state_dict(torch.load(self.model_path))

    def get_move(self, obs):
        if obs.ndim == 3:
            move = self.agent.get_move_stochastic(obs[None, ...])[0]
            probs = self.agent.get_move_probabilities(obs[None, ...])[0]
            next_loc = mu.compute_next_loc(mu.get_current_loc(obs), move, height=self.img_size, width=self.img_size)
            index = 0
            while obs[0][next_loc] == mu.white:
                # move = mu.get_next_direction_clockwise(move)
                move = sorted(zip(probs, range(4)), reverse=True)[index][1]
                next_loc = mu.compute_next_loc(mu.get_current_loc(obs), move, height=self.img_size, width=self.img_size)
                index += 1
                # collision checking false, all neighbours are white
                if index == 4:
                    return np.random.choice(4)
            return move
        elif obs.ndim == 4:
            return self.agent.get_move_stochastic(obs)

    def get_move_bkup(self, obs):
        if obs.ndim == 3:
            probs = self.agent.get_move_probabilities(obs[None, ...])[0]
            for prob, move in sorted(zip(probs, range(4)), reverse=True):
                next_loc = mu.compute_next_loc(mu.get_current_loc(obs), move, height=self.img_size, width=self.img_size)
                if obs[0][next_loc] != mu.white and obs[0][next_loc] != mu.black and obs[0][next_loc] != mu.current_black:
                    return move
            for prob, move in sorted(zip(probs, range(4)), reverse=True):
                next_loc = mu.compute_next_loc(mu.get_current_loc(obs), move, height=self.img_size, width=self.img_size)
                if obs[0][next_loc] != mu.white and obs[0][next_loc] != mu.current_black:
                    return move
        elif obs.ndim == 4:
            probs = self.agent.get_move_probabilities(obs)


class AllInONeExplorer(PPOExplorer, nn.Module):
    """ This is the same as PPO explorer except action dimension """
    pattern = 'all_in_one'

    def get_move(self, obs):
        """ not considering moving into white pixels """
        if obs.ndim == 3:
            move = self.agent.get_move_stochastic(obs[None, ...])[0]
            return move
        elif obs.ndim == 4:
            return self.agent.get_move_stochastic(obs)


class EdgeFollowExplorer:
    pattern = 'edge'

    # similar to a bug algorithm
    # This explorer does not handle parallel envs, because it has to track pre move and ob for each env
    def __init__(self, img_size=60):
        super(EdgeFollowExplorer, self).__init__()
        self.img_size = img_size

        self.pre_move = None
        self.old_obs = None

    def reset(self, obs):
        if obs.ndim == 3:
            dim_0, dim_1 = np.where(obs[0] == mu.white)
            current_loc = mu.get_current_loc(obs)
            first_white_loc = (dim_0[0], dim_1[0])
            self.pre_move = mu.get_direction(current_loc, first_white_loc)
            self.old_obs = copy.deepcopy(obs)
            self.old_obs[0][first_white_loc] = mu.unexplored
        else:
            raise NotImplementedError

    def get_move_single_ob(self, old_ob, ob):
        """ old_ob and ob are of shape (1, height, width) """
        # This algorithm won't work if agent bumps into the border. It will follow the boarder as well.
        # A move that leads the agent into the walls won't exit the while loop
        move = None
        current_loc = mu.get_current_loc(ob)
        collision, collision_loc = mu.check_grid_collision(old_ob, ob)
        if collision:
            # if collision happens: turn clockwise until you can move forward (unexplored, or black), return the action
            move = mu.get_next_direction_clockwise(self.pre_move)
        else:
            # if no collision happens: starting from the first anti-clockwise move from pre move
            # turn clockwise until you can move forward (unexplored, or black), return the action
            move = mu.get_next_direction_anti_clockwise(self.pre_move)
        initial_move = move
        new_loc = mu.compute_next_loc(current_loc, move, height=self.img_size, width=self.img_size)
        while not (ob[0][new_loc] == mu.unexplored or ob[0][new_loc] == mu.black):
            move = mu.get_next_direction_clockwise(move)
            if move == initial_move:
                return np.random.choice(4)
            new_loc = mu.compute_next_loc(current_loc, move, height=self.img_size, width=self.img_size)
        return move

    def get_move(self, obs):
        if obs.ndim == 3:
            move = self.get_move_single_ob(self.old_obs, obs)
            # self.old_obs should be a separate copy instead of another name of the occupancy grid of the emv.
            # Otherwise, env.step will change self.old_obs immediately
            self.old_obs = copy.deepcopy(obs)
            self.pre_move = move
        else:
            raise NotImplementedError
        return move


class InfoGainExplorer(Explorer):
    pattern = 'info'

    """ Can only handle a single ob, so only works with a single env """
    def __init__(self,
                 discriminator):
        super(InfoGainExplorer, self).__init__()
        self.move_dim = 4
        self.discriminator = discriminator  # info gain discriminator requires a

    def get_move_single_ob(self, ob):
        assert self.discriminator is not None, 'info_gain policy requires a discriminator'
        good_moves = mu.find_not_go_back_moves(ob)
        current_loc = mu.get_current_loc(ob)
        height, width = ob[0].shape

        if len(good_moves) == 0:
            # all explored
            move = np.random.choice(range(self.move_dim))
        else:
            prediction, max_prob, probs = self.discriminator.predict(ob)
            old_entropy = entropy(probs)

            info_gains = np.zeros(len(good_moves))
            for i, move in enumerate(good_moves):
                new_loc = mu.compute_next_loc(current_loc, move, height, width)
                masks = []

                # new pixel is white
                ob_w = copy.deepcopy(ob)
                ob_w[0][new_loc] = mu.white
                probs_w = self.discriminator.predict(ob_w)[2]
                if not any(probs_w):
                    # if this particular color of the pixel makes the ob not belong to any class
                    masks.append(0)
                    entropy_w = 1
                else:
                    masks.append(1)
                    entropy_w = entropy(probs_w)

                # new pixel is black
                ob_b = copy.deepcopy(ob)
                ob_b[0][new_loc] = mu.black
                probs_b = self.discriminator.predict(ob_b)[2]
                if not any(probs_b):
                    masks.append(0)
                    entropy_b = 1
                else:
                    masks.append(1)
                    entropy_b = entropy(probs_b)

                weights = np.array(masks) / np.array(masks).sum()
                avg_entropy = weights[0] * entropy_w + weights[1] * entropy_b
                info_gains[i] = old_entropy - avg_entropy

            # print(info_gains)
            if np.all(info_gains == info_gains[0]):
                move = np.random.choice(good_moves)
            else:
                move_idx = np.argmax(info_gains)
                move = good_moves[move_idx]
        return move

    def get_move(self, obs):
        if obs.ndim == 3:
            return self.get_move_single_ob(obs)
        else:
            raise NotImplementedError


class NotGoBackExplorer(Explorer):
    pattern = 'not_go_back'

    def __init__(self):
        super(NotGoBackExplorer, self).__init__()
        self.move_dim = 4

    def get_move_single_ob(self, ob):
        good_moves = mu.find_not_go_back_moves(ob)
        if len(good_moves) == 0:
            # all explored
            move = np.random.choice(range(self.move_dim))
        else:
            move = np.random.choice(good_moves)
        return move

    def get_move(self, obs):
        if obs.ndim == 3:
            return self.get_move_single_ob(obs)
        else:
            raise NotImplementedError
