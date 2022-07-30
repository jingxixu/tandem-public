# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py

import misc_utils as mu
from collections import deque
import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from ppo_discrete import Agent, VecPyTorch, linear_schedule
from discriminator import LearnedDiscriminator, GroundTruthDiscriminator
from discriminator_dataset import VariedMNISTDataset
import tqdm
import pprint
import dowel
from dowel import logger, tabular
import socket
import torchvision.transforms as T
from floating_finger_env import FloatingFingerEnv


""" Train the discriminator with a fixed explorer, only a single env is created """


def get_args():
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--env_name', type=str, default="floating_finger",
                        help='the id of the gym environment')
    parser.add_argument('--discriminator_lr', type=float, default=0.001,
                        help='the learning rate of the discriminator')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total_timesteps', type=int, default=100000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch_deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod_mode', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--wandb_project_name', type=str, default="ppo",
                        help="the wandb's project name")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Saves a policy every save_interval episodes (default: 10)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log a policy every log_interval episodes (default: 10)')
    parser.add_argument('--save_discriminator_data', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)

    # Algorithm specific arguments
    # fixed explorers cannot work with multiple envs
    parser.add_argument('--num_envs', type=int, default=1,
                        help='the number of parallel game environment')

    # my arguments
    # important arguments other than these: exp_name, prod_mode
    parser.add_argument('--render_pybullet', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--render_ob', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--discriminator', type=str)
    parser.add_argument('--discriminator_path', type=str)
    parser.add_argument('--explorer_path', type=str)
    parser.add_argument('--train_discriminator', type=lambda x: bool(strtobool(x)), default=False, nargs='?',
                        const=True)
    parser.add_argument('--max_ep_len', type=int, default=2000)
    parser.add_argument('--buffer_size', type=int, default=150000)
    parser.add_argument('--explorer_steps', type=int, default=200000)
    parser.add_argument('--tactile_sim', action='store_true', default=False)
    parser.add_argument('--add_prob', type=float, default=1.0)
    parser.add_argument('--save_length', type=float, default=2000)
    parser.add_argument('--save_success', type=float, default=0.0)
    parser.add_argument('--terminal_confidence', type=float, default=1.0)
    parser.add_argument('--start_on_border', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--discriminator_epochs', type=int, default=15)
    parser.add_argument('--running_stat_len', type=int, default=100)
    parser.add_argument('--reward_type', type=str, default='sparse')
    parser.add_argument('--reward_scale', type=float, default=1)
    parser.add_argument('--exp_knob', type=int)
    parser.add_argument('--num_orientations', type=int, default=10)
    parser.add_argument('--num_rotations', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='extruded_polygons_r_0.1_s_8_h_0.05', help='the dataset to use')
    parser.add_argument('--use_correctness', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--collect_initial_batch', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='whether we collect an initial batch of data to train the discriminator before updating the explorer')
    parser.add_argument('--initial_batch_ep_len', type=float, default=2000)
    parser.add_argument('--initial_batch_policy', type=str, default='random')
    parser.add_argument('--rotate', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--translate', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--translate_range', type=float, default=0.05)
    parser.add_argument('--sensor_noise', type=float, default=0)
    parser.add_argument('--explorer_type', type=str, default="edge")

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.exp_name = f'{args.exp_name}_{socket.gethostname()}_{args.timestr}'
    args.save_dir = os.path.join(args.save_dir, args.exp_name)
    args.log_dir = os.path.join(args.log_dir, args.exp_name)

    return args


def make_env(env_name, env_id, seed, max_x, max_y, step_size, scale, finger_height, use_correctness):
    def thunk():
        env = FloatingFingerEnv(
            env_id=env_id,
            render_pybullet=args.render_pybullet,
            render_ob=args.render_ob,
            debug=args.debug,
            reward_type=args.reward_type,
            reward_scale=args.reward_scale,
            exp_knob=args.exp_knob,
            threshold=args.terminal_confidence,
            start_on_border=args.start_on_border,
            num_orientations=args.num_orientations,
            translate=args.translate,
            translate_range=args.translate_range,
            max_x=max_x,
            max_y=max_y,
            step_size=step_size,
            object_scale=scale,
            finger_height=finger_height,
            dataset=args.dataset,
            use_correctness=use_correctness,
            sensor_noise=args.sensor_noise
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def train_discdriminator_f():
    global next_obs
    global discriminator_data_batch
    global discriminator_path
    global discriminator_train_loss
    global discriminator_train_acc
    global discriminator_test_loss
    global discriminator_test_acc
    if args.train_discriminator:
        if args.collect_initial_batch and discriminator_data_batch == 0:
            # collect and train on first batch of data
            pbar = tqdm.tqdm(total=varied_dataset.buffer_size)
            while len(varied_dataset) < varied_dataset.buffer_size:
                # TODO further verify list of dict or dict of list: look at source code, infos is a tuple of dicts
                # using the initialized policy instead of the random policy to collect data
                if args.initial_batch_policy == 'random':
                    move = [random.choice([0, 1, 2, 3]) for i in range(args.num_envs)]
                    current_steps = envs.get_attr('current_step')
                    done = [1 if current_steps[i] == args.initial_batch_ep_len else 0 for i in range(args.num_envs)]
                elif args.initial_batch_policy == 'agent':
                    # TODO the initial policy can only be random. agent is not tested / modified correctly
                    move = explorer.get_move(next_obs)[0]
                    move = [i.item() for i in move]
                    current_steps = envs.get_attr('current_step')
                    done = [1 if current_steps[i] == args.initial_batch_ep_len else 0 for i in range(args.num_envs)]
                else:
                    raise TypeError('unrecognized initial batch policy')
                action = [{'move': move[i],
                           'prediction': 0,
                           'done': done[i],
                           'max_prob': 0.1,
                           'probs': [0.1] * 10}
                          for i in range(args.num_envs)]
                next_obs, rs, ds, infos = envs.step(action)
                for (i, info, done) in zip(range(len(infos)), infos, ds):
                    if info['discover']:
                        # only next_obs is from the reset of the next episode and reset only returns obs
                        # without info
                        imgs = mu.generate_rotated_imgs(mu.get_discriminator_input(info['ob']),
                                                        num_rotations=args.num_rotations)
                        # imgs = mu.rotate_imgs(imgs, [-info['angle']])
                        varied_dataset.add_data(imgs,
                                                [info['num_gt']] * args.num_rotations)
                        pbar.update(args.num_rotations)
                    if len(varied_dataset) == varied_dataset.buffer_size:
                        break
            pbar.close()
            # reset the next_obs so that the RL training does not start with highly revealed observations from the random policy
            next_obs = envs.reset()
            explorer.reset(next_obs[0].cpu().numpy())
        if len(varied_dataset) >= varied_dataset.buffer_size and (env_step - 1) % explorer_env_step == 0:
            # train discriminator
            logger.log(str(len(varied_dataset)))
            logger.log(f'discriminator data batch: {discriminator_data_batch}')
            folder_name = f'discriminator_batch_{discriminator_data_batch:04d}'
            pixel_freq = mu.compute_pixel_freq(varied_dataset.imgs, visualize=False, save=True,
                                               save_path=os.path.join(args.save_dir, folder_name, 'data',
                                                                      'pixel_freq.png'))
            # set path for learning to save checkpoint
            discriminator.save_dir = os.path.join(args.save_dir, folder_name)
            if args.save_discriminator_data:
                varied_dataset.export_data(os.path.join(args.save_dir, folder_name, 'data'))
            train_loader, test_loader = mu.construct_loaders(dataset=varied_dataset, split=0.2)
            # always train 15 epochs for the first discriminator
            discriminator_path, discriminator_train_loss, discriminator_train_acc, discriminator_test_loss, discriminator_test_acc, stats = discriminator.learn(
                epochs=15 if discriminator_data_batch == 0 else args.discriminator_epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                logger=logger)

            # write discriminator stats
            for i, stat in enumerate(stats):
                writer.add_scalar('discriminator/train_loss', stat['train_loss'],
                                  discriminator_data_batch * args.discriminator_epochs + i)
                writer.add_scalar('discriminator/train_acc', stat['train_acc'],
                                  discriminator_data_batch * args.discriminator_epochs + i)
                writer.add_scalar('discriminator/test_loss', stat['test_loss'],
                                  discriminator_data_batch * args.discriminator_epochs + i)
                writer.add_scalar('discriminator/test_acc', stat['test_acc'],
                                  discriminator_data_batch * args.discriminator_epochs + i)

                if args.prod_mode:
                    data_to_log = {
                        'discriminator/train_loss': stat['train_loss'],
                        'discriminator/train_acc': stat['train_acc'],
                        'discriminator/test_loss': stat['test_loss'],
                        'discriminator/test_acc': stat['test_acc'],
                        'discriminator_batch': discriminator_data_batch * args.discriminator_epochs + i,
                    }
                    wandb.log(data_to_log)
            discriminator_data_batch += 1


def add_data_f():
    if args.train_discriminator:
        for (i, info, done) in zip(range(len(infos)), infos, ds):
            if info['discover'] and random.random() <= args.add_prob:
                # only next_obs is from the reset of the next episode and reset only returns obs
                # without info
                imgs = mu.generate_rotated_imgs(mu.get_discriminator_input(info['ob']),
                                                num_rotations=args.num_rotations)
                # imgs = mu.rotate_imgs(imgs, [-info['angle']])
                varied_dataset.add_data(imgs, [info['num_gt']] * args.num_rotations)


def write_explorer_log():
    global episode
    global min_running_length
    global max_success_rate
    for info in infos:
        if 'episode' in info.keys():
            explorer.reset(next_obs[0].cpu().numpy())
            episode += 1
            # episode stats wrapper uses numpy.float32 which is not json serializable
            episode_reward = float(info['episode']['r'])
            episode_length = float(info['episode']['l'])
            episode_success = info['success']
            episode_reward_queue.append(episode_reward)
            episode_length_queue.append(episode_length)
            episode_success_queue.append(episode_success)

            # the queue is full now, compute the running stats
            if len(episode_reward_queue) == args.running_stat_len:
                running_reward = np.array(episode_reward_queue).mean()
                running_length = np.array(episode_length_queue).mean()
                running_success = np.array(episode_success_queue).mean()
                logger.log(f"global_step={global_step}, episode={episode}, reward={episode_reward:.6f}, "
                           f"length={episode_length}, success={episode_success}, running_r={running_reward:.2f}, "
                           f"running_l={running_length:.2f}, running_s={running_success:.2f}")

                # logging
                if episode % args.log_interval == 0:
                    writer.add_scalar("charts/running_reward", running_reward, global_step)
                    writer.add_scalar("charts/running_length", running_length, global_step)
                    writer.add_scalar("charts/running_success", running_success, global_step)

                    if args.prod_mode:
                        data_to_log = {
                            "charts/running_reward": running_reward,
                            "charts/running_length": running_length,
                            "charts/running_success": running_success,
                            "global_step": global_step
                        }
                        wandb.log(data_to_log)

                # saving
                if episode % args.save_interval == 0 and running_length <= args.save_length and running_success >= args.save_success\
                        and (running_success >= max_success_rate or running_length <= min_running_length):
                    training_time = mu.convert_second(time.time() - start_time)
                    folder_name = f"episode_{episode:08d}_rl_{running_length:04.2f}"
                    folder_path = os.path.join(args.save_dir, folder_name)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    checkpoint_metadata = {
                        'steps': global_step,
                        'episode': episode,
                        'env_step': env_step,
                        'episode_reward': episode_reward,
                        'episode_length': episode_length,
                        'episode_success': bool(episode_success),  # convert numpy.bool__ to bool
                        'running_reward': running_reward,
                        'running_length': running_length,
                        'running_success': running_success,
                        'training_time': training_time,
                        'discriminator_path': discriminator_path,
                        'discriminator_train_loss': discriminator_train_loss,
                        'discriminator_train_acc': discriminator_train_acc,
                        'discriminator_test_loss': discriminator_test_loss,
                        'discriminator_test_acc': discriminator_test_acc,
                        'args': vars(args)}

                    mu.save_json(checkpoint_metadata, os.path.join(folder_path, 'metadata.json'))
                    logger.log("----------------------------------------")
                    if args.train_discriminator:
                        logger.log(f'Using discriminator path {discriminator_path}')
                    logger.log(f"Training time: {training_time}")
                    logger.log("----------------------------------------")
                    min_running_length = running_length
                    max_success_rate = running_success
            else:
                # otherwise print out info without running stats
                logger.log(f"global_step={global_step}, episode={episode}, reward={episode_reward:.6f}, "
                           f"length={episode_length}, success={episode_success}")


if __name__ == "__main__":
    args = get_args()
    mu.save_command(os.path.join(args.save_dir, 'command.txt'))
    # adding dowel output
    logger.add_output(dowel.StdOutput(with_timestamp=False))
    logger.add_output(dowel.TextOutput(os.path.join(args.save_dir, 'logs.txt'), with_timestamp=False))

    logger.log('\n')
    logger.log(pprint.pformat(vars(args), indent=4))
    logger.log('\n')

    # environment scale
    if args.dataset == "extruded_polygons":
        # the height of the polygon is 0.5 * 0.4 = 0.2
        max_x = 1.0
        max_y = 1.0
        step_size = 0.02
        scale = 0.4
        finger_height = 0.5 * 0.4 + 0.085
    else:
        # the height of the finger is ~0.095, the height of the polygon is 0.05
        max_x = 0.3
        max_y = 0.3
        step_size = 0.005
        scale = 1.0
        finger_height = 0.05 + 0.05 + 0.0185 * 0.25
    height = round(max_x / step_size)
    width = round(max_y / step_size)

    # TRY NOT TO MODIFY: setup the environment
    if args.prod_mode:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=False, config=vars(args),
                   name=args.exp_name, monitor_gym=False, save_code=True)
        wandb.save(os.path.join(args.save_dir, 'command.txt'), policy='now', base_path=args.save_dir)
    writer = SummaryWriter(f"{args.log_dir}")
    writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    # TRY NOT TO MODIFY: seeding
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # construct discriminator and the dataset
    discriminator = mu.construct_discriminator(discriminator_type=args.discriminator,
                                               height=height,
                                               width=width,
                                               discriminator_path=args.discriminator_path,
                                               lr=args.discriminator_lr)
    varied_dataset = VariedMNISTDataset(buffer_size=args.buffer_size, height=height, width=width)
    varied_dataset.clean_data()


    envs = VecPyTorch(
        DummyVecEnv([make_env(args.env_name, i, args.seed + i, max_x, max_y, step_size, scale, finger_height, args.use_correctness)
                     for i in range(args.num_envs)]), device)
    assert isinstance(envs.action_space['move'], Discrete), "only discrete action space is supported"

    explorer = mu.construct_explorer(args.explorer_type, image_size=height, explorer_path=None, discriminator=discriminator)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs = envs.reset()
    explorer.reset(next_obs[0].cpu().numpy())

    episode = 0
    episode_reward_queue = deque(maxlen=args.running_stat_len)
    episode_length_queue = deque(maxlen=args.running_stat_len)
    episode_success_queue = deque(maxlen=args.running_stat_len)
    min_running_length = 1000000
    max_success_rate = 0
    start_time = time.time()
    discriminator_data_batch = 0
    discriminator_path = None
    discriminator_train_loss = None
    discriminator_train_acc = None
    discriminator_test_loss = None
    discriminator_test_acc = None


    num_env_step = args.total_timesteps // args.num_envs
    explorer_env_step = args.explorer_steps // args.num_envs
    for env_step in range(1, num_env_step + 1):
        train_discdriminator_f()
        global_step += 1 * args.num_envs
        move = explorer.get_move(next_obs[0].cpu().numpy())

        # STEP!
        # build the actions to the envs
        prediction, max_prob, probs = discriminator.predict(next_obs[0].cpu().numpy())

        # action is a list of dictionary
        action = [{'move': move,
                   'prediction': prediction,
                   'max_prob': max_prob,
                   'probs': probs,
                   'done': 1 if max_prob >= args.terminal_confidence else 0
                   } for i in range(args.num_envs)]

        next_obs, rs, ds, infos = envs.step(action)
        add_data_f()

        # write log
        write_explorer_log()
        logger.dump_all()

    envs.close()
    writer.close()
    logger.remove_all()

