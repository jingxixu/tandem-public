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


def get_args():
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--env_name', type=str, default="floating_finger",
                        help='the id of the gym environment')
    parser.add_argument('--explorer_lr', type=float, default=0.001,
                        help='the learning rate of the explorer')
    parser.add_argument('--discriminator_lr', type=float, default=0.001,
                        help='the learning rate of the discriminator')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total_timesteps', type=int, default=1000000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch_deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod_mode', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--wandb_project_name', type=str, default="tandem",
                        help="the wandb's project name")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Saves a policy every save_interval episodes (default: 10)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log a policy every log_interval episodes (default: 10)')
    parser.add_argument('--save_discriminator_data', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)

    # Algorithm specific arguments
    parser.add_argument('--n_minibatch', type=int, default=4,
                        help='the number of mini batch')
    parser.add_argument('--num_envs', type=int, default=16,
                        help='the number of parallel game environment')
    parser.add_argument('--num_steps', type=int, default=128,
                        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent_coef', type=float, default=0.05,
                        help="coefficient of the entropy")
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip_coef', type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update_epochs', type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument('--kle_stop', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle_rollback', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target_kl', type=float, default=0.03,
                        help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Use GAE for advantage computation')
    parser.add_argument('--norm_adv', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggles advantages normalization")
    parser.add_argument('--anneal_lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip_vloss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

    # my arguments
    # important arguments other than these: exp_name, prod_mode
    parser.add_argument('--render_pybullet', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--render_ob', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--multiprocess', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--discriminator', type=str)
    parser.add_argument('--discriminator_path', type=str)
    parser.add_argument('--explorer_path', type=str)
    parser.add_argument('--train_discriminator', type=lambda x: bool(strtobool(x)), default=False, nargs='?',
                        const=True)
    parser.add_argument('--max_ep_len', type=int, default=2000)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--explorer_steps', type=int, default=200000)
    parser.add_argument('--tactile_sim', action='store_true', default=False)
    parser.add_argument('--add_prob', type=float, default=1.0)
    parser.add_argument('--save_length', type=float, default=2000)
    parser.add_argument('--save_success', type=float, default=0.0)
    parser.add_argument('--terminal_confidence', type=float, default=0.98)
    parser.add_argument('--start_on_border', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--discriminator_epochs', type=int, default=15)
    parser.add_argument('--running_stat_len', type=int, default=100)
    parser.add_argument('--reward_type', type=str, default='sparse')
    parser.add_argument('--reward_scale', type=float, default=1)
    parser.add_argument('--exp_knob', type=int)
    parser.add_argument('--num_orientations', type=int, default=-1)
    parser.add_argument('--num_rotations', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='extruded_polygons_r_0.1_s_8_h_0.05', help='the dataset to use')
    parser.add_argument('--use_correctness', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--collect_initial_batch', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='whether we collect an initial batch of data to train the discriminator before updating the explorer')
    parser.add_argument('--initial_batch_ep_len', type=float, default=2000)
    parser.add_argument('--initial_batch_policy', type=str, default='random')
    parser.add_argument('--rotate', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--translate', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--translate_range', type=float, default=0.01)
    parser.add_argument('--sensor_noise', type=float, default=0)
    parser.add_argument('--all_in_one', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='all in one policy: the agent output everything and discriminator is not needed')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.exp_name = f'{args.exp_name}_{socket.gethostname()}_{args.timestr}'
    args.save_dir = os.path.join(args.save_dir, args.exp_name)
    args.log_dir = os.path.join(args.log_dir, args.exp_name)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.n_minibatch)

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
                    move = agent.get_move(next_obs)[0]
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
        if len(varied_dataset) >= varied_dataset.buffer_size and (update - 1) % explore_updates == 0:
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
                    torch.save(agent.state_dict(), os.path.join(folder_path, 'explorer_model.pth'))

                    checkpoint_metadata = {
                        'steps': global_step,
                        'episode': episode,
                        'update': update,
                        'episode_reward': episode_reward,
                        'episode_length': episode_length,
                        'episode_success': bool(episode_success),  # convert numpy.bool__ to bool
                        'running_reward': running_reward,
                        'running_length': running_length,
                        'running_success': running_success,
                        'training_time': training_time,
                        'explorer_lr': optimizer.param_groups[0]['lr'],
                        'explorer_path': os.path.join(folder_path, 'explorer_model.pth'),
                        'discriminator_path': discriminator_path,
                        'discriminator_train_loss': discriminator_train_loss,
                        'discriminator_train_acc': discriminator_train_acc,
                        'discriminator_test_loss': discriminator_test_loss,
                        'discriminator_test_acc': discriminator_test_acc,
                        'args': vars(args)}

                    mu.save_json(checkpoint_metadata, os.path.join(folder_path, 'metadata.json'))
                    logger.log("----------------------------------------")
                    logger.log('Saving models to {}'.format(os.path.join(folder_path, 'explorer_model.pth')))
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

    if not args.all_in_one:
        # construct discriminator and the dataset
        discriminator = mu.construct_discriminator(discriminator_type=args.discriminator,
                                                   height=height,
                                                   width=width,
                                                   discriminator_path=args.discriminator_path,
                                                   lr=args.discriminator_lr)
        transform = T.RandomRotation((0, 360)) if args.rotate else None
        varied_dataset = VariedMNISTDataset(buffer_size=args.buffer_size, height=height, width=width, transform=transform)
        varied_dataset.clean_data()

    # we need the true multiprocessing for pybullet environments. Otherwise, you need to set the physics id correctly for each pybullet command
    if args.multiprocess:
        envs = VecPyTorch(
            SubprocVecEnv([make_env(args.env_name, i, args.seed + i, max_x, max_y, step_size, scale, finger_height, args.use_correctness)
                           for i in range(args.num_envs)], "fork"), device)
    else:
        envs = VecPyTorch(
            DummyVecEnv([make_env(args.env_name, i, args.seed + i, max_x, max_y, step_size, scale, finger_height, args.use_correctness)
                         for i in range(args.num_envs)]), device)
    assert isinstance(envs.action_space['move'], Discrete), "only discrete action space is supported"

    agent = Agent(envs.action_space['move'].n if not args.all_in_one else envs.action_space['move'].n + 10, device, frames=1, img_size=height)
    if args.explorer_path is not None:
        agent.load_state_dict(torch.load(args.explorer_path))
    optimizer = optim.Adam(agent.parameters(), lr=args.explorer_lr, eps=1e-5)

    # ALGO Logic: Storage for epoch data
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    moves = torch.zeros((args.num_steps, args.num_envs) + envs.action_space['move'].shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    # Note how `next_obs` and `next_done` are used; their usage is equivalent to
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)

    num_updates = args.total_timesteps // args.batch_size   # total number of updates
    explore_updates = args.explorer_steps // args.batch_size    # how many updates before we train the discriminator
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
    for update in range(1, num_updates + 1):
        train_discdriminator_f()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            if args.train_discriminator:
                # Each time we train the explorer, we strat from the original learning rate and then anneal
                lrnow = linear_schedule(args.explorer_lr, args.explorer_lr / 10, explore_updates,
                                        update % explore_updates)
            else:
                lrnow = linear_schedule(args.explorer_lr, args.explorer_lr / 10, num_updates, update)
            optimizer.param_groups[0]['lr'] = lrnow

        # TRY NOT TO MODIFY: prepare the execution of the game.
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                values[step] = agent.get_value(obs[step]).flatten()
                move, logproba, _ = agent.get_move(obs[step])

            moves[step] = move
            logprobs[step] = logproba

            # STEP!
            # all in one policy
            if args.all_in_one:
                action = []
                for i in range(args.num_envs):
                    if 0 <= move[i].item() < 4:
                        action.append({
                            'move': move[i].item(),
                            'prediction': 0,
                            'max_prob': 0.1,
                            'probs': np.full(10, 0.1),
                            'done': False
                        })
                    else:
                        prediction = move[i].item() - 4
                        probs = np.zeros(10)
                        probs[prediction] = 1
                        action.append({
                            'move': 0,
                            'prediction': prediction,
                            'max_prob': 1,
                            'probs': probs,
                            'done': True
                        })
            else:
                # build the actions to the envs
                prediction, max_prob, probs = discriminator.predict(obs[step].cpu().numpy())

                # angles = envs.get_attr('angle')
                # canonicals = mu.rotate_imgs(obs[step].cpu().numpy(), [-a for a in angles])
                # prediction, max_prob, probs = discriminator.predict(canonicals)

                # action is a list of dictionary
                action = [{'move': move[i].item(),
                           'prediction': prediction[i],
                           'max_prob': max_prob[i],
                           'probs': probs[i],
                           'done': 1 if max_prob[i] >= args.terminal_confidence else 0
                           } for i in range(args.num_envs)]

            next_obs, rs, ds, infos = envs.step(action)
            add_data_f()

            # making sure rs is flattened -> from (8, 1) to (8, ) and next_done is a tensor
            rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)
            # print([(env.prediction, env.num_gt) for env in envs.venv.envs])

            # write log
            write_explorer_log()

        # ---------------- explorer batch data collection finished --------------- #
        # bootstrap reward if not done. reached the batch limit
        with torch.no_grad():
            last_value = agent.get_value(next_obs.to(device)).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)  # [1024, 1, 50, 50]
        b_logprobs = logprobs.reshape(-1)
        b_moves = moves.reshape((-1,) + envs.action_space['move'].shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        target_agent = Agent(envs.action_space['move'].n if not args.all_in_one else envs.action_space['move'].n + 10, device, frames=1, img_size=height)
        inds = np.arange(args.batch_size, )
        for i_epoch_pi in range(args.update_epochs):
            np.random.shuffle(inds)
            target_agent.load_state_dict(agent.state_dict())
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                minibatch_ind = inds[start:end]
                mb_advantages = b_advantages[minibatch_ind]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                _, newlogproba, entropy = agent.get_move(b_obs[minibatch_ind], b_moves.long()[minibatch_ind])
                ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

                # Stats
                approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # Value loss
                new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                    v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind],
                                                                      -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2).mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.kle_stop:
                if approx_kl > args.target_kl:
                    break
            if args.kle_rollback:
                if (b_logprobs[minibatch_ind] - agent.get_move(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])[
                    1]).mean() > args.target_kl:
                    agent.load_state_dict(target_agent.state_dict())
                    break

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/explorer_lr", optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)

        if args.prod_mode:
            data_to_log = {
                "charts/explorer_lr": optimizer.param_groups[0]['lr'],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy.mean().item(),
                "losses/approx_kl": approx_kl.item(),
                "global_step": global_step
            }

        if args.kle_stop or args.kle_rollback:
            writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)

            if args.prod_mode:
                data_to_log = {
                    "debug/pg_stop_iter": i_epoch_pi,
                    "global_step": global_step
                }
                wandb.log(data_to_log)
        logger.dump_all()

    envs.close()
    writer.close()
    logger.remove_all()

