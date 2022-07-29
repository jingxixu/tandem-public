import json
import numpy as np
import argparse
import misc_utils as mu
import torch
import gym
from distutils.util import strtobool
import pprint
from floating_finger_env import FloatingFingerEnv
np.set_printoptions(suppress=True)
import time
import os
from html_vis import html_visualize


""" This evaluation file gives the generic control over different combinations of env, discriminators and explorers """


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_pybullet', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--render_ob', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--seed', type=int, default=10)

    # The good thing about keeping them separately instead of loading the meta data is that
    # we can test on different combinations
    # env related
    parser.add_argument('--num_orientations', type=int, default=-1)
    parser.add_argument('--translate', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--translate_range', type=float, default=0.01)
    parser.add_argument('--dataset', type=str, default='extruded_polygons_r_0.1_s_8_h_0.05', help='the dataset to use')
    parser.add_argument('--terminal_confidence', type=float, default=0.98)
    parser.add_argument('--sensor_noise', type=float, default=0)

    # discriminator
    parser.add_argument('--discriminator', type=str, default='learned',
                        help='one of "dummy", "gt", or "learned"')
    parser.add_argument('--discriminator_path', type=str,
                        help='path to the learned discriminator model checkpoint or to the gt discriminator grids path')

    # explorer
    parser.add_argument('--explorer', type=str, default='random')
    parser.add_argument('--explorer_path', type=str)

    # save stats
    parser.add_argument('--save_npy', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--exp_name', type=str, default='exp')

    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.exp_name = args.exp_name + '_' + args.timestr
    return args


def enjoy_floating_finger(env,
                          discriminator,
                          discriminator_path,
                          dataset,
                          explorer,
                          explorer_path,
                          terminal_confidence,
                          save_npy,
                          exp_name,
                          num_episodes=1000,
                          polygon_id=None,
                          angle=None):
    result = {}
    if explorer == 'all_in_one':
        discriminator = None
    else:
        discriminator = mu.construct_discriminator(discriminator_type=discriminator,
                                                   height=env.max_x_idx,
                                                   width=env.max_y_idx,
                                                   discriminator_path=discriminator_path,
                                                   dataset=dataset)

    explorer = mu.construct_explorer(explorer_type=explorer, image_size=env.max_x_idx, explorer_path=explorer_path, discriminator=discriminator)

    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    episode_explored_pixels = []
    episode_exploration_rate = []
    html_data = dict()
    for i in range(num_episodes):
        obs = []
        actions = []
        ob = env.reset(polygon_id=polygon_id, angle=angle)
        obs.append(ob)
        explorer.reset(ob)
        done = False
        while not done:
            action = mu.get_action(explorer, discriminator, ob, terminal_confidence=terminal_confidence)
            ob, reward, done, info = env.step(action)
            actions.append(action)
            obs.append(ob)
        print(f"exp {i}, polygon id {env.polygon_id}, angle: {env.angle}, actions: {info['episode']['l']}, prediction: {info['prediction']}, success: {info['success']}")
        episode_rewards.append(info['episode']['r'])
        episode_lengths.append(info['episode']['l'])
        episode_successes.append(info['success'])
        episode_explored_pixels.append(info['num_explored_pixels'])
        episode_exploration_rate.append(info['num_explored_pixels'] / info['episode']['l'])

        if save_npy:
            stat = dict()
            # html visualization data
            stat[f'{i}_actions'] = f"{info['episode']['l']}"
            stat[f'{i}_explored_pixels'] = f"{info['num_explored_pixels']}"
            stat[f'{i}_exploration-rate'] = f"{info['num_explored_pixels'] / info['episode']['l']}"
            stat[f'{i}_success'] = f"{info['success']}"
            stat[f'{i}_gt'] = f"{info['num_gt']}"
            stat[f'{i}_prediction'] = f"{info['prediction']}"
            stat[f'{i}_final-ob'] = mu.expand_occupancy_grid(ob, 10)
            html_data.update(stat)

            folder_path = os.path.join(exp_name, f"exp_{i:03d}_{info['success']}")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            stat_json = dict(stat)
            stat_json.pop(f'{i}_final-ob')
            mu.save_json(stat_json, os.path.join(folder_path, 'stat.json'))
            np.save(os.path.join(folder_path, 'obs.npy'), np.array(obs))
            np.save(os.path.join(folder_path, 'actions.npy'), np.array(actions))

    result['reward'] = float(np.mean(episode_rewards))
    result['reward_std'] = float(np.std(episode_rewards))
    result['actions'] = float(np.mean(episode_lengths))
    result['actions_std'] = float(np.std(episode_lengths))
    result['explored_pixels'] = float(np.mean(episode_explored_pixels))
    result['explored_pixels_std'] = float(np.std(episode_explored_pixels))
    result['exploration_rate'] = float(np.mean(episode_exploration_rate))
    result['exploration_rate_std'] = float(np.std(episode_exploration_rate))
    result['success_rate'] = float(np.mean(episode_successes))
    if save_npy:
        mu.save_json(result, os.path.join(exp_name, 'results.json'))

        # make the html visualization
        ids = [str(i) for i in range(num_episodes)]
        cols = ['actions', 'explored_pixels', 'exploration-rate', 'success', 'prediction', 'gt', 'final-ob']
        html_visualize(
            web_path=os.path.join(exp_name, 'html'),
            data=html_data,
            ids=ids,
            cols=cols,
            others=[{'name': 'summary', 'data': json.dumps(result, indent=4)}],
            title=exp_name,
            threading_num=4
        )
    return result


if __name__ == "__main__":
    args = get_args()

    # environment scale
    env = FloatingFingerEnv(
        render_pybullet=args.render_pybullet,
        render_ob=args.render_ob,
        debug=args.debug,
        num_orientations=args.num_orientations,
        translate=args.translate,
        translate_range=args.translate_range,
        dataset=args.dataset,
        threshold=args.terminal_confidence,
        sensor_noise=args.sensor_noise,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    mu.seed_env(env, args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)   # some of the explorers use np random, so needs to seed them as well.

    start_time = time.time()
    res = enjoy_floating_finger(env,
                                args.discriminator,
                                args.discriminator_path,
                                args.dataset,
                                args.explorer,
                                args.explorer_path,
                                args.terminal_confidence,
                                save_npy=args.save_npy,
                                exp_name=args.exp_name)

    print()
    pprint.pprint(res, indent=4)
    print()
    print(f'time: {time.time() - start_time}')
    env.close()

