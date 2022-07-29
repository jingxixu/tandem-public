import json
import os
import argparse
import misc_utils as mu
from enjoy_floating_finger import enjoy_floating_finger
from floating_finger_env import FloatingFingerEnv
import torch
import gym
import pprint
from distutils.util import strtobool
import numpy as np

""" This file takes the whole saved directory and finds the best one among the top ones based on a metric """


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_pybullet', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--render_ob', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--save_dir', type=str, required=True,
                        help='path to saved checkpoints')
    parser.add_argument('--metric', type=str, default='running_length')
    parser.add_argument('--sr_threshold', type=float, default=0.95,
                        help="only keep those models with high enough success rate")
    parser.add_argument('--reverse', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--dry_run', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='only list the training performance without evaluation')
    parser.add_argument('--sensor_noise', type=float, default=0)
    args = parser.parse_args()
    return args


def pick_rl_folder(folder):
    x = folder.split('_')
    return x[0] == 'episode'


if __name__ == "__main__":
    args = get_args()
    model_folders = os.listdir(args.save_dir)
    model_folders = list(filter(pick_rl_folder, model_folders))
    model_folders.sort()

    meta_datas = [mu.load_json(os.path.join(args.save_dir, mf, 'metadata.json')) for mf in model_folders]
    for md, mf in zip(meta_datas, model_folders):
        md['explorer_path'] = os.path.join(args.save_dir, mf, 'explorer_model.pth')
    meta_datas = sorted(meta_datas, key=lambda x: x[args.metric], reverse=args.reverse)
    # filter these meta datas
    meta_datas = [d for d in meta_datas if d[args.metric] != 0 and d['running_success'] >= args.sr_threshold]

    # now only keeps the top ones
    meta_datas = meta_datas[:args.top]

    # make the env
    env = FloatingFingerEnv(
        render_ob=args.render_ob,
        render_pybullet=args.render_pybullet,
        debug=args.debug,
        start_on_border=meta_datas[0]['args']['start_on_border'],
        num_orientations=meta_datas[0]['args']['num_orientations'],
        dataset=meta_datas[0]['args']['dataset'],
        threshold=meta_datas[0]['args']['terminal_confidence'],
        translate=meta_datas[0]['args']['translate'],
        translate_range=meta_datas[0]['args']['translate_range'],
        sensor_noise=args.sensor_noise)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    results = []
    for md in meta_datas:
        # this is important for reproducibility,
        # making sure each env is tested with the same seed to make it fair
        # seed pytorch so stochastic action is sampled the same
        mu.seed_env(env, args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        result = {}
        result['training'] = {}
        result['training'][args.metric] = md[args.metric]
        result['training']['steps'] = md['steps']
        result['training']['episode'] = md['episode']
        result['training']['training_time'] = md['training_time']
        result['training']['explorer_path'] = md['explorer_path']
        result['training']['disriminator_path'] = md['discriminator_path']
        result['training']['model_dir'] = os.path.dirname(md['explorer_path'])
        result['training']['running_length'] = md['running_length']
        result['training']['running_success'] = md['running_success']

        print()
        pprint.pprint(result['training'], indent=4)
        print()

        if not args.dry_run:
            result['testing'] = enjoy_floating_finger(env,
                                                      md['args']['discriminator'],
                                                      md['discriminator_path'],
                                                      'ppo' if 'explorer_type' not in md['args'].keys() else md['args']['explorer_type'],
                                                      md['explorer_path'],
                                                      md['args']['terminal_confidence'],
                                                      save_npy=False,
                                                      exp_name='exp')

            print()
            pprint.pprint(result['testing'], indent=4)
            print()

        print('--------------------')
        print('end')
        print('--------------------')

        results.append(result)

    if not args.dry_run:
        # training and testing have different naming convention
        if args.metric == 'running_length':
            testing_metric_name = 'actions'
        elif args.metric == 'running_success':
            testing_metric_name = 'success_rate'
        else:
            raise TypeError
        results = sorted(results, key=lambda x: x['testing'][args.metric], reverse=args.reverse)
    else:
        results = sorted(results, key=lambda x: x['training'][args.metric], reverse=args.reverse)

    print()
    pprint.pprint(results, indent=4)
    print()
