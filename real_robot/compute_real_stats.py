import numpy as np
import os
import sys
sys.path.append("../")
import misc_utils as mu


if __name__ == "__main__":
    real_exp_folder = 'real_exp'
    stats = []

    folders = os.listdir(real_exp_folder)
    for f in folders:
        if 'ppo' in f.split('_'):
            stat = mu.load_json(os.path.join(real_exp_folder, f, 'results.json'))
            stats.append(stat)

    success = np.array([stat['success_rate'] for stat in stats])
    actions = np.array([stat['actions'] for stat in stats])
    explored_pixels = np.array([stat['explored_pixels'] for stat in stats])
    exploration_rate = np.array([stat['exploration_rate'] for stat in stats])
    print(f'actions: {np.mean(actions)}\n'
          f'actions_std: {np.std(actions)}\n'
          f'explored_pixels: {np.mean(explored_pixels)}\n'
          f'explored_pixels_std: {np.std(explored_pixels)}\n'
          f'success_rate: {np.mean(success)}')
