import os
import numpy as np
import argparse
from discriminator import LearnedDiscriminator
from discriminator_dataset import VariedMNISTDataset
import misc_utils as mu
import pprint
import torchvision.transforms as T


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='path to the dataset directory')
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    height = 60
    width = 60
    X = np.load(os.path.join(args.dataset_dir, 'X.npy'))
    Y = np.load(os.path.join(args.dataset_dir, 'Y.npy'))
    assert X.shape[0] == Y.shape[0]
    buffer_size = X.shape[0]

    stats = mu.compute_class_balance(X, Y)
    pprint.pprint(stats, indent=4)

    mu.compute_pixel_freq(X, True)
    varied_dataset = VariedMNISTDataset(buffer_size=buffer_size, height=height, width=width, transform=T.RandomRotation((0, 360)))
    discriminator = LearnedDiscriminator(height, width, save_dir='debug')
    varied_dataset.add_data(X, Y)
    train_loader, test_loader = mu.construct_loaders(dataset=varied_dataset, split=0.2, train_batch_size=1000, test_batch_size=1000)
    discriminator.learn(epochs=20,
                        train_loader=train_loader,
                        test_loader=test_loader)

    print('here')
