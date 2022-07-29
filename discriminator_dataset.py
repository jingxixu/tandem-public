import copy

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import os
from PIL import Image


class OccludedGridsDataset(Dataset):
    def __init__(self, dataset_dir):
        # N, h, w
        self.X = np.load(os.path.join(dataset_dir, 'X.npy'))
        # N, 1
        self.Y = np.load(os.path.join(dataset_dir, 'Y.npy'))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[[idx]], self.Y[idx][0]


class VariedMNISTDataset(Dataset):
    def __init__(self, buffer_size, height, width, transform=None):
        self.buffer_size = buffer_size
        self.height = height
        self.width = width
        self.imgs = np.zeros((self.buffer_size, 1, height, width), dtype=np.uint8)
        self.nums = np.zeros((self.buffer_size, 1), dtype=np.int64)
        self.len = 0
        self.pointer = 0  # the starting index (inclusive) to add the data
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        return:
            img: (1, height, width), uint8
            num: int64
        """
        if idx >= self.len:
            raise ValueError
        img = copy.deepcopy(self.imgs[idx])
        # first transform if needed
        if self.transform:
            img = Image.fromarray(img[0])
            img = self.transform(img)
            img = np.asarray(img)[None, ...]
        # normalize the image to [-1, 1]
        img = (img / 255.0 - 0.5) / 0.5
        return img, self.nums[idx][0]

    def add_data(self, new_imgs, new_nums):
        """

        :param new_imgs: (n, 1, height, width) or list of (1, height, width)
        :param new_nums: (n, 1) or list of int
        :return:
        """
        new_imgs = np.array(new_imgs)
        new_nums = np.array(new_nums)

        if new_nums.ndim == 1:
            new_nums = new_nums[..., None]

        assert new_imgs.dtype == np.uint8
        assert new_nums.dtype == np.int64

        n = new_imgs.shape[0]

        if self.pointer + n < self.buffer_size:
            self.imgs[self.pointer:self.pointer+n, ...] = new_imgs
            self.nums[self.pointer:self.pointer+n, ...] = new_nums
            self.pointer = self.pointer + n
        else:
            overflow = self.pointer + n - self.buffer_size
            self.imgs[self.pointer:self.buffer_size, ...] = new_imgs[:n-overflow, ...]
            self.nums[self.pointer:self.buffer_size, ...] = new_nums[:n-overflow, ...]

            # if overflow is 0, this has no effect. a[100:, ...] will not throw error even 100 > len(a)
            self.imgs[0:overflow, ...] = new_imgs[n-overflow:, ...]
            self.nums[0:overflow, ...] = new_nums[n-overflow:, ...]
            self.pointer = overflow

        self.len = self.len + n
        if self.len >= self.buffer_size:
            self.len = self.buffer_size

    def clean_data(self):
        self.imgs = np.zeros((self.buffer_size, 1, self.height, self.width), dtype=np.uint8)
        self.nums = np.zeros((self.buffer_size, 1), dtype=np.int64)
        self.len = 0
        self.pointer = 0

    def export_data(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, 'X.npy'), self.imgs)
        np.save(os.path.join(save_dir, 'Y.npy'), self.nums)
