from __future__ import print_function
from PIL import Image
import csv
import os

import time

import torch.utils.data as data

'''
STATS

training_2019
mean: tensor([0.6681, 0.5301, 0.5247]) | std: tensor([0.1337, 0.1480, 0.1595])

'''


class vidix(data.Dataset):
    """ vidix Dataset. """

    def __init__(self, transform=None):
        start_time = time.time()
        self.transform = transform
        # self.split_list, self.lbls = self.read_csv(split_name)

        self.data_root = '/nas/softechict-nas-1/fpollastri/data/vidix_images/'
        self.split_list = self.read_dataset()

        print("Time: " + str(time.time() - start_time))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, name)
        """

        image = Image.open(os.path.join(self.data_root, self.split_list[index]))
        # start_time = time.time()
        if self.transform is not None:
            image = self.transform(image)
        # print("transformations time: " + str(time.time()-start_time))
        return image, self.split_list[index]

    def __len__(self):
        return len(self.split_list)

    def read_dataset(self):
        split_list = os.listdir(self.data_root)
        return split_list

