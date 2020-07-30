from __future__ import print_function
from PIL import Image
import csv

import time

import torch.utils.data as data

'''
STATS

training_2019
mean: tensor([0.6681, 0.5301, 0.5247]) | std: tensor([0.1337, 0.1480, 0.1595])

'''


class ISIC(data.Dataset):
    """ ISIC Dataset. """
    # data_root="/home/jmaronasm/ISIC_challenge_2019/Task_3/"
    data_root = '/nas/softechict-nas-1/sallegretti/data/ISIC/SIIM-ISIC/'

    splitsdic = {
        'training_v1_2020': data_root + "2k20_train_partition.csv",
        'test_v1_2020': data_root + "2k20_validation_partition.csv",
        'val_v1_2020': data_root + "2k20_test_partition.csv",
    }

    def __init__(self, split_list=None, split_name='training_v1_2020', classes=[[0], [1]], load=False,
                 size=(512, 512), segmentation_transform=None, transform=None, target_transform=None):
        start_time = time.time()
        self.segmentation_transform = segmentation_transform
        self.transform = transform
        self.target_transform = target_transform
        self.split_list = split_list
        self.load = load
        self.size = size
        self.split_name = split_name
        if len(classes) == 1:
            self.classes = [[c] for c in classes[0]]
        else:
            self.classes = classes

        print('loading ' + split_name)
        # self.split_list, self.lbls = self.read_csv(split_name)
        self.read_dataset()
        if load:
            print("LOADING " + str(len(self.split_list)) + " images in MEMORY")
            self.imgs = self.get_images(self.split_list, self.size)
        else:
            self.imgs = self.get_names(self.split_list)

        print("Time: " + str(time.time() - start_time))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, ground)
        """

        if not self.load:
            image = Image.open(self.imgs[index])

        else:
            image = self.imgs[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, self.lbls[index], self.imgs[index]

    def __len__(self):
        return len(self.split_list)

    def read_dataset(self):
        split_list = []
        labels_list = []
        fname = self.splitsdic.get(self.split_name)

        with open(fname) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                if row[0] == 'image_name':
                    continue
                split_list.append(row[0])
                labels_list.append(int(row[7]))
        self.split_list = split_list
        self.lbls = labels_list
        return split_list, labels_list

    @classmethod
    def get_names(cls, n_list):
        imgs = []
        for n in n_list:
            imgs.append(cls.data_root + "images/" + n + ".jpg")
        return imgs

    @classmethod
    def get_images(cls, i_list, size):
        imgs = []
        for i in i_list:
            imgs.append(Image.open(cls.data_root + "images/ISIC_" + str(i) + ".jpg").resize(size, Image.BICUBIC))
        return imgs

    @classmethod
    def read_csv(cls, csv_filename):
        split_list = []
        labels_list = []
        fname = cls.splitsdic.get(csv_filename)

        with open(fname) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                if row[0] == 'image':
                    continue
                split_list.append(row[0])
                for i in range(len(row) - 1):
                    if row[1 + i] == '1.0':
                        labels_list.append(i)
                        break

        return split_list, labels_list


def denormalize(img):
    # mean = (0.6681, 0.5301, 0.5247)
    # std = (0.1337, 0.1480, 0.1595)
    mean = [0.6681, 0.5301, 0.5247]
    std = [0.1337, 0.1480, 0.1595]

    for i in range(img.shape[0]):
        img[i, :, :] = img[i, :, :] * std[i]
        img[i, :, :] = img[i, :, :] + mean[i]
    return img
