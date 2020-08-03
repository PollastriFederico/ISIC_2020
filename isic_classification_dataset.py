from __future__ import print_function
from PIL import Image
import csv

import time
import os
import shutil

import torch.utils.data as data

from single_file_dataset import Sfd

'''
STATS

training_2019
mean: tensor([0.6681, 0.5301, 0.5247]) | std: tensor([0.1337, 0.1480, 0.1595])

training_2020
training
mean: tensor([0.8062, 0.6214, 0.5914]) | std: tensor([0.0826, 0.0964, 0.1085])
'''


class ISIC(data.Dataset):
    """ ISIC Dataset. """
    # data_root="/home/jmaronasm/ISIC_challenge_2019/Task_3/"
    data_root = '/nas/softechict-nas-1/sallegretti/data/ISIC/SIIM-ISIC/'

    splitsdic = {
        'training_v1_2020': data_root + "2k20_train_partition.csv",
        'test_v1_2020': data_root + "2k20_test_partition.csv",
        'val_v1_2020': data_root + "2k20_validation_partition.csv",
        'isic2020_testset': data_root + "test.csv",
    }

    sfddic = {
        'training_v1_2020': "train.sfd",
        'test_v1_2020': "test.sfd",
        'val_v1_2020': "val.sfd",
        'isic2020_testset': "submission_test.sfd",
    }

    def __init__(self, split_name='training_v1_2020', classes=[[0], [1]], size=(512, 512),
                 transform=None, workers=0, copy_into_tmp=False, use_sfd=False,):
        start_time = time.time()
        self.transform = transform
        self.split_list = None
        self.size = size
        self.split_name = split_name
        self.workers = workers
        self.copy_into_tmp = copy_into_tmp
        self.dataset_path = os.path.join(self.data_root, self.sfddic[self.split_name])
        if len(classes) == 1:
            self.classes = [[c] for c in classes[0]]
        else:
            self.classes = classes

        print('loading ' + split_name)
        # self.split_list, self.lbls = self.read_csv(split_name)
        self.read_dataset()

        if self.copy_into_tmp:
            self.perform_copy_into_tmp()

        self.images = Sfd(self.dataset_path, workers=self.workers)
        # self.imgs = self.get_names(self.split_list)  #FROM 2019
        print("Time: " + str(time.time() - start_time))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, ground)
        """
        # start_time = time.time()

        image = self.images[index]
        # image = Image.open(self.imgs[index])  #FROM 2019
        # print(f'Get image time: {time.time() - start_time}')
        if self.transform is not None:
            #start_time = time.time()
            image = self.transform(image)
            #print(f'Get transform time: {time.time() - start_time}')

        return image, self.lbls[index], self.split_list[index]

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
                if len(row) == 8:
                    labels_list.append(int(row[7]))
                else:
                    labels_list.append(0)
        self.split_list = split_list
        self.lbls = labels_list
        return split_list, labels_list

    def perform_copy_into_tmp(self):
        dataset_path_tmp = '/tmp/sallegretti_dataset_' + self.sfddic[self.split_name]
        finished_path = '/tmp/sallegretti_dataset_' + self.sfddic[self.split_name] + '.finished'

        # atomically create tmp file if it does not exists
        try:
            fd = os.open(dataset_path_tmp, os.O_CREAT | os.O_EXCL)
            os.close(fd)
            # file does not exist, copy it from nas
            print('Copying dataset from nas...', end='\n')
            shutil.copyfile(self.dataset_path, dataset_path_tmp)
            open(finished_path, 'w').close()
        except FileExistsError:
            # file exists, check if it is finished
            counter = 0
            print('Waiting for another process to finish copying dataset from nas...', end='\n')
            while not os.path.exists(finished_path):
                time.sleep(5)
                counter += 1
                if counter > 200:
                    raise Warning('Failure.')
        print('Done.')
        self.dataset_path = dataset_path_tmp

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
