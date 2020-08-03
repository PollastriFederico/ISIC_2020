from torchvision import transforms
import torch
import isic_classification_dataset as isic
import only_skin_dataset as only_skin
import isic_classification_wsegm_dataset as isic_wsegm

from torch.utils.data import DataLoader
import numpy as np
import numpy
import imgaug as ia
import random
import matplotlib.pyplot as plt
from PIL import Image


# Cutout Data Augmentation Class. Can be used in the torchvision compose pipelin
class CutOut(object):
    """Randomly mask out one or more patches from an image.
    Args:
    n_holes (int): Number of patches to cut out of each image.
    length (int): The length (in pixels) of each square patch.

    Return:
     -image with mask and original image
    """

    def __init__(self, n_holes, length):
        # if isinstance(n_holes, int):
        #     self.n_holes = n_holes
        #     self.is_list = False
        # else:
        #     self.is_list = True
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        if not (self.length[0] == 0 and len(self.length) == 1):
            # if self.is_list:
            t_n_holes = random.choice(self.n_holes)
            # else:
            #     t_n_holes = self.n_holes

            h = img.size(1)
            w = img.size(2)

            mask = numpy.ones((h, w), numpy.float32)

            for n in range(t_n_holes):
                t_length = random.choice(self.length)
                y = random.randint(0, h - 1)
                x = random.randint(0, w - 1)

                y1 = numpy.clip(y - t_length // 2, 0, h)
                y2 = numpy.clip(y + t_length // 2, 0, h)
                x1 = numpy.clip(x - t_length // 2, 0, w)
                x2 = numpy.clip(x + t_length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            return img * mask
        else:
            return img


class ImgAugTransformSquareofSkin:
    def __init__(self, config_code, size=512, SRV=False):
        self.SRV = SRV
        self.size = size
        self.config = config_code

        sometimes = lambda aug: ia.augmenters.Sometimes(0.5, aug)

        # FIRST "BIT" OF CONFIG CODE IS THE PADDING MODE
        cc = max(self.config, 0)
        if cc % 2:
            self.mode = 'constant'
            cc -= 1
        else:
            self.mode = 'reflect'
        self.possible_aug_list = [
            None,  # dummy for padding mode                                                         # 1
            None,  # placeholder for future inclusion                                               # 2
            sometimes(ia.augmenters.AdditivePoissonNoise((0, 10), per_channel=True)),  # 4
            sometimes(ia.augmenters.Dropout((0, 0.02), per_channel=False)),  # 8
            sometimes(ia.augmenters.GaussianBlur((0, 0.8))),  # 16
            sometimes(ia.augmenters.AddToHueAndSaturation((-20, 10))),  # 32
            sometimes(ia.augmenters.GammaContrast((0.5, 1.5))),  # 64
            None,  # placeholder for future inclusion                                               # 128
            None,  # placeholder for future inclusion                                               # 256
            sometimes(ia.augmenters.PiecewiseAffine((0, 0.04))),  # 512
            sometimes(ia.augmenters.Affine(shear=(-20, 20), mode=self.mode)),  # 1024
            sometimes(ia.augmenters.CropAndPad(percent=(-0.2, 0.05), pad_mode=self.mode))  # 2048
        ]
        self.aug_list = [
            ia.augmenters.Fliplr(0.5),
            ia.augmenters.Flipud(0.5),
        ]

        # FIRST "BIT" OF CONFIG CODE IS THE PADDING MODE, FIRST LOOP IS A DUMMY
        for i in range(len(self.possible_aug_list)):
            if cc % 2:
                self.aug_list.append(self.possible_aug_list[i])
            cc = cc // 2
            if not cc:
                break

        self.aug_list.append(ia.augmenters.Affine(rotate=(-180, 180), mode=self.mode))
        # self.aug_list.append(ia.augmenters.GaussianBlur(4))
        self.aug = ia.augmenters.Sequential(self.aug_list)
        if self.config >= 0:
            print(self.mode)
            for a in self.aug_list:
                print(a.name)

    def __call__(self, img):
        self.aug.reseed(random.randint(1, 10000))

        img = np.array(img)
        if not self.SRV:
            plt.figure()
            plt.imshow(img)
            plt.show(block=False)
        # img = ia.augmenters.GaussianBlur(10).augment_image(img)

        if self.config == -1:
            img = ia.augmenters.CropToFixedSize(width=self.size, height=self.size,
                                                position='uniform').augment_image(img)
            if not self.SRV:
                plt.figure()
                plt.imshow(img)
                plt.show(block=False)
            return img

        crop_size_h = 120
        crop_size_w = 120
        img = ia.augmenters.CropToFixedSize(width=crop_size_w, height=crop_size_h, position='uniform').augment_image(
            img)
        if not self.SRV:
            plt.figure()
            plt.imshow(img)
            plt.show(block=False)

        img = ia.augmenters.Resize({"width": self.size // 2, "height": self.size // 2}).augment_image(img)
        if not self.SRV:
            plt.figure()
            plt.imshow(img)
            plt.show(block=False)

        img = ia.augmenters.PadToFixedSize(width=self.size,
                                           height=self.size,
                                           pad_mode='reflect', position='center').augment_image(img)
        if not self.SRV:
            plt.figure()
            plt.imshow(img)
            plt.show(block=False)

        img = self.aug.augment_image(img)
        if not self.SRV:
            plt.figure()
            plt.imshow(img)
            plt.show(block=False)
        return img


class ImgAugTransform:
    def __init__(self, config_code, size=512, SRV=False):
        self.SRV = SRV
        self.size = size
        self.config = config_code

        # sometimes = lambda aug: ia.augmenters.Sometimes(0.5, aug)
        sometimes = lambda aug: ia.augmenters.Sometimes(1.0, aug)

        # FIRST "BIT" OF CONFIG CODE IS THE PADDING MODE
        cc = max(self.config, 0)
        if cc % 2:
            self.mode = 'constant'
            cc -= 1
        else:
            self.mode = 'reflect'
        self.possible_aug_list = [
            None,  # dummy for padding mode                                                         # 1
            None,  # placeholder for future inclusion                                               # 2
            sometimes(ia.augmenters.AdditivePoissonNoise((0, 10), per_channel=True)),  # 4
            sometimes(ia.augmenters.Dropout((0, 0.02), per_channel=False)),  # 8
            # sometimes(ia.augmenters.GaussianBlur((0, 0.8))),  # 16
            sometimes(ia.augmenters.GaussianBlur(0.6)),  # 16
            sometimes(ia.augmenters.AddToHueAndSaturation((-20, 10))),  # 32
            sometimes(ia.augmenters.GammaContrast((0.5, 1.5))),  # 64
            None,  # placeholder for future inclusion                                               # 128
            None,  # placeholder for future inclusion                                               # 256
            sometimes(ia.augmenters.PiecewiseAffine((0, 0.04))),  # 512
            sometimes(ia.augmenters.Affine(shear=(-20, 20), mode=self.mode)),  # 1024
            sometimes(ia.augmenters.CropAndPad(percent=(-0.2, 0.05), pad_mode=self.mode))  # 2048
        ]
        self.aug_list = [
            # ia.augmenters.Fliplr(0.5),
            # ia.augmenters.Flipud(0.5)
        ]

        # FIRST "BIT" OF CONFIG CODE IS THE PADDING MODE, FIRST LOOP IS A DUMMY
        for i in range(len(self.possible_aug_list)):
            if cc % 2:
                self.aug_list.append(self.possible_aug_list[i])
            cc = cc // 2
            if not cc:
                break

        # self.aug_list.append(ia.augmenters.Affine(rotate=(-180, 180), mode=self.mode))
        self.aug = ia.augmenters.Sequential(self.aug_list)
        if self.config >= 0:
            print(self.mode)
            for a in self.aug_list:
                print(a.name)

    def __call__(self, img):
        self.aug.reseed(random.randint(1, 10000))

        img = np.array(img)
        img = ia.augmenters.PadToFixedSize(width=max(img.shape[0], img.shape[1]),
                                           height=max(img.shape[0], img.shape[1]),
                                           pad_mode=self.mode, position='center').augment_image(img)
        img = ia.augmenters.Resize({"width": self.size, "height": self.size}).augment_image(img)

        if not self.SRV:
            plot(img)

        if self.config == -1:
            if not self.SRV:
                plot(img)
            return img
        else:
            img = self.aug.augment_image(img)
            if not self.SRV:
                plot(img)
            return img


class ImgAugTransformwSegm:
    def __init__(self, config_code, size=512, SRV=False):
        self.SRV = SRV
        self.size = size
        self.config = config_code

        sometimes = lambda aug: ia.augmenters.Sometimes(0.5, aug)

        # FIRST "BIT" OF CONFIG CODE IS THE PADDING MODE
        cc = max(self.config, 0)
        if cc % 2:
            self.mode = 'constant'
            cc -= 1
        else:
            self.mode = 'reflect'
        self.possible_aug_list = [
            None,  # dummy for padding mode                                                         # 1
            None,  # placeholder for future inclusion                                               # 2
            sometimes(ia.augmenters.AdditivePoissonNoise((0, 10), per_channel=True)),  # 4
            sometimes(ia.augmenters.Dropout((0, 0.02), per_channel=False)),  # 8
            sometimes(ia.augmenters.GaussianBlur((0, 0.8))),  # 16
            sometimes(ia.augmenters.AddToHueAndSaturation((-20, 10))),  # 32
            sometimes(ia.augmenters.GammaContrast((0.5, 1.5))),  # 64
            None,  # placeholder for future inclusion                                               # 128
            None,  # placeholder for future inclusion                                               # 256
            sometimes(ia.augmenters.PiecewiseAffine((0, 0.04))),  # 512
            sometimes(ia.augmenters.Affine(shear=(-20, 20), mode=self.mode)),  # 1024
            sometimes(ia.augmenters.CropAndPad(percent=(-0.2, 0.05), pad_mode=self.mode))  # 2048
        ]
        self.aug_list = [
            ia.augmenters.Fliplr(0.5),
            ia.augmenters.Flipud(0.5)
        ]

        # FIRST "BIT" OF CONFIG CODE IS THE PADDING MODE, FIRST LOOP IS A DUMMY
        for i in range(len(self.possible_aug_list)):
            if cc % 2:
                self.aug_list.append(self.possible_aug_list[i])
            cc = cc // 2
            if not cc:
                break

        self.aug_list.append(ia.augmenters.Affine(rotate=(-180, 180), mode=self.mode))
        self.aug = ia.augmenters.Sequential(self.aug_list)
        if self.config >= 0:
            print(self.mode)
            for a in self.aug_list:
                print(a.name)

    def __call__(self, img, segm):
        self.aug.reseed(random.randint(1, 10000))

        img = np.array(img)
        segm = np.array(segm)
        if not self.SRV:
            plt.figure()
            plt.imshow(img)
            plt.show(block=False)
            plt.figure()
            plt.imshow(segm)
            plt.show(block=False)
        img, segm = ia.augmenters.PadToFixedSize(width=max(img.shape[0], img.shape[1]),
                                                 height=max(img.shape[0], img.shape[1]),
                                                 pad_mode=self.mode, position='center').augment_images([img, segm])
        img, segm = ia.augmenters.Resize({"width": self.size, "height": self.size}).augment_images([img, segm])
        if self.config == -1:
            if not self.SRV:
                plt.figure()
                plt.imshow(img)
                plt.show(block=False)
                plt.figure()
                plt.imshow(segm)
                plt.show(block=False)
            return img, segm
        else:
            if not self.SRV:
                plt.figure()
                plt.imshow(self.aug.augment_image(img))
                plt.show(block=False)
                plt.figure()
                plt.imshow(self.aug.augment_image(segm))
                plt.show(block=False)
            return self.aug.augment_images([img, segm])


counter = 0
def plot(img):
    return
    global counter
    im = Image.fromarray(img)
    im.save(f'/nas/softechict-nas-2/fpollastri/gaussian_blur_images/img_{counter}.png')
    counter += 1
    return
    plt.figure()
    plt.imshow(img)
    plt.show(block=False)


def get_dataset(dname='isic2020', dataset_classes=[[0], [1]], size=512, SRV=False,
                batch_size=16, n_workers=0, augm_config=0, cutout_params=[[0], [0]], drop_last_flag=False,
                copy_into_tmp=False):
    dataset = None
    test_dataset = None
    valid_dataset = None
    imgaug_transforms = ImgAugTransform(config_code=augm_config, size=size, SRV=SRV)
    inference_imgaug_transforms = ImgAugTransform(config_code=-1, size=size, SRV=SRV)
    if dname == 'isic2020':
        training_split_name = 'training_v1_2020'
        training_transforms = transforms.Compose([
            imgaug_transforms,
            transforms.ToTensor(),
            transforms.Normalize((0.8062, 0.6214, 0.5914), (0.0826, 0.0964, 0.1085)),
            CutOut(*cutout_params)
        ])

        test_split_name = 'test_v1_2020'
        test_transforms = transforms.Compose([
            inference_imgaug_transforms,
            transforms.ToTensor(),
            transforms.Normalize((0.8062, 0.6214, 0.5914), (0.0826, 0.0964, 0.1085)),
        ])

        valid_split_name = 'val_v1_2020'
        valid_transforms = test_transforms
    elif dname == 'isic2020_inference':
        training_split_name = 'isic2020_testset'

        test_split_name = 'isic2020_testset'
        test_transforms = transforms.Compose([
            inference_imgaug_transforms,
            transforms.ToTensor(),
            transforms.Normalize((0.8062, 0.6214, 0.5914), (0.0826, 0.0964, 0.1085)),
        ])

        valid_split_name = 'isic2020_testset'
        valid_transforms = test_transforms
        training_transforms = test_transforms
    else:
        print("WRONG DATASET NAME")
        raise Exception("WRONG DATASET NAME")

    dataset = isic.ISIC(split_name=training_split_name, classes=dataset_classes, size=(size, size),
                        transform=training_transforms, workers=n_workers, copy_into_tmp=copy_into_tmp)
    test_dataset = isic.ISIC(split_name=test_split_name, classes=dataset_classes, size=(size, size),
                             transform=test_transforms, workers=n_workers, copy_into_tmp=copy_into_tmp)
    valid_dataset = isic.ISIC(split_name=valid_split_name, classes=dataset_classes, size=(size, size),
                              transform=valid_transforms, workers=n_workers, copy_into_tmp=copy_into_tmp)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=n_workers,
                             drop_last=drop_last_flag,
                             pin_memory=True)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=n_workers,
                                  drop_last=False,
                                  pin_memory=True)

    valid_data_loader = DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=n_workers,
                                   drop_last=False,
                                   pin_memory=True)

    return data_loader, test_data_loader, valid_data_loader


def get_dataset_wsegm(dname='isic2019', dataset_classes=[[0], [1], [2], [3], [4], [5], [6], [7]], size=512, SRV=False,
                      batch_size=16, n_workers=0, augm_config=0, cutout_params=[[0], [0]], drop_last_flag=False):
    dataset = None
    test_dataset = None
    valid_dataset = None
    imgaug_transforms = ImgAugTransformwSegm(config_code=augm_config, size=size, SRV=SRV)
    inference_imgaug_transforms = ImgAugTransformwSegm(config_code=-1, size=size, SRV=SRV)
    if dname == 'isic2019':
        training_split_name = 'training_v1_2019'
        training_transforms = imgaug_transforms

        test_split_name = 'test_v1_2019'
        test_transforms = inference_imgaug_transforms

        valid_split_name = 'val_v1_2019'
        valid_transforms = test_transforms

    elif dname == 'isic2019_novalid':

        training_split_name = 'training_p1_2019'
        training_transforms = transforms.Compose([
            imgaug_transforms,
        ])

        test_split_name = 'test_p1_2019'
        test_transforms = transforms.Compose([
            inference_imgaug_transforms,
        ])

        valid_split_name = test_split_name
        valid_transforms = test_transforms

    elif dname == 'isic2019_testwaugm':

        training_split_name = 'training_v1_2019'
        training_transforms = transforms.Compose([
            imgaug_transforms,
        ])

        test_split_name = 'test_v1_2019'
        test_transforms = training_transforms

        valid_split_name = 'val_v1_2019'
        valid_transforms = training_transforms

    elif dname[:21] == 'isic2019_submission_p':
        split_id = dname[-1]
        training_split_name = 'submission_training_p' + split_id + '_2019'
        training_transforms = transforms.Compose([
            imgaug_transforms,
        ])

        test_split_name = 'submission_validation_p' + split_id + '_2019'
        test_transforms = transforms.Compose([
            inference_imgaug_transforms,
        ])

        valid_split_name = test_split_name
        valid_transforms = test_transforms

    else:
        print("WRONG DATASET NAME")
        raise Exception("WRONG DATASET NAME")

    only_img_transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595))])
    only_segm_transform = transforms.Compose([transforms.ToTensor()])

    dataset = isic_wsegm.ISIC(split_name=training_split_name, classes=dataset_classes, load=False, size=(size, size),
                              segmentation_transform=training_transforms, transform=only_img_transforms,
                              mask_transform=only_segm_transform)
    test_dataset = isic_wsegm.ISIC(split_name=test_split_name, classes=dataset_classes, load=False, size=(size, size),
                                   segmentation_transform=test_transforms, transform=only_img_transforms,
                                   mask_transform=only_segm_transform)
    valid_dataset = isic_wsegm.ISIC(split_name=valid_split_name, classes=dataset_classes, load=False, size=(size, size),
                                    segmentation_transform=valid_transforms, transform=only_img_transforms,
                                    mask_transform=only_segm_transform)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=n_workers,
                             drop_last=drop_last_flag,
                             pin_memory=True)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=n_workers,
                                  drop_last=False,
                                  pin_memory=True)

    valid_data_loader = DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=n_workers,
                                   drop_last=False,
                                   pin_memory=True)

    return data_loader, test_data_loader, valid_data_loader


def get_dataset_squares_of_skin(dname='isic2019', dataset_classes=[[0], [1], [2], [3], [4], [5], [6], [7]], size=512,
                                SRV=False,
                                batch_size=16, n_workers=0, augm_config=1632, cutout_params=[[0], [0]],
                                drop_last_flag=False):
    dataset = None
    test_dataset = None
    valid_dataset = None
    imgaug_transforms = ImgAugTransformSquareofSkin(config_code=augm_config, size=size, SRV=SRV)
    inference_imgaug_transforms_SoS = ImgAugTransformSquareofSkin(config_code=-1, size=size, SRV=SRV)
    inference_imgaug_transforms = ImgAugTransform(config_code=-1, size=size, SRV=SRV)
    if dname == 'isic2019':
        training_split_name = 'training_v1_2019'
        training_transforms = transforms.Compose([
            imgaug_transforms,
            transforms.ToTensor(),
            transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595)),
            CutOut(*cutout_params)
        ])

        test_split_name = 'test_v1_2019'
        test_transforms = transforms.Compose([
            inference_imgaug_transforms,
            transforms.ToTensor(),
            transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595)),
        ])

        valid_split_name = 'val_v1_2019'
        valid_transforms = transforms.Compose([
            inference_imgaug_transforms_SoS,
            transforms.ToTensor(),
            transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595)),
        ])

    elif dname == 'isic2019_novalid':

        training_split_name = 'training_p1_2019'
        training_transforms = transforms.Compose([
            imgaug_transforms,
            transforms.ToTensor(),
            transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595)),
        ])

        test_split_name = 'test_p1_2019'
        test_transforms = transforms.Compose([
            inference_imgaug_transforms,
            transforms.ToTensor(),
            transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595)),
        ])

        valid_split_name = test_split_name
        valid_transforms = test_transforms

    elif dname == 'isic2019_testwaugm':

        training_split_name = 'training_v1_2019'
        training_transforms = transforms.Compose([
            imgaug_transforms,
            transforms.ToTensor(),
            transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595)),
            CutOut(*cutout_params)
        ])

        test_split_name = 'test_v1_2019'
        test_transforms = training_transforms

        valid_split_name = 'val_v1_2019'
        valid_transforms = training_transforms

    elif dname[:21] == 'isic2019_submission_p':
        split_id = dname[-1]
        training_split_name = 'submission_training_p' + split_id + '_2019'
        training_transforms = transforms.Compose([
            imgaug_transforms,
            transforms.ToTensor(),
            transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595)),
        ])

        test_split_name = 'submission_validation_p' + split_id + '_2019'
        test_transforms = transforms.Compose([
            inference_imgaug_transforms,
            transforms.ToTensor(),
            transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595)),
        ])

        valid_split_name = test_split_name
        valid_transforms = transforms.Compose([
            inference_imgaug_transforms_SoS,
            transforms.ToTensor(),
            transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595)),
        ])

    else:
        print("WRONG DATASET NAME")
        raise Exception("WRONG DATASET NAME")

    dataset = isic.ISIC(split_name=training_split_name, classes=dataset_classes, load=False, size=(size, size),
                        transform=training_transforms)
    # test_dataset = isic.ISIC(split_name=test_split_name, classes=dataset_classes, load=False, size=(size, size),
    #                          transform=test_transforms)
    # valid_dataset = isic.ISIC(split_name=valid_split_name, classes=dataset_classes, load=False, size=(size, size),
    #                           transform=valid_transforms)

    test_dataset = only_skin.skin(transform=test_transforms)
    valid_dataset = only_skin.skin(transform=valid_transforms)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=n_workers,
                             drop_last=drop_last_flag,
                             pin_memory=True)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=n_workers,
                                  drop_last=False,
                                  pin_memory=True)

    valid_data_loader = DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=n_workers,
                                   drop_last=False,
                                   pin_memory=True)

    return data_loader, test_data_loader, valid_data_loader


def find_stats(data_loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _, _ in data_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print("\ntraining")
    print("mean: " + str(mean) + " | std: " + str(std))


def draw_histogram(data_loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _, _ in data_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print("\ntraining")
    print("mean: " + str(mean) + " | std: " + str(std))


if __name__ == '__main__':

    dataset = isic.ISIC(split_name='training_v1_2020',  # classes=[[0, 1, 2, 3, 4, 5, 6]],
                        size=(512, 512),
                        transform=transforms.Compose([
                            transforms.Resize((512, 512)),
                            transforms.ToTensor(),
                        ]),
                        workers=16
                        )
    data_loader = DataLoader(dataset,
                             batch_size=16,
                             shuffle=False,
                             num_workers=16,
                             drop_last=False,
                             pin_memory=True)

    find_stats(data_loader)

    # data_loader, _, _ = get_dataset(augm_config=16, dname='isic2020_inference')
    # for stuff in data_loader:
    #     pass
