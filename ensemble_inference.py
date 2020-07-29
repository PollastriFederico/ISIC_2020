import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
from torch import nn
import time
import numpy as np
import os
from classification_net import ClassifyNet
import inference_dataset as skin_lesion
from torch.utils.data import DataLoader
from torchvision import transforms
import imgaug as ia
import random

class_dict = {
    0: 'Melanoma',
    1: 'Melanocytic nevus',
    2: 'Basal cell carcinoma',
    3: 'Actinic keratosis',
    4: 'Benign keratosis',
    5: 'Dermatofibroma',
    6: 'Vascular lesion',
    7: 'Squamous cell carcinoma',
}


class ImgAugTransform:
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
        start_time = time.time()
        self.aug.reseed(random.randint(1, 10000))
        # img = ia.augmenters.Resize({"width": self.size, "height": self.size}).augment_image(img)
        # print('resize time: ' + str(time.time() - start_time))
        start_time = time.time()
        if self.config == -1:
            return img
        else:
            img = np.ascontiguousarray(self.aug.augment_image(img))
            # print('everything else time: ' + str(time.time() - start_time))
            return img


def get_dataset(dname='isic2019', dataset_classes=[[0], [1], [2], [3], [4], [5], [6], [7]], size=512, SRV=False,
                load_flag=False, batch_size=16, n_workers=0, augm_config=0, drop_last_flag=False):
    dataset = None
    test_dataset = None
    valid_dataset = None
    imgaug_transforms = ImgAugTransform(config_code=augm_config, size=size, SRV=SRV)
    inference_imgaug_transforms = ImgAugTransform(config_code=-1, size=size, SRV=SRV)

    if dname == 'inference_waugm':

        test_split_name = 'inference'
        test_transforms = transforms.Compose([
            imgaug_transforms,
            transforms.ToTensor(),
            transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595)),
        ])

    elif dname == 'cropped_inference_waugm':

        test_split_name = 'cropped_inference'
        test_transforms = transforms.Compose([
            imgaug_transforms,
            transforms.ToTensor(),
            transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595)),
        ])

    else:
        print("WRONG DATASET NAME")
        raise Exception("WRONG DATASET NAME")

    test_dataset = skin_lesion.DERMO(split_name=test_split_name, classes=dataset_classes, load=load_flag,
                                     size=(size, size),
                                     transform=test_transforms)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=n_workers,
                                  drop_last=False,
                                  pin_memory=True)

    return test_data_loader


def eval(class_model, e_loader, with_temp_scal=False):
    with torch.no_grad():

        class_model.n.eval()
        sofmx = nn.Softmax(dim=-1)

        predictions_all = None
        predictions = None
        predicted_c = None
        start_time = time.time()

        for idx, (x, target, img_name) in enumerate(e_loader):
            # print('data time: ' + str(time.time() - start_time))
            start_time = time.time()
            x = x.to('cuda')
            out = class_model.n(x)
            # print('model time: ' + str(time.time() - start_time))
            start_time = time.time()
            if with_temp_scal:
                out = class_model.temp_scal_model(out)

            output = out
            check_output_all = sofmx(output)
            check_output, res = torch.max(check_output_all, -1)

            if idx == 0:
                predictions_all = check_output_all.cpu()
                prediction = check_output.cpu()
                predicted_c = res.cpu()
            else:
                predictions_all = np.vstack((predictions_all, check_output_all.cpu()))
                prediction = np.append(prediction, check_output.cpu())
                predicted_c = np.append(predicted_c, res.cpu())
            # print('everything else time: ' + str(time.time() - start_time))
            start_time = time.time()

        return predictions_all, predictions, predicted_c


def ensemble_aug_eval(n_iter, class_model, e_loader, with_temp_scal=False):
    for i in range(1, n_iter + 1):
        predictions_all, _, _ = eval(class_model, e_loader, with_temp_scal)
        if i == 1:
            ens_predictions_all = predictions_all
        else:
            ens_predictions_all += predictions_all

    temp_ens_preds = ens_predictions_all / n_iter
    check_output, res = torch.max(torch.tensor(temp_ens_preds, device='cuda'), 1)

    return temp_ens_preds, check_output, res


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    net_parser = argparse.ArgumentParser()
    parser.add_argument('--avg', default=None, help='name of the file with ensemble parameters')
    parser.add_argument('--da_n_iter', type=int, default=0, help='number of iterations for Data Augmentation ensemble')
    parser.add_argument('--calibrated', action='store_true', help='Boolean flag for applying temperature scaling')
    parser.add_argument('--dataset', default='isic2019', help='name of the dataset to use')
    parser.add_argument('--validation', action='store_true', help='Boolean flag for using validation set')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size during the training')
    parser.add_argument('--log_ens', action='store_true', help='Boolean flag for ensembling through logarithms')

    opt = parser.parse_args()
    if opt.dataset == 'isic2019' and opt.da_n_iter != 0:
        opt.dataset = 'isic2019_testwaugm'
    elif opt.da_n_iter != 0:
        opt.dataset = opt.dataset + '_waugm'

    print(opt)

    net_parser.add_argument('--network', default='resnet50',
                            choices=['resnet18', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169',
                                     'densenet201', 'densenet161', 'seresnext50', 'seresnext101'])
    net_parser.add_argument('--save_dir', help='directory where to save model weights')
    net_parser.add_argument('--dropout', action='store_true', help='Boolean flag for DropOut inclusion')
    net_parser.add_argument('--classes', '-c', type=int, nargs='+',
                            action='append', help='classes to train the model with')
    net_parser.add_argument('--load_epoch', type=int, default=0, help='load custom-trained models')
    net_parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    net_parser.add_argument('--batch_size', type=int, default=16, help='batch size during the training')
    net_parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    net_parser.add_argument('--loss', default='cross_entropy', choices=['cross_entropy', 'focal'])
    net_parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'])
    net_parser.add_argument('--scheduler', default='plateau', choices=['plateau', 'None'])
    net_parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    net_parser.add_argument('--size', type=int, default=512, help='size of images')
    net_parser.add_argument('--SRV', action='store_true', help='Boolean flag for training on remote server')
    net_parser.add_argument('--from_scratch', action='store_true',
                            help='Boolean flag for training a model from scratch')
    net_parser.add_argument('--augm_config', type=int, default=1,
                            help='configuration code for augmentation techniques choice')
    net_parser.add_argument('--cutout_pad', nargs='+', type=int, default=[0], help='cutout pad.')
    net_parser.add_argument('--cutout_holes', nargs='+', type=int, default=[0], help='number of cutout holes.')
    net_parser.add_argument('--mixup', type=float, default=0.0,
                            help='mixout coefficient. If 0 is provided no mixup is applied')

    ens_preds = None
    counter = 0.0
    w_acc_test = 0.0
    acc_test = 0.0
    start_time = time.time()
    f = open(opt.avg, "r")
    names = None
    base_eval_data_loader = get_dataset(dname=opt.dataset,
                                        size=512,
                                        # dataset_classes=n.classes,
                                        SRV=True,
                                        load_flag=True,
                                        batch_size=12,
                                        n_workers=4,
                                        augm_config=0,
                                        drop_last_flag=False)

    for line in f:
        if not line.strip():
            continue
        print(line)
        counter += 1.0
        net_opt = net_parser.parse_args(line.split())
        print(net_opt)

        n = ClassifyNet(net=net_opt.network,
                        dname='isic2019',
                        dropout=net_opt.dropout,
                        classes=net_opt.classes, l_r=net_opt.learning_rate, loss=net_opt.loss,
                        optimizer=net_opt.optimizer, scheduler=net_opt.scheduler, size=net_opt.size,
                        batch_size=opt.batch_size, n_workers=net_opt.workers, pretrained=(not net_opt.from_scratch),
                        augm_config=net_opt.augm_config, save_dir=net_opt.save_dir, mixup_coeff=net_opt.mixup,
                        cutout_params=[net_opt.cutout_holes, net_opt.cutout_pad], total_epochs=net_opt.epochs,
                        SRV=net_opt.SRV, no_logs=True, optimize_temp_scal=opt.calibrated)

        if not net_opt.load_epoch == 0:
            n.load(net_opt.load_epoch)

        eval_data_loader = get_dataset(dname=opt.dataset,
                                       size=n.size,
                                       dataset_classes=n.classes,
                                       SRV=n.SRV,
                                       load_flag=False,
                                       batch_size=12,
                                       n_workers=n.n_workers,
                                       augm_config=n.augm_config,
                                       drop_last_flag=False)
        if names is None:
            names = eval_data_loader.dataset.imgs
        eval_data_loader.dataset.load = True
        eval_data_loader.dataset.imgs = base_eval_data_loader.dataset.imgs

        if opt.da_n_iter != 0:
            predictions_all, predictions, predicted_c = ensemble_aug_eval(opt.da_n_iter, n, eval_data_loader,
                                                                          opt.calibrated)

        else:
            predictions_all, predictions, predicted_c = eval(n, eval_data_loader, opt.calibrated)

        if ens_preds is None:
            ens_preds = predictions_all

        else:
            ens_preds += predictions_all

    temp_ens_preds = ens_preds / counter

    check_output, res = torch.max(torch.tensor(temp_ens_preds, device='cuda'), 1)
    for n, r in zip(names, res):
        print(os.path.basename(n) + ': ' + class_dict[r.item()])

    avgname = os.path.basename(opt.avg)
    fname = opt.dataset + "_" + os.path.splitext(avgname)[0]
    if opt.calibrated:
        fname += "_calibrated"
    if opt.da_n_iter > 0:
        fname += "_" + str(opt.da_n_iter) + "DAiter"
    if opt.validation:
        fname += "_validation"

    np.savetxt("/nas/softechict-nas-1/fpollastri/isic_submission_output/cropped_output_" + fname + ".csv", temp_ens_preds,
               delimiter=",")
    np.save("/nas/softechict-nas-1/fpollastri/isic_submission_output/cropped_output_" + fname + ".npy", temp_ens_preds)
