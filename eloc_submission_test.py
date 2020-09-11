import argparse
import torch
from torch import nn
import torch.nn.functional as F

import os
import time
import numpy as np
import itertools
from classification_net import ClassifyNet
from utils import ConfusionMatrix, compute_calibration_measures, entropy_categorical
from data import get_dataloader
from eloc_test import eloc_eval, eloc_ensemble_aug_eval, eloc_output
from torch.utils.data import DataLoader
from torchvision import transforms

from birds import birds_caltech_2011


def submission_eloc_output(temp_ens_preds, true_lab, t_test=None):
    conf_matrix_test = ConfusionMatrix(opt.tot_num_classes)
    check_output, res = torch.max(torch.tensor(temp_ens_preds, device='cpu'), 1)
    conf_matrix_test.update_matrix(res, torch.tensor(true_lab, device='cpu'))

    ens_acc, ens_w_acc = conf_matrix_test.get_metrics()
    ECE_test, MCE_test, BRIER_test, NNL_test = compute_calibration_measures(temp_ens_preds, true_lab,
                                                                            apply_softmax=False,
                                                                            bins=15)

    total_entropy = entropy_categorical(temp_ens_preds).cpu()

    print("\n|| took {:.1f} minutes \n"
          "| Mean Accuracy statistics: weighted Acc test: {:.3f} Acc test: {:.3f} \n"
          "| Ensemble Accuracy statistics: weighted Acc test: {:.3f} Acc test: {:.3f} \n"
          "| Calibration test: ECE: {:.5f} MCE: {:.5f} BRIER: {:.5f}  NNL: {:.5f} \n"
          "| Entropy Mean: {:.5f} Entropy Std {:.5f}\n"
          "| Confidence Mean: {:.5f} Confidence Std: {:.5f} \n\n".
          format((time.time() - start_time) / 60., w_acc_test / counter, acc_test / counter, ens_w_acc, ens_acc,
                 ECE_test * 100, MCE_test * 100, BRIER_test, NNL_test, total_entropy.mean(), total_entropy.std(),
                 check_output.mean(), check_output.std()))
    print(conf_matrix_test.conf_matrix)

    # compute paper score: softmax - entropy
    total_score = check_output - total_entropy

    if t_test is not None:

        ood_bin_preds = []

        for t in t_test:
            id_ratio = float((total_score > t).sum()) / float(len(total_score))
            print('\n fixed t: ' + str(t) + ' | TPR: ' + str(id_ratio) + '\n')

            ood_bin_preds.append(total_score > t)
        return ood_bin_preds

    else:
        thresholds = []

        best_ratio = 0.95

        for t in np.arange(total_score.min(), total_score.max(), (total_score.max() - total_score.min()) / 1000):
            id_ratio = float((total_score > t).sum()) / float(len(total_score))
            # if corr_ratio > 0.75 and ood_ratio < 0.98:
            if id_ratio < best_ratio:
                print('\n t: ' + str(t) + ' | TPR: ' + str(id_ratio) + '\n')
                thresholds.append(t)
                best_ratio -= 0.05
                if best_ratio < 0.49:
                    return thresholds


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--avg', default=None, help='name of the file with ensemble parameters')
    parser.add_argument('--da_n_iter', type=int, default=0, help='number of iterations for Data Augmentation ensemble')
    parser.add_argument('--calibrated', action='store_true', help='Boolean flag for applying temperature scaling')
    parser.add_argument('--validation', action='store_true', help='Boolean flag for using validation set')
    parser.add_argument('--tot_num_classes', type=int, default=8, help='number of total classes')
    parser.add_argument('--eps', type=float, default=0.002, help='number of total classes')
    parser.add_argument('--temperature', type=float, default=1000, help='number of total classes')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size during inference')
    parser.add_argument('--load', type=int, default=0, help='number to lines to skip if loading previous results')
    parser.add_argument('--threshods_file', default=None, help='name of the file with threshold parameters')

    opt = parser.parse_args()
    print(opt)

    net_parser = argparse.ArgumentParser()
    net_parser.add_argument('--network', default='resnet50',
                            choices=['resnet18', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169',
                                     'densenet201', 'densenet161'])
    net_parser.add_argument('--save_dir', help='directory where to save model weights')
    net_parser.add_argument('--dataset', default='isic2019', help='name of the dataset to use')
    net_parser.add_argument('--dropout', action='store_true', help='Boolean flag for DropOut inclusion')
    net_parser.add_argument('--classes', '-c', type=int, nargs='+',
                            action='append', help='classes to train the model with')
    net_parser.add_argument('--load_epoch', type=int, default=0, help='load custom-trained models')
    net_parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
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
    net_parser.add_argument('--ood_classes', '-ood', type=int, nargs='+',
                            action='append', help='ood classes to train the model with')

    avgname = os.path.basename(opt.avg)
    fname = "official_testset_submission_" + os.path.splitext(avgname)[0]
    if opt.calibrated:
        fname += "_calibrated"
    if opt.da_n_iter > 0:
        fname += "_" + str(opt.da_n_iter) + "DAiter"
    if opt.validation:
        fname += "_validation"

    if opt.load > 0:
        ens_bin_preds = []
        for key, value in np.load(
                "/nas/softechict-nas-1/fpollastri/OOD_ARRS/submission_preds_" + fname + "_OOD_bin_preds.npz").items():
            ens_bin_preds.append(torch.Tensor(value).to(dtype=torch.uint8))
        ens_preds = torch.Tensor(
            np.load("/nas/softechict-nas-1/fpollastri/OOD_ARRS/submission_preds_" + fname + "_OOD_ens_preds.npy"))

        # FILL TRUE LABELS
        n_samples = len(ens_preds)
        true_lab = torch.zeros(n_samples, ).long()

    else:
        bin_preds = None
        ens_preds = None

    counter = 0.0
    w_acc_test = 0.0
    acc_test = 0.0
    start_time = time.time()
    f = open(opt.avg, "r")
    th_f = open(opt.threshods_file, "r")
    for line in f:
        if not line.strip():
            continue
        print(line)
        counter += 1.0
        net_opt = net_parser.parse_args(line.split())
        print(net_opt)
        if counter <= opt.load:
            th_line = th_f.readline()
            while not th_line.strip():
                th_line = th_f.readline()
            print(th_line)
            print('already computed')
            continue

        n = ClassifyNet(net=net_opt.network, dname=net_opt.dataset, dropout=net_opt.dropout,
                        classes=net_opt.classes, l_r=net_opt.learning_rate, loss=net_opt.loss,
                        optimizer=net_opt.optimizer, scheduler=net_opt.scheduler, size=net_opt.size,
                        batch_size=opt.batch_size, n_workers=net_opt.workers, pretrained=(not net_opt.from_scratch),
                        augm_config=net_opt.augm_config, save_dir=net_opt.save_dir, mixup_coeff=net_opt.mixup,
                        cutout_params=[net_opt.cutout_holes, net_opt.cutout_pad], total_epochs=net_opt.epochs,
                        SRV=net_opt.SRV, no_logs=True, optimize_temp_scal=opt.calibrated)

        if not net_opt.load_epoch == 0:
            n.load(net_opt.load_epoch)

        if opt.da_n_iter == 0:

            n.data_loader, n.test_data_loader, n.valid_data_loader = get_dataloader(dname='isic2019_test',
                                                                                    SRV=True,
                                                                                    batch_size=opt.batch_size)
        else:
            n.data_loader, n.test_data_loader, n.valid_data_loader = get_dataloader(dname='isic2019_test_waugm',
                                                                                    SRV=True,
                                                                                    batch_size=opt.batch_size,
                                                                                    augm_config=net_opt.augm_config,
                                                                                    cutout_params=[net_opt.cutout_holes,
                                                                                                net_opt.cutout_pad])

        # if opt.OOD is not None:
        # to measures calibration stuff
        predictions_train = torch.zeros(len(n.data_loader.dataset), opt.tot_num_classes).float()
        labels_train = torch.zeros(len(n.data_loader.dataset), ).long()
        predictions_valid = torch.zeros(len(n.valid_data_loader.dataset), opt.tot_num_classes).float()
        labels_valid = torch.zeros(len(n.valid_data_loader.dataset), ).long()
        predictions_test = torch.zeros(len(n.test_data_loader.dataset), opt.tot_num_classes).float()
        labels_test = torch.zeros(len(n.test_data_loader.dataset), ).long()

        n.calibration_variables = [[predictions_train, labels_train], [predictions_valid, labels_valid],
                                   [predictions_test, labels_test]]

        if opt.validation:
            n.test_data_loader = n.valid_data_loader
            n.calibration_variables[2] = n.calibration_variables[1]
        if opt.da_n_iter != 0:
            acc, w_acc, preds, true_lab = eloc_ensemble_aug_eval(opt.da_n_iter, n, eps=opt.eps,
                                                                 temperature=opt.temperature,
                                                                 with_temp_scal=opt.calibrated,
                                                                 total_num_classes=opt.tot_num_classes)

        else:
            acc, w_acc, calib, conf_matrix, _ = eloc_eval(n, n.test_data_loader, *n.calibration_variables[2],
                                                          eps=opt.eps,
                                                          temperature=opt.temperature,
                                                          with_temp_scal=opt.calibrated,
                                                          total_num_classes=opt.tot_num_classes)
            _, preds, true_lab = calib

        acc_test += acc
        w_acc_test += w_acc

        th_line = th_f.readline()
        while not th_line.strip():
            th_line = th_f.readline()
        t_l = [float(s_t) for s_t in th_line.split('\t')]

        bin_preds = submission_eloc_output(temp_ens_preds=preds, true_lab=true_lab, t_test=t_l)
        # print(bin_preds)

        if ens_preds is None:
            ens_preds = preds
            ens_bin_preds = bin_preds
        else:
            ens_preds += preds
            if ens_bin_preds is not None:
                temp_bin_preds = []
                for ens_v, temp_v in zip(ens_bin_preds, bin_preds):
                    temp_bin_preds.append(ens_v + temp_v)
                ens_bin_preds = temp_bin_preds

        if opt.da_n_iter > 0:
            np.savez("/nas/softechict-nas-1/fpollastri/OOD_ARRS/submission_preds_" + fname + "_OOD_bin_preds",
                     *ens_bin_preds)
            np.save("/nas/softechict-nas-1/fpollastri/OOD_ARRS/submission_preds_" + fname + "_OOD_ens_preds", ens_preds)

    temp_ens_preds = ens_preds / counter

    print("\n ----- FINAL PRINT ----- \n")

    # print(ens_bin_preds)

    th_line = th_f.readline()
    while not th_line.strip():
        th_line = th_f.readline()
    print(th_line)
    final_t_l = [float(s_t) for s_t in th_line.split('\t')]

    fin_out = submission_eloc_output(temp_ens_preds=temp_ens_preds, true_lab=true_lab, t_test=final_t_l)
    print(fin_out)
    starting_tpr = 0.95
    for rt_out in ens_bin_preds:

        print('tpr: ' + str(starting_tpr))
        for nt in range(1, int(counter)):
            id_ratio = float((rt_out > np.float64(nt)).sum()) / float(len(rt_out))
            print('\n t: ' + str(nt) + ' | TPR: ' + str(id_ratio) + '\n')
            if 0.94 < starting_tpr < 0.96 and nt == 5:
                np.savetxt("/nas/softechict-nas-1/fpollastri/submission_labels_" + fname + "_voting090.csv",
                           (rt_out > np.float64(nt)).to(dtype=torch.float), delimiter=",")
        starting_tpr -= 0.05
