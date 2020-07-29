import argparse
import torch
from torch import nn
import torch.nn.functional as F

import time
import numpy as np
import itertools
from classification_net import ClassifyNet
from utils import ConfusionMatrix, compute_calibration_measures, entropy_categorical
from data import get_dataset
from GAOODD import GAOODN
from torch.utils.data import DataLoader
from torchvision import transforms

from birds import birds_caltech_2011


def eloc_eval(class_model, e_loader, predictions, labels, eps=0.02, temperature=1000, with_temp_scal=False,
              compute_separate_metrics_for_errors=False, total_num_classes=8):
    entropy_of_predictions = torch.zeros_like(labels).float()

    '''to measure stuff on correct and incorrect classified samples'''
    if compute_separate_metrics_for_errors:
        corr_entropy = torch.zeros_like(labels).float()
        incorr_entropy = torch.zeros_like(labels).float()

        corr_labels = torch.zeros_like(labels).float()
        incorr_labels = torch.zeros_like(labels).float()

        corr_predictions = torch.zeros_like(predictions).float()
        incorr_predictions = torch.zeros_like(predictions).float()

        corr_count = 0
        incorr_count = 0

    class_model.n.eval()
    sofmx = nn.Softmax(dim=-1)
    conf_matrix = ConfusionMatrix(total_num_classes)

    start_time = time.time()
    for idx, (x, target, img_name) in enumerate(e_loader):
        # measure data loading time
        # print("data time: " + str(time.time() - start_time))
        # compute output
        x = x.to('cuda')
        x.requires_grad = True
        out = class_model.n(x)
        loss = -torch.mean(torch.sum(F.log_softmax(out, dim=1) * F.softmax(out, dim=1), dim=1))
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        # mean: tensor([0.6681, 0.5301, 0.5247]) | std: tensor([0.1337, 0.1480, 0.1595])
        gradient[0, 0, :, :] = (gradient[0, 0, :, :]) / 0.1337
        gradient[0, 1, :, :] = (gradient[0, 1, :, :]) / 0.1480
        gradient[0, 2, :, :] = (gradient[0, 2, :, :]) / 0.1595
        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -eps, gradient)
        out = class_model.n(tempInputs)
        out = out / temperature

        if with_temp_scal:
            out = class_model.temp_scal_model(out)

        # output = torch.squeeze(out)
        output = out

        target = target.to('cuda', torch.long)

        temp_check_output_all = sofmx(output)

        # fix output indexes
        check_output_all = torch.zeros(output.size(0), total_num_classes)
        for c_i, net_c in enumerate(n.classes):
            check_output_all[:, net_c[0]] += temp_check_output_all[:, c_i].data.cpu()

        check_output, res = torch.max(check_output_all, -1)
        # res = torch.tensor([n.classes[r][0] for r in res])

        aux = target.size(0)
        predictions[idx * class_model.batch_size:idx * class_model.batch_size + aux,
        :] = check_output_all.data.cpu()
        labels[idx * class_model.batch_size:idx * class_model.batch_size + aux] = target.data.cpu()

        entropy_of_predictions[
        idx * class_model.batch_size:idx * class_model.batch_size + aux] = entropy_categorical(
            check_output_all).cpu()

        # update the confusion matrix
        conf_matrix.update_matrix(res, target)
        # measure batch time
        # print("batch " + str(idx) + " of " + str(len(e_loader)) + "; time: " + str(time.time() - start_time))
        # start_time = time.time()

        # if idx == 0:
        #     break

        if compute_separate_metrics_for_errors:
            # if true we compute the entropy and calibration measures on correct and incorrect samples separately
            corr_idx = check_output_all.argmax(dim=1) == target
            incorr_idx = check_output_all.argmax(dim=1) != target

            corr_samples_prob = check_output_all[corr_idx, :]
            incorr_samples_prob = check_output_all[incorr_idx, :]

            corr_numel = corr_idx.sum().long()
            incorr_numel = incorr_idx.sum().long()

            corr_entropy[corr_count:corr_count + corr_numel] = entropy_categorical(corr_samples_prob).cpu()
            incorr_entropy[incorr_count:incorr_count + incorr_numel] = entropy_categorical(
                incorr_samples_prob).cpu()

            corr_predictions[corr_count:corr_count + corr_numel] = corr_samples_prob.cpu()
            incorr_predictions[incorr_count:incorr_count + incorr_numel] = incorr_samples_prob.cpu()

            corr_labels[corr_count:corr_count + corr_numel] = target[corr_idx].cpu()
            incorr_labels[incorr_count:incorr_count + incorr_numel] = target[incorr_idx].cpu()

            corr_count += corr_numel
            incorr_count += incorr_numel

    # filter out the zeros
    per_samples_stats = None
    if compute_separate_metrics_for_errors:
        corr_entropy = corr_entropy[0:corr_count]
        incorr_entropy = incorr_entropy[0:incorr_count]

        corr_predictions = corr_predictions[0:corr_count]
        incorr_predictions = incorr_predictions[0:incorr_count]

        corr_labels = corr_labels[0:corr_count]
        incorr_labels = incorr_labels[0:incorr_count]

        per_samples_stats = {
            'corr': [corr_entropy, corr_predictions, corr_labels],
            'incorr': [incorr_entropy, incorr_predictions, incorr_labels]
        }

    acc, w_acc = conf_matrix.get_metrics()

    return acc, w_acc, [entropy_of_predictions, predictions, labels], conf_matrix.conf_matrix, per_samples_stats


def eloc_ensemble_aug_eval(n_iter, class_model, eps=0.002, temperature=1000, with_temp_scal=False, total_num_classes=8):
    acc_test = 0
    w_acc_test = 0
    ens_preds = torch.zeros_like(class_model.calibration_variables[2][0])

    start_time = time.time()
    # data_loader, test_data_loader, valid_data_loader = get_dataset(dname='isic2019_testwaugm', size=class_model.size,
    #                                                                SRV=class_model.SRV,
    #                                                                batch_size=class_model.batch_size,
    #                                                                n_workers=class_model.n_workers,
    #                                                                augm_config=class_model.augm_config,
    #                                                                cutout_params=[class_model.cutout_nholes,
    #                                                                               class_model.cutout_pad_size])

    for i in range(1, n_iter + 1):
        acc_test_temp, w_acc_test_temp, calibration_statistics, conf_matrix_temp, _ = \
            eloc_eval(n, n.test_data_loader, *n.calibration_variables[2],
                      eps=eps,
                      temperature=temperature,
                      with_temp_scal=with_temp_scal,
                      total_num_classes=total_num_classes)
        # print('iteration ' + str(i) + ' completed in ' + str(time.time()-start_time) + ' seconds')
        # print('Acc: ' + str(acc_test_temp) + ' | Weighted Acc: ' + str(w_acc_test_temp) + '\n')
        # print(conf_matrix_temp)
        acc_test += acc_test_temp
        w_acc_test += w_acc_test_temp

        _, preds, true_lab = calibration_statistics
        ens_preds += preds

    conf_matrix_test = ConfusionMatrix(total_num_classes)
    temp_ens_preds = ens_preds / n_iter
    check_output, res = torch.max(torch.tensor(temp_ens_preds, device='cuda'), 1)
    conf_matrix_test.update_matrix(res, torch.tensor(true_lab, device='cuda'))

    ens_acc, ens_w_acc = conf_matrix_test.get_metrics()
    ECE_test, MCE_test, BRIER_test, NNL_test = compute_calibration_measures(temp_ens_preds, true_lab,
                                                                            apply_softmax=False,
                                                                            bins=15)
    print("\n|| took {:.1f} minutes \n"
          "| Mean Accuracy statistics: weighted Acc test: {:.3f} Acc test: {:.3f} \n"
          "| Ensemble Accuracy statistics: weighted Acc test: {:.3f} Acc test: {:.3f} \n"
          "| Calibration test: ECE: {:.5f} MCE: {:.5f} BRIER: {:.5f}  NNL: {:.5f}\n\n".
          format((time.time() - start_time) / 60., w_acc_test / i, acc_test / i, ens_w_acc, ens_acc,
                 ECE_test * 100, MCE_test * 100, BRIER_test, NNL_test))
    print(conf_matrix_test.conf_matrix)

    return ens_acc, ens_w_acc, (ens_preds / n_iter), true_lab


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    net_parser = argparse.ArgumentParser()
    parser.add_argument('--avg', default=None, help='name of the file with ensemble parameters')
    parser.add_argument('--da_n_iter', type=int, default=0, help='number of iterations for Data Augmentation ensemble')
    parser.add_argument('--calibrated', action='store_true', help='Boolean flag for applying temperature scaling')
    parser.add_argument('--validation', action='store_true', help='Boolean flag for using validation set')
    parser.add_argument('--OOD', type=int, default=None, help='OOD class to check')
    parser.add_argument('--tot_num_classes', type=int, default=8, help='number of total classes')
    parser.add_argument('--eps', type=float, default=0.002, help='number of total classes')
    parser.add_argument('--temperature', type=float, default=1000, help='number of total classes')

    opt = parser.parse_args()
    print(opt)

    net_parser.add_argument('--network', default='resnet50',
                            choices=['resnet18', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169',
                                     'densenet201', 'densenet161'])
    net_parser.add_argument('--save_dir', required=True, help='directory where to save model weights')
    net_parser.add_argument('--dataset', default='isic2019', help='name of the dataset to use')
    net_parser.add_argument('--dropout', action='store_true', help='Boolean flag for DropOut inclusion')
    net_parser.add_argument('--classes', '-c', type=int, nargs='+',
                            # , default=[[0], [1], [2], [3], [4], [5], [6], [7]]
                            action='append', help='classes to train the model with')
    net_parser.add_argument('--ood_classes', '-ood', type=int, nargs='+',
                            action='append', help='ood classes to train the model with')
    net_parser.add_argument('--true_ood_classes', '-t_ood', type=int, nargs='+',
                            action='append', help='Real ood classes, to TEST the model against')
    net_parser.add_argument('--load_epoch_net', type=int, default=0, help='load custom-trained models')
    net_parser.add_argument('--load_epoch_gan', type=int, default=0, help='load custom-trained models')
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
    net_parser.add_argument('--calibrated', action='store_true', help='Boolean flag for applying temperature scaling')

    net_parser.add_argument('--nz', type=int, default=100, help='size of latent vector')
    net_parser.add_argument('--ngf', type=int, default=512, help='size of G layers')
    net_parser.add_argument('--ndf', type=int, default=512, help='size of D layers')

    ens_preds = None
    counter = 0.0
    w_acc_test = 0.0
    acc_test = 0.0
    start_time = time.time()
    f = open(opt.avg, "r")
    for line in f:
        if not line.strip():
            continue
        print(line)
        counter += 1.0
        net_opt = net_parser.parse_args(line.split())
        print(net_opt)
        if net_opt.classes is None:
            classes = [[0], [1], [2], [3], [4], [5], [6], [7]]
        elif len(net_opt.classes) == 1:
            classes = [[c] for c in net_opt.classes[0]]
        else:
            classes = net_opt.classes
        num_classes = len(classes)

        n = ClassifyNet(net=net_opt.network, dname=net_opt.dataset, dropout=net_opt.dropout,
                        classes=net_opt.classes, l_r=net_opt.learning_rate, loss=net_opt.loss,
                        optimizer=net_opt.optimizer, scheduler=net_opt.scheduler, size=net_opt.size,
                        batch_size=net_opt.batch_size, n_workers=net_opt.workers, pretrained=(not net_opt.from_scratch),
                        augm_config=net_opt.augm_config, save_dir=net_opt.save_dir, mixup_coeff=net_opt.mixup,
                        cutout_params=[net_opt.cutout_holes, net_opt.cutout_pad], total_epochs=net_opt.epochs,
                        SRV=net_opt.SRV, no_logs=True, optimize_temp_scal=opt.calibrated)

        gan = GAOODN(net=net_opt.network, dname=net_opt.dataset, l_r=net_opt.learning_rate, save_dir=net_opt.save_dir,
                     total_epochs=net_opt.epochs, SRV=net_opt.SRV, OOD=net_opt.ood_classes, n_classes=num_classes,
                     n_features=1920,  # WARNING: HARD-CODED, TO FIX
                     nz=net_opt.nz, ngf=net_opt.ngf, ndf=net_opt.ndf, ngpu=1)

        if not net_opt.load_epoch_net == 0:
            n.load(net_opt.load_epoch_net)
        if not net_opt.load_epoch_gan == 0:
            n.load(net_opt.load_epoch_gan)

        n.data_loader, n.test_data_loader, n.valid_data_loader = get_dataset(dname='isic2019',
                                                                             SRV=True,
                                                                             batch_size=net_opt.batch_size)

        if opt.OOD is not None:
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
        if ens_preds is None:
            ens_preds = preds
        else:
            ens_preds += preds

    conf_matrix_test = ConfusionMatrix(opt.tot_num_classes)
    temp_ens_preds = ens_preds / counter
    check_output, res = torch.max(torch.tensor(temp_ens_preds, device='cpu'), 1)
    conf_matrix_test.update_matrix(res, torch.tensor(true_lab, device='cpu'))

    ens_acc, ens_w_acc = conf_matrix_test.get_metrics()
    ECE_test, MCE_test, BRIER_test, NNL_test = compute_calibration_measures(temp_ens_preds, true_lab,
                                                                            apply_softmax=False,
                                                                            bins=15)

    total_entropy = entropy_categorical(temp_ens_preds).cpu()

    corr_idx = temp_ens_preds.argmax(dim=-1) == true_lab
    incorr_idx = temp_ens_preds.argmax(dim=-1) != true_lab

    corr_samples_prob = temp_ens_preds[corr_idx, :]
    incorr_samples_prob = temp_ens_preds[incorr_idx, :]

    corr_entropy = entropy_categorical(corr_samples_prob).cpu()
    incorr_entropy = entropy_categorical(incorr_samples_prob).cpu()

    corr_ECE_test, corr_MCE_test, corr_BRIER_test, corr_NNL_test = compute_calibration_measures(corr_samples_prob.cpu(),
                                                                                                true_lab[
                                                                                                    corr_idx].cpu(),
                                                                                                apply_softmax=False,
                                                                                                bins=15)

    incorr_ECE_test, incorr_MCE_test, incorr_BRIER_test, incorr_NNL_test = compute_calibration_measures(
        incorr_samples_prob.cpu(),
        true_lab[incorr_idx].cpu(),
        apply_softmax=False,
        bins=15)

    print("\n ----- FINAL PRINT ----- \n")

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

    print("\nCORRECT PREDICTED SAMPLES\n"
          "| Calibration test: ECE: {:.5f} MCE: {:.5f} BRIER: {:.5f}  NNL: {:.5f} \n"
          "| Entropy Mean: {:.5f} Entropy Std {:.5f} \n"
          "| Confidence Mean: {:.5f} Confidence Std: {:.5f} \n\n".
          format(corr_ECE_test * 100, corr_MCE_test * 100, corr_BRIER_test, corr_NNL_test, corr_entropy.mean(),
                 corr_entropy.std(), corr_samples_prob.max(dim=-1)[0].mean(), corr_samples_prob.max(dim=-1)[0].std()))

    print("\nINCORRECT PREDICTED SAMPLES\n"
          "| Calibration test: ECE: {:.5f} MCE: {:.5f} BRIER: {:.5f}  NNL: {:.5f} \n"
          "| Entropy Mean: {:.5f} Entropy Std {:.5f} \n"
          "| Confidence Mean: {:.5f} Confidence Std: {:.5f} \n\n".
          format(incorr_ECE_test * 100, incorr_MCE_test * 100, incorr_BRIER_test, incorr_NNL_test,
                 incorr_entropy.mean(), incorr_entropy.std(), incorr_samples_prob.max(dim=-1)[0].mean(),
                 incorr_samples_prob.max(dim=-1)[0].std()))

    cl_entropy = []
    for i in range(opt.tot_num_classes):
        cl_entropy.append([])
    for i, entr in enumerate(total_entropy):
        cl_entropy[true_lab[i]].append(entr)

    print('\n')
    print('\n')
    for i in range(opt.tot_num_classes):
        print(str(i) + ' mean: ' + str(np.mean(cl_entropy[i])))
        print(str(i) + ' std: ' + str(np.std(cl_entropy[i])))
        print('\n')

    in_cl_entropy = cl_entropy[:opt.OOD - 1] + cl_entropy[opt.OOD:]
    in_cl_entropy = list(itertools.chain.from_iterable(in_cl_entropy))
    for t in np.arange(total_entropy.min(), total_entropy.max(), (total_entropy.max() - total_entropy.min()) / 1000):
        ood_ratio = float((cl_entropy[opt.OOD] < t).sum()) / float(len(cl_entropy[opt.OOD]))
        id_ratio = float((in_cl_entropy < t).sum()) / float(len(in_cl_entropy))
        corr_ratio = float((corr_entropy < t).sum()) / float(len(corr_entropy))
        if corr_ratio > 0.75 and ood_ratio < 0.98:
            print('\n t: ' + str(t) + ' | FPR: ' + str(ood_ratio) + ' | TPR: ' + str(id_ratio) + ' | CTPR: ' + str(
                corr_ratio) + '\n')

    # compute paper score: softmax - entropy
    total_score = check_output - total_entropy
    cl_ood_score = []
    for i in range(opt.tot_num_classes):
        cl_ood_score.append([])
    for i, entr in enumerate(total_entropy):
        cl_ood_score[true_lab[i]].append(check_output[i] - entr)
    corr_ood_score = corr_samples_prob.max(dim=-1)[0] - corr_entropy
    in_cl_score = cl_ood_score[:opt.OOD - 1] + cl_ood_score[opt.OOD:]
    in_cl_score = list(itertools.chain.from_iterable(in_cl_score))
    print(' NOW WITH PAPER SCORE')
    for t in np.arange(total_score.min(), total_score.max(), (total_score.max() - total_score.min()) / 1000):
        ood_ratio = float((cl_ood_score[opt.OOD] > t).sum()) / float(len(cl_ood_score[opt.OOD]))
        id_ratio = float((in_cl_score > t).sum()) / float(len(in_cl_score))
        corr_ratio = float((corr_ood_score > t).sum()) / float(len(corr_ood_score))
        if corr_ratio > 0.75 and ood_ratio < 0.98:
            print('\n t: ' + str(t) + ' | FPR: ' + str(ood_ratio) + ' | TPR: ' + str(id_ratio) + ' | CTPR: ' + str(
                corr_ratio) + '\n')
