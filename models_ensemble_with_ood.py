import argparse
import torch
import time
import numpy as np
from classification_net import ClassifyNet, eval, ensemble_aug_eval
from utils import ConfusionMatrix, compute_calibration_measures, entropy_categorical
from data import get_dataloader
from torch.utils.data import DataLoader
from torchvision import transforms

from birds import birds_caltech_2011

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    net_parser = argparse.ArgumentParser()
    parser.add_argument('--avg', default=None, help='name of the file with ensemble parameters')
    parser.add_argument('--da_n_iter', type=int, default=0, help='number of iterations for Data Augmentation ensemble')
    parser.add_argument('--calibrated', action='store_true', help='Boolean flag for applying temperature scaling')
    parser.add_argument('--validation', action='store_true', help='Boolean flag for using validation set')
    parser.add_argument('--OOD', default=None, help='OOD class to check')

    opt = parser.parse_args()
    print(opt)

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

        n = ClassifyNet(net=net_opt.network, dname=net_opt.dataset, dropout=net_opt.dropout,
                        classes=net_opt.classes, l_r=net_opt.learning_rate, loss=net_opt.loss,
                        optimizer=net_opt.optimizer, scheduler=net_opt.scheduler, size=net_opt.size,
                        batch_size=net_opt.batch_size, n_workers=net_opt.workers, pretrained=(not net_opt.from_scratch),
                        augm_config=net_opt.augm_config, save_dir=net_opt.save_dir, mixup_coeff=net_opt.mixup,
                        cutout_params=[net_opt.cutout_holes, net_opt.cutout_pad], total_epochs=net_opt.epochs,
                        SRV=net_opt.SRV, no_logs=True, optimize_temp_scal=opt.calibrated)

        if not net_opt.load_epoch == 0:
            n.load(net_opt.load_epoch)

        if opt.OOD == 'isic':
            n_cl_list = [c[0] for c in n.classes]
            cl_list = [[c for c in range(8) if c not in n_cl_list]]
            n.data_loader, n.test_data_loader, n.valid_data_loader = get_dataloader(dname='isic2019',
                                                                                    dataset_classes=cl_list, SRV=True,
                                                                                    batch_size=net_opt.batch_size)
        elif opt.OOD == 'birds':
            birds_dataset = birds_caltech_2011('/nas/softechict-nas-1/fpollastri/data/birds/',
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.6681, 0.5301, 0.5247),
                                                                        (0.1337, 0.1480, 0.1595)),
                                               ]),
                                               target_transform=None, asnumpy=False, download=True, isTrain=False,
                                               tam_image=512, interpolation='bilinear', padding='wrap',
                                               return_bounding_box=False)
            n.test_data_loader = DataLoader(birds_dataset,
                                            batch_size=n.batch_size,
                                            shuffle=False,
                                            num_workers=n.n_workers,
                                            drop_last=False,
                                            pin_memory=True)

        if opt.OOD is not None:
            # to measures calibration stuff
            predictions_train = torch.zeros(len(n.data_loader.dataset), n.num_classes).float()
            labels_train = torch.zeros(len(n.data_loader.dataset), ).long()
            predictions_valid = torch.zeros(len(n.valid_data_loader.dataset), n.num_classes).float()
            labels_valid = torch.zeros(len(n.valid_data_loader.dataset), ).long()
            predictions_test = torch.zeros(len(n.test_data_loader.dataset), n.num_classes).float()
            labels_test = torch.zeros(len(n.test_data_loader.dataset), ).long()

            n.calibration_variables = [[predictions_train, labels_train], [predictions_valid, labels_valid],
                                       [predictions_test, labels_test]]

        if opt.validation:
            n.test_data_loader = n.valid_data_loader
            n.calibration_variables[2] = n.calibration_variables[1]
        if opt.da_n_iter != 0:
            acc, w_acc, preds, true_lab = ensemble_aug_eval(opt.da_n_iter, n, opt.calibrated)

        else:
            acc, w_acc, calib, conf_matrix, _ = eval(n, n.test_data_loader, *n.calibration_variables[2],
                                                     opt.calibrated)
            _, preds, true_lab = calib

        acc_test += acc
        w_acc_test += w_acc
        if ens_preds is None:
            ens_preds = preds
        else:
            ens_preds += preds

    conf_matrix_test = ConfusionMatrix(n.num_classes)
    temp_ens_preds = ens_preds / counter
    check_output, res = torch.max(torch.tensor(temp_ens_preds, device='cuda'), 1)
    conf_matrix_test.update_matrix(res, torch.tensor(true_lab, device='cuda'))

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

    if opt.OOD is None:
        for t in np.arange(0.1, 1.8, 0.05):
            total_ratio = (total_entropy < t).sum().float() / float(len(total_entropy))
            corr_ratio = (corr_entropy < t).sum().float() / float(len(corr_entropy))
            print('\n t: ' + str(t) + ' | TPR: ' + str(total_ratio) + ' | CTPR: ' + str(corr_ratio) + '\n')
    else:
        for t in np.arange(0.1, 1.8, 0.05):
            total_ratio = (total_entropy < t).sum().float() / float(len(total_entropy))
            print('\n t: ' + str(t) + ' | FPR: ' + str(total_ratio) + '\n')

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
