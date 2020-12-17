import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import torch
import time
import numpy as np
from utils_metrics import compute_accuracy_metrics
import csv

from classification_net import ClassifyNet, eval, ensemble_aug_eval
from utils import ConfusionMatrix, compute_calibration_measures

import config

def make_submission_csv(file_name, img_names, preds):
    with open(file_name, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['image_name', 'target'])
        for i in range(len(img_names)):
            csv_writer.writerow([img_names[i], f'{preds[i].item():.3f}'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    net_parser = argparse.ArgumentParser()
    parser.add_argument('--avg', default=None, help='name of the file with ensemble parameters')
    parser.add_argument('--da_n_iter', type=int, default=0, help='number of iterations for Data Augmentation ensemble')
    parser.add_argument('--calibrated', action='store_true', help='Boolean flag for applying temperature scaling')
    parser.add_argument('--dataset', default='isic2020_inference', help='name of the dataset to use')
    parser.add_argument('--validation', action='store_true', help='Boolean flag for using validation set')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size during the training')
    parser.add_argument('--log_ens', action='store_true', help='Boolean flag for ensembling through logarithms')
    parser.add_argument('--copy_into_tmp', action='store_true', help='Boolean flag for copying dataset into /tmp')

    opt = parser.parse_args()
    if opt.dataset == 'isic2018_test' and opt.da_n_iter != 0:
        opt.dataset = 'isic2018_test_waugm'
    if opt.dataset == 'isic2019' and opt.da_n_iter != 0:
        opt.dataset = 'isic2019_testwaugm'
    if opt.da_n_iter != 0:
        opt.dataset += '_waugm'

    print(opt)

    net_parser.add_argument('--network', default='resnet50',
                            choices=['resnet18', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169',
                                     'densenet201', 'densenet161', 'seresnext50', 'seresnext101'])
    net_parser.add_argument('--save_dir', help='directory where to save model weights')
    net_parser.add_argument('--dropout', action='store_true', help='Boolean flag for DropOut inclusion')
    net_parser.add_argument('--classes', '-c', type=int, nargs='+',
                            action='append', help='classes to train the model with')
    net_parser.add_argument('--load_epoch', type=int, default=0, help='load custom-trained models')
    net_parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    net_parser.add_argument('--batch_size', type=int, default=16, help='batch size during the training')
    net_parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    net_parser.add_argument('--loss', default='cross_entropy', choices=['cross_entropy', 'focal', 'combo'])
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
    net_parser.add_argument('--pretrained_isic', action='store_true', help='Pretrained on ISIC2019')

    output_path = config.ensemble_output_path
    avgname = os.path.basename(opt.avg)
    fname = opt.dataset + "_" + os.path.splitext(avgname)[0]
    if opt.calibrated:
        fname += "_calibrated"
    if opt.da_n_iter > 0:
        fname += "_" + str(opt.da_n_iter) + "DAiter"
    if opt.validation:
        fname += "_validation"

    ens_preds = None
    counter = 0.0
    w_acc_test = 0.0
    auc_test = 0.0
    start_time = time.time()
    f = open(opt.avg, "r")
    for line in f:
        if not line.strip():
            continue
        print(line)
        counter += 1.0
        net_opt = net_parser.parse_args(line.split())
        print(net_opt)

        output_file = os.path.join(output_path, "output_" + fname + f"_{int(counter)}.npy")

        if os.path.exists(output_file):
            print('Loading prediction file...')
            try:
                if ens_preds is None:
                    ens_preds = np.load(output_file)
                else:
                    ens_preds += np.load(output_file)
            except Exception as e:
                print(f'Exception {e}')
            continue

        n = ClassifyNet(net=net_opt.network, dname=opt.dataset, dropout=net_opt.dropout,
                        classes=net_opt.classes, l_r=net_opt.learning_rate, loss=net_opt.loss,
                        optimizer=net_opt.optimizer, scheduler=net_opt.scheduler, size=net_opt.size,
                        batch_size=opt.batch_size, n_workers=net_opt.workers, pretrained=(not net_opt.from_scratch),
                        augm_config=net_opt.augm_config, save_dir=net_opt.save_dir, mixup_coeff=net_opt.mixup,
                        cutout_params=[net_opt.cutout_holes, net_opt.cutout_pad], total_epochs=net_opt.epochs,
                        SRV=net_opt.SRV, no_logs=True, optimize_temp_scal=opt.calibrated,
                        copy_into_tmp=opt.copy_into_tmp, pretrained_isic=net_opt.pretrained_isic)

        if not net_opt.load_epoch == 0:
            n.load(net_opt.load_epoch)

        if opt.validation:
            n.test_data_loader = n.valid_data_loader
            n.calibration_variables[2] = n.calibration_variables[1]

        if opt.da_n_iter != 0:
            acc, w_acc, conf_matrix, acc_1, pr, rec, fscore, auc, preds, true_lab = ensemble_aug_eval(opt.da_n_iter, n,
                                                                                                      opt.calibrated)

        else:
            acc, w_acc, conf_matrix, acc_1, pr, rec, fscore, auc, preds, true_lab = eval(n, n.test_data_loader,
                                                                                         *n.calibration_variables[2],
                                                                                         opt.calibrated)

        if not isinstance(preds, np.ndarray):
            preds = preds.numpy()

        np.save(output_file, preds)     # only save predictions of this line

        try:
            auc_test += auc
            w_acc_test += w_acc
            if ens_preds is None:
                ens_preds = preds
            else:
                ens_preds += preds
        except Exception as e:
            print(f'Exception {e}')

    conf_matrix_test = ConfusionMatrix(n.num_classes)
    temp_ens_preds = ens_preds / counter

    check_output, res = torch.max(torch.tensor(temp_ens_preds, device='cuda'), 1)
    conf_matrix_test.update_matrix(res, torch.tensor(true_lab, device='cuda'))

    ens_acc, ens_w_acc = conf_matrix_test.get_metrics()
    ens_acc_1, pr, rec, fscore, auc = compute_accuracy_metrics(temp_ens_preds, true_lab)

    print("\n ----- FINAL PRINT ----- \n")

    print("\n|| took {:.1f} minutes \n"
          "| Mean Accuracy statistics: Weighted Acc: {:.3f} AUC: {:.3f} \n"
          "| Ensemble Accuracy statistics: Weighted Acc: {:.3f} AUC: {:.3f} Recall: {:.3f} Precision: {:.3f} Fscore: {:.3f} \n"
          .format((time.time() - start_time) / 60., w_acc_test / counter, auc_test / counter, ens_w_acc, auc, rec, pr,
                  fscore))
    print(conf_matrix_test.conf_matrix)

    np.save(os.path.join(output_path, "output_" + fname + ".npy"), temp_ens_preds)
    make_submission_csv(os.path.join(output_path, "output_" + fname + ".csv"), n.test_data_loader.dataset.split_list,
                        temp_ens_preds[:, 1])  # TODO: try this
