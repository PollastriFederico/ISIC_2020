import torch
import time
import numpy as np
import argparse
import torch.nn.functional as F

from classification_net import ClassifyNet, eval
from utils import ConfusionMatrix, compute_calibration_measures, entropy_categorical
from data import get_dataset, get_dataset_squares_of_skin

from scipy.stats import wasserstein_distance


def eloc_train(class_model, ood_data, sood_data, rood_data, n_sood_iter, beta, m_coeff, num_epochs, starting_e=0):
    ood_train_dataloader, ood_test_dataloader, ood_valid_dataloader = ood_data
    sood_train_dataloader, sood_test_dataloader, sood_valid_dataloader = sood_data
    rood_train_dataloader, rood_test_dataloader, rood_valid_dataloader = rood_data
    ood_train_iter = iter(ood_train_dataloader)
    s_ood_train_iter = iter(sood_train_dataloader)
    best_ood_score_1 = 0.0
    best_ood_score_2 = 0.0
    for epoch in range(starting_e + 1, num_epochs):
        class_model.n.train()
        losses = []
        start_time = time.time()
        for idx, (x, target, _) in enumerate(class_model.data_loader):

            # if idx == 2:
            #     break

            if n_sood_iter < 1000 and (idx % n_sood_iter) == 0:
                try:
                    data_ood, _, _ = s_ood_train_iter.next()
                except:
                    s_ood_train_iter = iter(sood_train_dataloader)
                    data_ood, _, _ = s_ood_train_iter.next()

            else:
                try:
                    data_ood, _, _ = ood_train_iter.next()
                except:
                    ood_train_iter = iter(ood_train_dataloader)
                    data_ood, _, _ = ood_train_iter.next()

            # measure data loading time
            # print("data time: " + str(time.time() - start_time))

            # compute output
            x = x.to('cuda')
            target = target.to('cuda', torch.long)
            output = class_model.n(x)

            E_id = -torch.mean(torch.sum(F.log_softmax(output, dim=1) * F.softmax(output, dim=1), dim=1))
            output_ood = class_model.n(data_ood.to('cuda'))
            E_ood = -torch.mean(torch.sum(F.log_softmax(output_ood, dim=1) * F.softmax(output_ood, dim=1), dim=1))

            output = torch.squeeze(output)
            crit = class_model.criterion(output, target)

            loss = crit + beta * torch.clamp(m_coeff + E_id - E_ood, min=0)
            losses.append(loss.item())
            # compute gradient and do SGD step
            class_model.optimizer.zero_grad()
            loss.backward()
            class_model.optimizer.step()
            # print('idx: ' + str(idx) + " | time: " + str(time.time()-start_time))

        _, _, calibration_statistics, _, _ = eval(class_model,
                                                  ood_valid_dataloader,
                                                  torch.zeros(len(ood_valid_dataloader.dataset),
                                                              class_model.num_classes),
                                                  torch.zeros(len(ood_valid_dataloader.dataset)),
                                                  class_model.optimize_temp_scal)
        ood_entropy, _, _ = calibration_statistics

        _, _, calibration_statistics, _, _ = eval(class_model,
                                                  rood_test_dataloader,
                                                  torch.zeros(len(rood_test_dataloader.dataset),
                                                              class_model.num_classes),
                                                  torch.zeros(len(rood_test_dataloader.dataset)),
                                                  class_model.optimize_temp_scal)
        r_ood_entropy, _, _ = calibration_statistics

        _, _, calibration_statistics, _, _ = eval(class_model,
                                                  sood_test_dataloader,
                                                  torch.zeros(len(sood_test_dataloader.dataset),
                                                              class_model.num_classes),
                                                  torch.zeros(len(sood_test_dataloader.dataset)),
                                                  class_model.optimize_temp_scal)
        s_ood_entropy_test, _, _ = calibration_statistics

        acc_valid, w_acc_valid, calibration_statistics, conf_matrix_valid, _ = eval(class_model,
                                                                                    class_model.valid_data_loader,
                                                                                    *class_model.calibration_variables[
                                                                                        1],
                                                                                    class_model.optimize_temp_scal)

        iid_entropy, preds, true_lab = calibration_statistics
        ECE_valid, MCE_valid, BRIER_valid, NNL_valid = compute_calibration_measures(preds, true_lab,
                                                                                    apply_softmax=False, bins=15)

        # acc_test, w_acc_test, calibration_statistics, conf_matrix_test, _ = eval(class_model,
        #                                                                          class_model.test_data_loader,
        #                                                                          *class_model.calibration_variables[2],
        #                                                                          class_model.optimize_temp_scal)
        #
        # _, preds, true_lab = calibration_statistics
        # ECE_test, MCE_test, BRIER_test, NNL_test = compute_calibration_measures(preds, true_lab, apply_softmax=False,
        #                                                                         bins=15)
        acc_test, w_acc_test, ECE_test, MCE_test, BRIER_test, NNL_test = acc_valid, w_acc_valid, ECE_valid, MCE_valid, BRIER_valid, NNL_valid

        print("\n|| Epoch {} took {:.1f} minutes \t LossCE {:.5f} \n"
              "| Accuracy statistics: weighted Acc valid: {:.3f}  weighted Acc test: {:.3f} Acc valid: {:.3f}  Acc test: {:.3f} \n"
              "| Calibration valid: ECE: {:.5f} MCE: {:.3f} BRIER: {:.3f} NNL: {:.5f} \n"
              "| Calibration test: ECE: {:.5f} MCE: {:.5f} BRIER: {:.5f}  NNL: {:.5f}\n\n".format(epoch, (
                time.time() - start_time) / 60., np.mean(losses), w_acc_valid, w_acc_test, acc_valid,
                                                                                                  acc_test,
                                                                                                  ECE_valid * 100.,
                                                                                                  MCE_valid * 100.,
                                                                                                  BRIER_valid,
                                                                                                  NNL_valid,
                                                                                                  ECE_test * 100.,
                                                                                                  MCE_test * 100.,
                                                                                                  BRIER_test, NNL_test))

        print(conf_matrix_valid)
        print('\n')
        # print(conf_matrix_test)
        ood_score_1 = wasserstein_distance(iid_entropy, ood_entropy)
        print('ID Mean: ' + str(iid_entropy.mean().item()) + ' | Std: ' + str(iid_entropy.std().item()))
        print('OOD Mean: ' + str(ood_entropy.mean().item()) + ' | Std: ' + str(ood_entropy.std().item()))
        print('REAL OOD Mean: ' + str(r_ood_entropy.mean().item()) + ' | Std: ' + str(r_ood_entropy.std().item()))
        print('SECOND OOD Mean: ' + str(s_ood_entropy_test.mean().item()) + ' | Std: ' + str(
            s_ood_entropy_test.std().item()))
        print("OOD score: " + str(ood_score_1))
        print("Mean Diff: " + str(ood_entropy.mean().item() - iid_entropy.mean().item()))
        ood_score_2 = ood_entropy.mean().item() - ood_entropy.std().item() - \
                      iid_entropy.mean().item() - iid_entropy.std().item()
        print("Mean & Std Diff: " + str(ood_score_2))

        if epoch >= 20:

            if w_acc_valid > class_model.best_acc:
                print("SAVING MODEL BECAUSE OF WEIGHTED ACCURACY")
                class_model.save(epoch)
                class_model.best_acc = w_acc_valid
            if ood_score_1 > best_ood_score_1:
                print("SAVING MODEL BECAUSE OF OOD WASSERSTEIN DISTANCE")
                class_model.save(epoch)
                best_ood_score_1 = ood_score_1
            if ood_score_2 > best_ood_score_2:
                print("SAVING MODEL BECAUSE OF OOD STUPID DISTANCE")
                class_model.save(epoch)
                best_ood_score_2 = ood_score_2

            if epoch % 10 == 0:
                # print("SAVE MODEL BECAUSE OF EPOCH NUMBER")
                class_model.save(epoch)
        if num_epochs <= 20:
            class_model.save(epoch)

        if class_model.schedname == 'plateau':
            class_model.scheduler.step(w_acc_valid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--network', default='resnet50',
                        choices=['resnet18', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169',
                                 'densenet201', 'densenet161'])
    parser.add_argument('--save_dir', required=True, help='directory where to save model weights')
    parser.add_argument('--dataset', default='isic2019', help='name of the dataset to use')
    parser.add_argument('--dropout', action='store_true', help='Boolean flag for DropOut inclusion')
    parser.add_argument('--classes', '-c', type=int, nargs='+',  # , default=[[0], [1], [2], [3], [4], [5], [6], [7]]
                        action='append', help='classes to train the model with')
    parser.add_argument('--ood_classes', '-ood', type=int, nargs='+',
                        action='append', help='ood classes to train the model with')
    parser.add_argument('--load_epoch', type=int, default=0, help='load custom-trained models')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size during the training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--loss', default='cross_entropy', choices=['cross_entropy', 'focal'])
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--scheduler', default='plateau', choices=['plateau', 'None'])
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--size', type=int, default=512, help='size of images')
    parser.add_argument('--SRV', action='store_true', help='Boolean flag for training on remote server')
    parser.add_argument('--from_scratch', action='store_true', help='Boolean flag for training a model from scratch')
    parser.add_argument('--augm_config', type=int, default=1,
                        help='configuration code for augmentation techniques choice')
    parser.add_argument('--cutout_pad', nargs='+', type=int, default=[0], help='cutout pad.')
    parser.add_argument('--cutout_holes', nargs='+', type=int, default=[0], help='number of cutout holes.')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixout coefficient. If 0 is provided no mixup is applied')
    parser.add_argument('--calibrated', action='store_true', help='Boolean flag for applying temperature scaling')

    parser.add_argument('--m', type=float, default=0.4, help='m coefficient')
    parser.add_argument('--beta', type=float, default=0.2, help='beta coefficient')
    parser.add_argument('--n_sood_iter', type=int, default=10000,
                        help='number of iteration with proper ood class to perform each time one iteration with '
                             'synthetic ood images is performed')

    opt = parser.parse_args()
    print(opt)

    n = ClassifyNet(net=opt.network, dname=opt.dataset, dropout=opt.dropout, classes=opt.classes,
                    l_r=opt.learning_rate, loss=opt.loss, optimizer=opt.optimizer, scheduler=opt.scheduler,
                    size=opt.size, batch_size=opt.batch_size, n_workers=opt.workers, pretrained=(not opt.from_scratch),
                    augm_config=opt.augm_config, save_dir=opt.save_dir, mixup_coeff=opt.mixup,
                    cutout_params=[opt.cutout_holes, opt.cutout_pad], total_epochs=opt.epochs, SRV=opt.SRV,
                    no_logs=True, optimize_temp_scal=opt.calibrated, drop_last=True)

    ood_dataloaders = get_dataset(dname=opt.dataset, size=opt.size, dataset_classes=opt.ood_classes, SRV=True,
                                  batch_size=opt.batch_size, n_workers=opt.workers, augm_config=opt.augm_config,
                                  cutout_params=[opt.cutout_holes, opt.cutout_pad], drop_last_flag=True)

    sood_dataloaders = get_dataset_squares_of_skin(dname=opt.dataset, size=opt.size,
                                                   SRV=True,
                                                   batch_size=opt.batch_size, n_workers=opt.workers,
                                                   augm_config=1632,
                                                   cutout_params=[opt.cutout_holes, opt.cutout_pad],
                                                   drop_last_flag=True)

    rood_class = [[7]]

    rood_dataloaders = get_dataset(dname=opt.dataset, size=opt.size, dataset_classes=rood_class, SRV=True,
                                   batch_size=opt.batch_size, n_workers=opt.workers, augm_config=opt.augm_config,
                                   cutout_params=[opt.cutout_holes, opt.cutout_pad], drop_last_flag=True)

    if not opt.load_epoch == 0:
        n.load(opt.load_epoch)
        acc, w_acc, calib, conf_matrix, _ = eval(n, n.test_data_loader, *n.calibration_variables[2], opt.calibrated)
        print(acc)
        print(w_acc)
        print(conf_matrix)
        # ensemble_aug_eval(100, n)
        # eval(n, n.valid_data_loader, *n.calibration_variables[1])
    eloc_train(class_model=n, ood_data=ood_dataloaders, sood_data=sood_dataloaders, rood_data=rood_dataloaders,
               n_sood_iter=opt.n_sood_iter, beta=opt.beta, m_coeff=opt.m, num_epochs=opt.epochs,
               starting_e=opt.load_epoch)
    n.save(opt.epochs)
    # eval(n, n.test_data_loader, *n.calibration_variables[2])
