import argparse
import torch
from torch.backends import cudnn
import time
import os
import numpy as np
from classification_net import ClassifyNet, eval, ensemble_aug_eval
from model import MLP_G, MLP_D, GradientPenalty
from data import get_dataloader

from utils import categorical_to_one_hot


# from torch.utils.data import DataLoader
# from torchvision import transforms
#
# from birds import birds_caltech_2011


def hypersphere(z, radius=1):
    return z * radius / z.norm(p=2, dim=1, keepdim=True)


class GAOODN:
    def __init__(self, net, dname, l_r, save_dir, total_epochs, SRV, OOD, n_classes, n_features, nz, ngf, ndf, ngpu):
        # Hyper-parameters
        self.n_classes = n_classes
        self.n_features = n_features
        self.net = net
        self.dname = dname
        self.l_r = l_r
        self.save_dir = save_dir
        self.total_epochs = total_epochs
        self.SRV = SRV
        self.OOD = OOD
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf

        self.netG = MLP_G(self.n_classes, self.n_features, self.nz, self.ngf, self.ngpu)
        self.netD = MLP_D(self.n_classes, self.n_features, self.nz, self.ndf, self.ngpu)
        self.netD.cuda()
        self.netG.cuda()

        # setup optimizer
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.l_r, betas=(0.5, 0.9))
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.l_r, betas=(0.5, 0.9))

        self.ood_training_dataloader, self.ood_test_dataloader, self.ood_valid_dataloader = get_dataloader(
            dname='isic2019', size=opt.size, dataset_classes=opt.ood_classes, SRV=True,
            batch_size=opt.batch_size, n_workers=opt.workers, augm_config=opt.augm_config,
            cutout_params=[opt.cutout_holes, opt.cutout_pad], drop_last_flag=True)

        self.true_ood_training_dataloader = None
        self.true_ood_test_dataloader = None
        self.true_ood_valid_dataloader = None

        if opt.true_ood_classes:
            self.true_ood_training_dataloader, self.true_ood_test_dataloader, self.true_ood_valid_dataloader = get_dataloader(
                dname='isic2019', size=opt.size, dataset_classes=opt.true_ood_classes, SRV=True,
                batch_size=opt.batch_size, n_workers=opt.workers, augm_config=opt.augm_config,
                cutout_params=[opt.cutout_holes, opt.cutout_pad], drop_last_flag=True)

    def save(self, n_epoch=0):
        self.save_mode_one(n_epoch)

    def save_mode_zero(self, n_epoch=0):
        try:
            torch.save(self.n.state_dict(),
                       os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                           self.augm_config) + '_mixupcoeff.' + str(self.mixup_coeff) + '_cutout.holes' + str(
                           self.cutout_nholes) + '.pad.' + str(self.cutout_pad_size) + '_net.pth'))
            torch.save(self.optimizer.state_dict(),
                       os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                           self.augm_config) + '_mixupcoeff.' + str(self.mixup_coeff) + '_cutout.holes' + str(
                           self.cutout_nholes) + '.pad.' + str(self.cutout_pad_size) + '_opt.pth'))
            print("mode zero model weights successfully saved")
        except Exception:
            print("Error during Saving")

    def save_mode_one(self, n_epoch=0):
        try:
            torch.save(self.netG.state_dict(),
                       os.path.join(self.save_dir, 'GAOODD' + '_epoch.' + str(n_epoch) + '.classes.' + str(
                           opt.classes) + '_Generator.pth'))
            torch.save(self.optimizerG.state_dict(),
                       os.path.join(self.save_dir, 'GAOODD' + '_epoch.' + str(n_epoch) + '.classes.' + str(
                           opt.classes) + '_G_opt.pth'))
            torch.save(self.netD.state_dict(),
                       os.path.join(self.save_dir, 'GAOODD' + '_epoch.' + str(n_epoch) + '.classes.' + str(
                           opt.classes) + '_Discriminator.pth'))
            torch.save(self.optimizerD.state_dict(),
                       os.path.join(self.save_dir, 'GAOODD' + '_epoch.' + str(n_epoch) + '.classes.' + str(
                           opt.classes) + '_D_opt.pth'))
            print("mode one model weights successfully saved")
        except Exception:
            print("Error during Saving")

    def save_with_temp_scal(self, n_epoch=0):
        try:
            saved_dict = {'model_parameters': self.n.state_dict(),
                          'temperature_scal': self.temp_scal_model.state_dict()}
            torch.save(saved_dict,
                       os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                           self.augm_config) + '_mixupcoeff.' + str(self.mixup_coeff) + '_cutout.holes' + str(
                           self.cutout_nholes) + '.pad.' + str(
                           self.cutout_pad_size) + '_temperature.scaling.decoupled_net.pth'))
            torch.save(self.optimizer.state_dict(),
                       os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                           self.augm_config) + '_mixupcoeff.' + str(self.mixup_coeff) + '_cutout.holes' + str(
                           self.cutout_nholes) + '.pad.' + str(
                           self.cutout_pad_size) + '_temperature.scaling.decoupled_opt.pth'))
            print("model weights and temp scal T successfully saved")
        except Exception:
            print("Error during Saving")

    def load(self, n_epoch=0):
        try:
            self.load_mode_one(n_epoch)
        except:
            raise FileExistsError

    def load_mode_zero(self, n_epoch=0):
        self.n.load_state_dict(
            torch.load(os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                self.augm_config) + '_mixupcoeff.' + str(self.mixup_coeff) + '_cutout.holes' + str(
                self.cutout_nholes) + '.pad.' + str(self.cutout_pad_size) + '_net.pth')))
        self.optimizer.load_state_dict(
            torch.load(os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                self.augm_config) + '_mixupcoeff.' + str(self.mixup_coeff) + '_cutout.holes' + str(
                self.cutout_nholes) + '.pad.' + str(self.cutout_pad_size) + '_opt.pth')))
        print("mode zero model weights successfully loaded")

    def load_mode_one(self, n_epoch=0):
        self.netG.load_state_dict(
            torch.load(os.path.join(self.save_dir, 'GAOODD' + '_epoch.' + str(n_epoch) + '.classes.' + str(
                opt.classes) + '_Generator.pth')))
        self.optimizerG.load_state_dict(
            torch.load(os.path.join(self.save_dir, 'GAOODD' + '_epoch.' + str(n_epoch) + '.classes.' + str(
                opt.classes) + '_G_opt.pth')))
        self.netD.load_state_dict(
            torch.load(os.path.join(self.save_dir, 'GAOODD' + '_epoch.' + str(n_epoch) + '.classes.' + str(
                opt.classes) + '_Discriminator.pth')))
        self.optimizerD.load_state_dict(
            torch.load(os.path.join(self.save_dir, 'GAOODD' + '_epoch.' + str(n_epoch) + '.classes.' + str(
                opt.classes) + '_D_opt.pth')))
        print("mode one model weights successfully loaded")

    def load_with_temp_scal(self, n_epoch=0):
        saved_dict = torch.load(
            os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                self.augm_config) + '_mixupcoeff.' + str(self.mixup_coeff) + '_cutout.holes' + str(
                self.cutout_nholes) + '.pad.' + str(self.cutout_pad_size) + '_temperature.scaling.decoupled_net.pth'))

        self.n.load_state_dict(saved_dict['model_parameters'])
        self.temp_scal_model.load_state_dict(saved_dict['temperature_scal'])
        self.optimizer.load_state_dict(
            torch.load(os.path.join(self.save_dir, self.nname + '_epoch.' + str(n_epoch) + '_augmentidx' + str(
                self.augm_config) + '_mixupcoeff.' + str(self.mixup_coeff) + '_cutout.holes' + str(
                self.cutout_nholes) + '.pad.' + str(self.cutout_pad_size) + '_temperature.scaling.decoupled_opt.pth')))
        print("model weights and temp scal T successfully loaded")


def train(class_model, ood_model, num_epochs, lambdaGP=10, gamma=1, e_drift=0.001, starting_e=0):
    if ood_model.ngpu > 0:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    sofmx = torch.nn.Softmax(dim=-1)
    GP = GradientPenalty(class_model.batch_size, lambdaGP, gamma, DEVICE)
    best_fpr = 1.0
    for epoch in range(starting_e + 1, num_epochs):
        class_model.n.train()
        start_time = time.time()
        for idx, (x, target, _) in enumerate(class_model.data_loader):
            # measure data loading time
            # print("data time: " + str(time.time() - start_time))

            # GET HIDDEN REPRESENTATION
            with torch.no_grad():
                # compute output
                x = x.to('cuda')
                target = target.to('cuda', torch.long)
                output, c_pred = class_model.n(x, w_code=True)
                check_output_all = sofmx(c_pred)
                check_output, res = torch.max(check_output_all, -1)
                net_c_pred = categorical_to_one_hot(res.to('cpu', dtype=torch.long), class_model.num_classes).to('cuda')

            lossEpochG = []
            lossEpochD = []
            lossEpochD_W = []

            ood_model.netG.train()
            cudnn.benchmark = True

            # ============= Train the discriminator =============#

            # zeroing gradients in D
            ood_model.netD.zero_grad()
            # compute fake images with G
            z = hypersphere(torch.randn(class_model.batch_size, ood_model.nz, 1, 1, device=DEVICE))
            with torch.no_grad():
                fake_labels = np.random.randint(0, class_model.num_classes, class_model.batch_size)
                fake_labels = torch.tensor(fake_labels, dtype=torch.long)
                fake_labels = categorical_to_one_hot(fake_labels, class_model.num_classes).to('cuda')
                fake_images = ood_model.netG(z, fake_labels)

            # compute scores for real images
            D_real = ood_model.netD(output, net_c_pred)
            D_realm = D_real.mean()

            # compute scores for fake images
            D_fake = ood_model.netD(fake_images, fake_labels)
            D_fakem = D_fake.mean()

            # compute gradient penalty for WGAN-GP as defined in the article
            gradient_penalty = GP(ood_model.netD, output.data, fake_images.data,
                                  net_c_pred.to(dtype=torch.float), fake_labels.to(dtype=torch.float))

            # prevent D_real from drifting too much from 0
            drift = (D_real ** 2).mean() * e_drift

            # Backprop + Optimize
            d_loss = D_fakem - D_realm
            d_loss_W = d_loss + gradient_penalty + drift
            d_loss_W.backward()
            ood_model.optimizerD.step()

            lossEpochD.append(d_loss.item())
            lossEpochD_W.append(d_loss_W.item())

            # =============== Train the generator ===============#

            ood_model.netG.zero_grad()

            z = hypersphere(torch.randn(class_model.batch_size, ood_model.nz, 1, 1, device=DEVICE))
            fake_labels = np.random.randint(0, class_model.num_classes, class_model.batch_size)
            fake_labels = torch.tensor(fake_labels, dtype=torch.long)
            fake_labels = categorical_to_one_hot(fake_labels, class_model.num_classes).to('cuda')
            fake_images = ood_model.netG(z, fake_labels)
            # compute scores with new fake images
            G_fake = ood_model.netD(fake_images, fake_labels)
            G_fakem = G_fake.mean()
            # no need to compute D_real as it does not affect G
            g_loss = -G_fakem

            # Optimize
            g_loss.backward()
            ood_model.optimizerG.step()

            lossEpochG.append(g_loss.item())

            # break

        print('epoch: ' + str(epoch) + ' | G learning rate: ' + str(
            ood_model.optimizerG.param_groups[0].get('lr')) + ' | D learning rate: ' + str(
            ood_model.optimizerD.param_groups[0].get('lr')) + ' | loss D: ' + str(
            np.mean(lossEpochD)) + ' | W loss D: ' + str(np.mean(
            lossEpochD_W)) + ' | loss G: ' + str(np.mean(lossEpochG)) + ' | time: ' + str(time.time() - start_time))
        print('VALIDATION:\n')
        curr_fpr = gaood_eval(class_model, ood_model, class_model.valid_data_loader, ood_model.ood_valid_dataloader,
                              ood_model.true_ood_valid_dataloader, best_fpr=best_fpr)
        # if epoch % 5 == 0:
        print('TEST:\n')
        gaood_eval(class_model, ood_model, class_model.test_data_loader, ood_model.ood_test_dataloader,
                   ood_model.true_ood_test_dataloader, best_fpr=best_fpr)

        if curr_fpr < best_fpr:
            best_fpr = curr_fpr
            ood_model.save(epoch)
            # ood_model.load(epoch)


def gaood_eval(class_model, ood_model, iid_loader, ood_loader, true_ood_loader=None, best_fpr=1.0):
    with torch.no_grad():
        D_real = torch.FloatTensor().to('cuda')
        D_fake = torch.FloatTensor().to('cuda')
        D_true_fake = torch.FloatTensor().to('cuda')
        sofmx = torch.nn.Softmax(dim=-1)
        for idx, (x, target, _) in enumerate(iid_loader):
            # measure data loading time
            # print("data time: " + str(time.time() - start_time))

            # GET HIDDEN REPRESENTATION
            x = x.to('cuda')
            target = target.to('cuda', torch.long)
            output, c_pred = class_model.n(x, w_code=True)
            check_output_all = sofmx(c_pred)
            check_output, res = torch.max(check_output_all, -1)
            net_c_pred = categorical_to_one_hot(res.to('cpu', dtype=torch.long), class_model.num_classes).to('cuda')
            d_out = ood_model.netD(output, net_c_pred)
            D_real = torch.cat((D_real, d_out))
            # break
        for idx, (x, target, _) in enumerate(ood_loader):
            # measure data loading time
            # print("data time: " + str(time.time() - start_time))

            # GET HIDDEN REPRESENTATION
            x = x.to('cuda')
            target = target.to('cuda', torch.long)
            output, c_pred = class_model.n(x, w_code=True)
            check_output_all = sofmx(c_pred)
            check_output, res = torch.max(check_output_all, -1)
            net_c_pred = categorical_to_one_hot(res.to('cpu', dtype=torch.long), class_model.num_classes).to('cuda')
            d_out = ood_model.netD(output, net_c_pred)
            D_fake = torch.cat((D_fake, d_out))
            # break
        if true_ood_loader is not None:
            for idx, (x, target, _) in enumerate(true_ood_loader):
                # measure data loading time
                # print("data time: " + str(time.time() - start_time))

                # GET HIDDEN REPRESENTATION
                x = x.to('cuda')
                target = target.to('cuda', torch.long)
                output, c_pred = class_model.n(x, w_code=True)
                check_output_all = sofmx(c_pred)
                check_output, res = torch.max(check_output_all, -1)
                net_c_pred = categorical_to_one_hot(res.to('cpu', dtype=torch.long), class_model.num_classes).to('cuda')
                d_out = ood_model.netD(output, net_c_pred)
                D_true_fake = torch.cat((D_true_fake, d_out))

        TOOD_FPR = 0
        min_v = min(D_real.min(), D_fake.min()).item()
        max_v = max(D_real.max(), D_fake.max()).item()
        min_tpr = 0.95
        for t in np.arange(min_v, max_v, (max_v - min_v) / 10000.):
            # for i in range(10):
            # t = i / 10.
            if min_tpr < 0.51:
                return best_fpr
            TPR = float((D_real > t).sum()) / float(len(iid_loader.dataset))
            FPR = float((D_fake > t).sum()) / float(len(ood_loader.dataset))
            if true_ood_loader is not None:
                TOOD_FPR = float((D_true_fake > t).sum()) / float(len(true_ood_loader.dataset))
            if TPR < min_tpr:
                print('t: ' + str(t) + ' | TPR: ' + str(TPR) + ' | FPR: ' + str(FPR) + ' | TOOD_FPR: ' + str(TOOD_FPR))
                min_tpr = max(0.50, min_tpr - 0.05)
                if 0.85 < TPR < 0.905 and FPR < best_fpr:
                    best_fpr = FPR
        return best_fpr


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
    parser.add_argument('--true_ood_classes', '-t_ood', type=int, nargs='+',
                        action='append', help='Real ood classes, to TEST the model against')
    parser.add_argument('--load_epoch_net', type=int, default=0, help='load custom-trained models')
    parser.add_argument('--load_epoch_gan', type=int, default=0, help='load custom-trained models')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
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

    parser.add_argument('--nz', type=int, default=100, help='size of latent vector')
    parser.add_argument('--ngf', type=int, default=512, help='size of G layers')
    parser.add_argument('--ndf', type=int, default=512, help='size of D layers')

    opt = parser.parse_args()
    print(opt)

    if opt.classes is None:
        classes = [[0], [1], [2], [3], [4], [5], [6], [7]]
    elif len(opt.classes) == 1:
        classes = [[c] for c in opt.classes[0]]
    else:
        classes = opt.classes
    num_classes = len(classes)

    n = ClassifyNet(net=opt.network, dname=opt.dataset, dropout=opt.dropout, classes=opt.classes,
                    l_r=opt.learning_rate, loss=opt.loss, optimizer=opt.optimizer, scheduler=opt.scheduler,
                    size=opt.size, batch_size=opt.batch_size, n_workers=opt.workers, pretrained=(not opt.from_scratch),
                    augm_config=opt.augm_config, save_dir=opt.save_dir, mixup_coeff=opt.mixup,
                    cutout_params=[opt.cutout_holes, opt.cutout_pad], total_epochs=opt.epochs, SRV=opt.SRV,
                    optimize_temp_scal=opt.calibrated, drop_last=True)

    gan = GAOODN(net=opt.network, dname=opt.dataset, l_r=opt.learning_rate, save_dir=opt.save_dir,
                 total_epochs=opt.epochs, SRV=opt.SRV, OOD=opt.ood_classes, n_classes=num_classes, n_features=1920,
                 # WARNING: HARD-CODED, TO FIX
                 nz=opt.nz, ngf=opt.ngf, ndf=opt.ndf, ngpu=1)

    if not opt.load_epoch_net == 0:
        n.load(opt.load_epoch_net)
        # acc, w_acc, calib, conf_matrix, _ = eval(n, n.test_data_loader, *n.calibration_variables[2], opt.calibrated)
        # print(acc)
        # print(w_acc)
        # print(conf_matrix)
        # ensemble_aug_eval(100, n)
        # eval(n, n.valid_data_loader, *n.calibration_variables[1])

    train(n, gan, num_epochs=opt.epochs, starting_e=opt.load_epoch_gan)

    # eval(n, n.test_data_loader, *n.calibration_variables[2])
