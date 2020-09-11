from __future__ import print_function
import argparse

from data import get_dataloader
from model import get_criterion
from classification_net import ClassifyNet

import torch
from torch.autograd import Variable

import numpy as np
import time


def tpr95(name):
    # calculate the falsepositive error when tpr is 95%
    # calculate baseline
    T = 1
    cifar = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_Base_In.txt',
                       delimiter=',')
    other = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_Base_Out.txt',
                       delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 1
    if name == "CIFAR-100":
        start = 0.01
        end = 1
    if name == 'isic':
        start = 0.1
        end = 1
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        # print('tpr:' + str(tpr) + ' | error: ' + str(error2))
        if tpr <= 0.96 and tpr >= 0.94:
            fpr += error2
            total += 1
    fprBase = fpr / total

    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_ODIN_In.txt',
                       delimiter=',')
    other = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_ODIN_Out.txt',
                       delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    if name == 'isic':
        start = 0.1
        end = 1
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    total = 0.0000000000001
    fpr = 0.000000001
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        # print('tpr:' + str(tpr) + ' | error: ' + str(error2))
        if tpr <= 0.96 and tpr >= 0.94:
            fpr += error2
            total += 1
    fprNew = fpr / total

    return fprBase, fprNew


def auroc(name):
    # calculate the AUROC
    # calculate baseline
    T = 1
    cifar = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_Base_In.txt',
                       delimiter=',')
    other = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_Base_Out.txt',
                       delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 1
    if name == "CIFAR-100":
        start = 0.01
        end = 1
    if name == "isic":
        start = 0.1
        end = 1
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    aurocBase = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        aurocBase += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocBase += fpr * tpr
    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_ODIN_In.txt',
                       delimiter=',')
    other = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_ODIN_Out.txt',
                       delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    if name == 'isic':
        start = 0.1
        end = 1
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    aurocNew = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        aurocNew += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocNew += fpr * tpr
    return aurocBase, aurocNew


def auprIn(name):
    # calculate the AUPR
    # calculate baseline
    T = 1
    cifar = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_Base_In.txt',
                       delimiter=',')
    other = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_Base_Out.txt',
                       delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 1
    if name == "CIFAR-100":
        start = 0.01
        end = 1
    if name == "isic":
        start = 0.1
        end = 1
    gap = (end - start) / 100000
    precisionVec = []
    recallVec = []
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision
    # print(recall, precision)

    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_ODIN_In.txt',
                       delimiter=',')
    other = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_ODIN_Out.txt',
                       delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    if name == 'isic':
        start = 0.1
        end = 1
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    auprNew = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        # precisionVec.append(precision)
        # recallVec.append(recall)
        auprNew += (recallTemp - recall) * precision
        recallTemp = recall
    auprNew += recall * precision
    return auprBase, auprNew


def auprOut(name):
    # calculate the AUPR
    # calculate baseline
    T = 1
    cifar = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_Base_In.txt',
                       delimiter=',')
    other = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_Base_Out.txt',
                       delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 1
    if name == "CIFAR-100":
        start = 0.01
        end = 1
    if name == "isic":
        start = 0.1
        end = 1
    gap = (end - start) / 100000
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision

    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_ODIN_In.txt',
                       delimiter=',')
    other = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_ODIN_Out.txt',
                       delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    if name == 'isic':
        start = 0.1
        end = 1
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    auprNew = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprNew += (recallTemp - recall) * precision
        recallTemp = recall
    auprNew += recall * precision
    return auprBase, auprNew


def detection(name):
    # calculate the minimum detection error
    # calculate baseline
    T = 1
    cifar = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_Base_In.txt',
                       delimiter=',')
    other = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_Base_Out.txt',
                       delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 1
    if name == "CIFAR-100":
        start = 0.01
        end = 1
    if name == "isic":
        start = 0.1
        end = 1
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    errorBase = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr + error2) / 2.0)

    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_ODIN_In.txt',
                       delimiter=',')
    other = np.loadtxt('/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_ODIN_Out.txt',
                       delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    if name == 'isic':
        start = 0.1
        end = 1
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    errorNew = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorNew = np.minimum(errorNew, (tpr + error2) / 2.0)

    return errorBase, errorNew


def compute_odin(net1, criterion, CUDA_DEVICE, testloader10, testloader, nnName, dataName, noiseMagnitude1, temper):
    t0 = time.time()
    f1 = open("/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_ODIN_In.txt", 'w')
    g2 = open("/nas/softechict-nas-1/fpollastri/OOD_MODELS/softmax_scores/confidence_ODIN_Out.txt", 'w')
    N = 10000
    # if dataName == "iSUN": N = 8925
    print("Processing in-distribution images")
    ########################################In-distribution###########################################
    for j, data in enumerate(testloader10):
        if j % 10: continue
        # print(j)
        images, target, _ = data

        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad=True)
        outputs = net1(inputs)

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        # Using temperature scaling
        outputs = outputs / temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        # mean: tensor([0.6681, 0.5301, 0.5247]) | std: tensor([0.1337, 0.1480, 0.1595])
        gradient[0, 0, :, :] = (gradient[0, 0, :, :]) / 0.1337
        gradient[0, 1, :, :] = (gradient[0, 1, :, :]) / 0.1480
        gradient[0, 2, :, :] = (gradient[0, 2, :, :]) / 0.1595
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
        outputs = net1(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        if (j + 9) % 500 == 499:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j / 10 + 1, len(testloader10.dataset) / 10,
                                                                            time.time() - t0))
            t0 = time.time()

        # if j == N - 1: break

    t0 = time.time()
    print("Processing out-of-distribution images")
    ###################################Out-of-Distributions#####################################
    for j, data in enumerate(testloader):
        # if j < 1000: continue
        images, target, _ = data

        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad=True)
        outputs = net1(inputs)

        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        # Using temperature scaling
        outputs = outputs / temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0, 0, :, :] = (gradient[0, 0, :, :]) / 0.1337
        gradient[0, 1, :, :] = (gradient[0, 1, :, :]) / 0.1480
        gradient[0, 2, :, :] = (gradient[0, 2, :, :]) / 0.1595
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
        outputs = net1(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        if j % 50 == 49:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j + 1, len(testloader.dataset),
                                                                            time.time() - t0))
            t0 = time.time()

        # if j == N - 1: break


def metric(nn, data):
    if nn == "densenet10" or nn == "wideresnet10": indis = "CIFAR-10"
    if nn == "densenet100" or nn == "wideresnet100": indis = "CIFAR-100"
    if nn == "densenet10" or nn == "densenet100": nnStructure = "DenseNet-BC-100"
    if nn == "wideresnet10" or nn == "wideresnet100": nnStructure = "Wide-ResNet-28-10"

    if data == "Imagenet": dataName = "Tiny-ImageNet (crop)"
    if data == "Imagenet_resize": dataName = "Tiny-ImageNet (resize)"
    if data == "LSUN": dataName = "LSUN (crop)"
    if data == "LSUN_resize": dataName = "LSUN (resize)"
    if data == "iSUN": dataName = "iSUN"
    if data == "Gaussian": dataName = "Gaussian noise"
    if data == "Uniform": dataName = "Uniform Noise"
    if data == 'isic':
        nnStructure = 'densenet'
        indis = 'isic'
        dataName = 'isic'
    fprBase, fprNew = tpr95(indis)
    errorBase, errorNew = detection(indis)
    aurocBase, aurocNew = auroc(indis)
    auprinBase, auprinNew = auprIn(indis)
    auproutBase, auproutNew = auprOut(indis)
    print("{:31}{:>22}".format("Neural network architecture:", nnStructure))
    print("{:31}{:>22}".format("In-distribution dataset:", indis))
    print("{:31}{:>22}".format("Out-of-distribution dataset:", dataName))
    print("")
    print("{:>34}{:>19}".format("Baseline", "Our Method"))
    print("{:20}{:13.1f}%{:>18.1f}% ".format("FPR at TPR 95%:", fprBase * 100, fprNew * 100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("Detection error:", errorBase * 100, errorNew * 100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUROC:", aurocBase * 100, aurocNew * 100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUPR In:", auprinBase * 100, auprinNew * 100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUPR Out:", auproutBase * 100, auproutNew * 100))


if __name__ == '__main__':
    net_parser = argparse.ArgumentParser()
    net_parser.add_argument('--eps', type=float, default=0.002, help='epsilon value')
    net_parser.add_argument('--t', '-t', type=float, default=1000, help='temperature scaling value')

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

    net_opt = net_parser.parse_args()
    print(net_opt)

    _, in_dataset, _ = get_dataloader(dname='isic2019', dataset_classes=[[0], [1], [2], [3], [4], [5], [6]], SRV=True,
                                      batch_size=net_opt.batch_size)
    _, out_dataset, _ = get_dataloader(dname='isic2019', dataset_classes=[[7]], SRV=True, batch_size=net_opt.batch_size)
    criterion = get_criterion(lossname='cross_entropy', dataset_classes=[[0], [1], [2], [3], [4], [5], [6]])

    n = ClassifyNet(net=net_opt.network, dname=net_opt.dataset, dropout=net_opt.dropout,
                    classes=net_opt.classes, l_r=net_opt.learning_rate, loss=net_opt.loss,
                    optimizer=net_opt.optimizer, scheduler=net_opt.scheduler, size=net_opt.size,
                    batch_size=net_opt.batch_size, n_workers=net_opt.workers, pretrained=(not net_opt.from_scratch),
                    augm_config=net_opt.augm_config, save_dir=net_opt.save_dir, mixup_coeff=net_opt.mixup,
                    cutout_params=[net_opt.cutout_holes, net_opt.cutout_pad], total_epochs=net_opt.epochs,
                    SRV=net_opt.SRV, no_logs=True, optimize_temp_scal=False)

    if not net_opt.load_epoch == 0:
        n.load(net_opt.load_epoch)

    compute_odin(net1=n.n, criterion=criterion, CUDA_DEVICE='cuda', testloader10=in_dataset, testloader=out_dataset,
                 nnName='name', dataName='name', noiseMagnitude1=net_opt.eps, temper=net_opt.t)

    metric('densenet', 'isic')
