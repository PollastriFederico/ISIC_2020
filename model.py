from torchvision import models
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad
from pytorchcv.model_provider import get_model as ptcv_get_model
from efficientnet_pytorch import EfficientNet


class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.T = nn.Parameter(torch.ones(1, ))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        return x * self.T

    def loss(self, x, t):
        return self.ce(x, t)


class MyResnet(nn.Module):
    def __init__(self, net='resnet101', pretrained=False, num_classes=1, dropout_flag=False, size=512):
        super(MyResnet, self).__init__()
        self.dropout_flag = dropout_flag
        if net == 'resnet18':
            resnet = models.resnet18(pretrained)
            bl_exp = 1
        elif net == 'resnet34':
            resnet = models.resnet34(pretrained)
            bl_exp = 1
        elif net == 'resnet50':
            resnet = models.resnet50(pretrained)
            bl_exp = 4
        elif net == 'resnet101':
            resnet = models.resnet101(pretrained)
            bl_exp = 4
        elif net == 'resnet152':
            resnet = models.resnet152(pretrained)
            bl_exp = 4
        else:
            raise Warning("Wrong Net Name!!")
        self.resnet = nn.Sequential(*(list(resnet.children())[:-2]))
        self.avgpool = nn.AvgPool2d(int(size / 32), stride=1)
        if self.dropout_flag:
            self.dropout = nn.Dropout(0.5)
        self.last_fc = nn.Linear(512 * bl_exp, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout_flag:
            x = self.dropout(x)
        x = self.last_fc(x)
        return x


class MyEfficientnet(nn.Module):
    def __init__(self, net='efficientnetb0', pretrained=False, num_classes=1, dropout_flag=False, size=512):
        super(MyEfficientnet, self).__init__()
        self.dropout_flag = dropout_flag
        if net[:13] == 'efficientnetb' and len(net) == 14 and int(net[-1]) <= 7:
            if pretrained:
                efficientnet = EfficientNet.from_pretrained('efficientnet-b' + net[-1], num_classes=num_classes)
            else:
                efficientnet = EfficientNet.from_name('efficientnet-b' + net[-1], num_classes=num_classes)

        else:
            raise Warning("Wrong Net Name!!")

        self.efficientnet = efficientnet
        # self.resnet = nn.Sequential(*(list(efficientnet.children())[:-1]))
        # # self.avgpool = nn.AvgPool2d(int(size / 32), stride=1)
        # # if self.dropout_flag:
        # #     self.dropout = nn.Dropout(0.5)
        # self.last_fc = nn.Linear(512 * bl_exp, num_classes)

    def forward(self, x):
        # x = self.resnet(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # if self.dropout_flag:
        #     x = self.dropout(x)
        # x = self.last_fc(x)
        x = self.efficientnet(x)
        return x


class MySeResnext(nn.Module):
    def __init__(self, net='seresnext50', pretrained=False, num_classes=1, dropout_flag=False, size=512):
        super(MySeResnext, self).__init__()
        self.dropout_flag = dropout_flag
        self.size = size

        if net == 'seresnext50':
            seresnext50 = ptcv_get_model("seresnext50_32x4d", pretrained=pretrained)
            if self.size == 224:
                self.model = nn.Sequential(*(list(seresnext50.children())[0]))
            else:
                self.model = nn.Sequential(*(list(seresnext50.children())[0][:-1]))
        elif net == 'seresnext101':
            seresnext101 = ptcv_get_model("seresnext101_32x4d", pretrained=pretrained)
            if self.size == 224:
                self.model = nn.Sequential(*(list(seresnext101.children())[0]))
            else:
                self.model = nn.Sequential(*(list(seresnext101.children())[0][:-1]))
        else:
            raise Warning("Wrong Net Name!!")

        if self.size != 224:
            self.avgpool = nn.AvgPool2d(int(size / 32), stride=1)

        if self.dropout_flag:
            self.dropout = nn.Dropout(0.5)

        self.last_fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)

        if self.size != 224:
            x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.dropout_flag:
            x = self.dropout(x)

        x = self.last_fc(x)
        return x


class MyPolynet(nn.Module):
    def __init__(self, net='seresnext50', pretrained=False, num_classes=1, dropout_flag=False, size=512):
        super(MyPolynet, self).__init__()
        self.dropout_flag = dropout_flag
        self.size = size

        if net == 'polynet':
            polynet = ptcv_get_model("polynet", pretrained=pretrained)
            if self.size == 224:
                self.model = nn.Sequential(*(list(polynet.children())[0]))
            else:
                self.model = nn.Sequential(*(list(polynet.children())[0][:-1]))
        else:
            raise Warning("Wrong Net Name!!")

        if self.size == 512:
            self.avgpool = nn.AvgPool2d(int(14), stride=1)

        if self.dropout_flag:
            self.dropout = nn.Dropout(0.5)

        self.last_fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)

        if self.size != 224:
            x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.dropout_flag:
            x = self.dropout(x)

        x = self.last_fc(x)
        return x


class MyDensenet(nn.Module):
    def __init__(self, net='densenet169', pretrained=False, num_classes=1, dropout_flag=False, size=512):
        super(MyDensenet, self).__init__()
        self.dropout_flag = dropout_flag
        if net == 'densenet121':
            densenet = models.densenet121(pretrained)
            num_features = 1024
        elif net == 'densenet169':
            densenet = models.densenet169(pretrained)
            num_features = 1664
        elif net == 'densenet201':
            densenet = models.densenet201(pretrained)
            num_features = 1920
        elif net == 'densenet161':
            densenet = models.densenet161(pretrained)
            num_features = 2208
        else:
            raise Warning("Wrong Net Name!!")
        self.densenet = nn.Sequential(*(list(densenet.children())[0]))
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=int(size / 32), stride=1)
        if self.dropout_flag:
            self.dropout = nn.Dropout(0.5)
        self.last_fc = nn.Linear(num_features, num_classes)

    def forward(self, x, w_code=False):
        x = self.densenet(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout_flag:
            x = self.dropout(x)
        out = self.last_fc(x)
        if w_code:
            return x, out
        return out


class MyDensenetwSegm(nn.Module):
    def __init__(self, net='densenet169', pretrained=False, num_classes=1, dropout_flag=False, size=512):
        super(MyDensenetwSegm, self).__init__()
        self.dropout_flag = dropout_flag
        if net == 'densenet121':
            densenet = models.densenet121(pretrained)
            num_init_features = 64
            num_features = 1024
        elif net == 'densenet169':
            densenet = models.densenet169(pretrained)
            num_init_features = 64
            num_features = 1664
        elif net == 'densenet201':
            densenet = models.densenet201(pretrained)
            num_init_features = 64
            num_features = 1920
        elif net == 'densenet161':
            densenet = models.densenet161(pretrained)
            num_init_features = 96
            num_features = 2208
        else:
            raise Warning("Wrong Net Name!!")
        self.first_layer = nn.Conv2d(4, num_init_features, kernel_size=7, stride=2,
                                     padding=3, bias=False)
        self.densenet = nn.Sequential(*(list(densenet.children())[0][1:]))
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=int(size / 32), stride=1)
        if self.dropout_flag:
            self.dropout = nn.Dropout(0.5)
        self.last_fc = nn.Linear(num_features, num_classes)

    def forward(self, x, w_code=False):
        x = self.first_layer(x)
        x = self.densenet(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout_flag:
            x = self.dropout(x)
        out = self.last_fc(x)
        if w_code:
            return x, out
        return out


class MLP_G(nn.Module):
    def __init__(self, n_classes, n_features, nz, ngf, ngpu=1):
        super(MLP_G, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ngf
            nn.Linear(nz + n_classes, ngf),
            # nn.Linear(nz, ngf),
            nn.LeakyReLU(0.2, True),
            # nn.BatchNorm1d(ngf),
            nn.Linear(ngf, ngf * 2),
            nn.LeakyReLU(0.2, True),
            # nn.BatchNorm1d(ngf * 2),
            nn.Linear(ngf * 2, ngf * 4),
            nn.LeakyReLU(0.2, True),
            # nn.BatchNorm1d(ngf * 4),
            nn.Linear(ngf * 4, n_features),
        )
        self.main = main
        self.n_classes = n_classes
        self.n_features = n_features
        self.nz = nz

    def forward(self, input, c):
        input = input.view(input.size(0), input.size(1))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            # output = self.main(input)
            output = self.main(torch.cat((input, c), dim=-1))
        # return output.view(output.size(0), self.nc, self.isize, self.isize)
        return output


class MLP_D(nn.Module):
    def __init__(self, n_classes, n_features, nz, ndf, ngpu=1):
        super(MLP_D, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(n_features + n_classes, ndf * 4),
            # nn.Linear(n_features, ndf * 4),
            nn.LeakyReLU(0.2, True),
            # nn.BatchNorm1d(ndf * 4),
            nn.Linear(ndf * 4, ndf * 2),
            nn.LeakyReLU(0.2, True),
            # nn.BatchNorm1d(ndf * 2),
            nn.Linear(ndf * 2, ndf),
            nn.LeakyReLU(0.2, True),
            # nn.BatchNorm1d(ndf),
            nn.Linear(ndf, 1),
        )
        self.main = main
        self.n_features = n_features
        self.n_classes = n_classes
        self.nz = nz

    def forward(self, input, c):
        # input = input.view(input.size(0),
        #                    input.size(1) * input.size(2) * input.size(3))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(torch.cat((input, c), dim=-1))
            # output = self.main(input)
        # output = output.mean(0)
        # return output.view(1)
        return output


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


class GradientPenalty:
    """Computes the gradient penalty as defined in "Improved Training of Wasserstein GANs"
    (https://arxiv.org/abs/1704.00028)
    Args:
        batchSize (int): batch-size used in the training. Must be updated w.r.t the current batchsize
        lambdaGP (float): coefficient of the gradient penalty as defined in the article
        gamma (float): regularization term of the gradient penalty, augment to minimize "ghosts"
    """

    def __init__(self, batchSize, lambdaGP, gamma=1, device='cuda'):
        self.batchSize = batchSize
        self.lambdaGP = lambdaGP
        self.gamma = gamma
        self.device = device

    def __call__(self, netD, real_data, fake_data, real_labels, fake_labels):
        alpha = torch.rand(self.batchSize, 1, 1, 1, requires_grad=True, device=self.device)
        # randomly mix real and fake data
        interpolates = real_data + alpha * (fake_data - real_data)
        interpolates_lab = real_labels + alpha * (fake_labels - real_labels)
        # compute output of D for interpolated input
        disc_interpolates = netD(interpolates, interpolates_lab)
        # compute gradients w.r.t the interpolated outputs
        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0].contiguous().view(self.batchSize,
                                                                                                      -1)
        gradient_penalty = (((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2).mean() * self.lambdaGP

        return gradient_penalty


def get_model(net, pretrained, num_classes, dropout, size):
    if net[:8] == 'densenet':
        return MyDensenet(net=net, pretrained=pretrained, num_classes=num_classes, dropout_flag=dropout, size=size).to(
            'cuda')
    elif net[:6] == 'resnet':
        return MyResnet(net=net, pretrained=pretrained, num_classes=num_classes, dropout_flag=dropout, size=size).to(
            'cuda')
    elif "seresnext" in net:
        return MySeResnext(net=net, pretrained=pretrained, num_classes=num_classes, dropout_flag=dropout, size=size).to(
            'cuda')
    elif "polynet" in net:
        return MyPolynet(net=net, pretrained=pretrained, num_classes=num_classes, dropout_flag=dropout, size=size).to(
            'cuda')
    elif "efficientnet" in net:
        return MyEfficientnet(net=net, pretrained=pretrained, num_classes=num_classes, dropout_flag=dropout,
                              size=size).to(
            'cuda')
    else:
        raise Warning("Wrong Net Name!!")


def get_model_wsegm(net, pretrained, num_classes, dropout, size):
    if net[:8] == 'densenet':
        return MyDensenetwSegm(net=net, pretrained=pretrained, num_classes=num_classes, dropout_flag=dropout,
                               size=size).to(
            'cuda')
    elif net[:6] == 'resnet':
        return MyResnet(net=net, pretrained=pretrained, num_classes=num_classes, dropout_flag=dropout, size=size).to(
            'cuda')
    elif "seresnext" in net:
        return MySeResnext(net=net, pretrained=pretrained, num_classes=num_classes, dropout_flag=dropout, size=size).to(
            'cuda')
    elif "polynet" in net:
        return MyPolynet(net=net, pretrained=pretrained, num_classes=num_classes, dropout_flag=dropout, size=size).to(
            'cuda')
    else:
        raise Warning("Wrong Net Name!!")


def get_criterion(lossname, dataset_classes=[[0], [1]]):
    # Defining the loss
    whole_training_stats = [1 - 0.0176291, 0.0176291]
    training_stats = []
    for c in dataset_classes:
        training_stats.append(np.sum([whole_training_stats[c_i] for c_i in c]))
    weights = np.divide(1, training_stats, dtype='float32')

    criterion = None
    if lossname == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device='cuda'))
    elif lossname == 'focal':
        criterion = FocalLoss(weight=torch.tensor(weights, device='cuda'))

    return criterion


def get_optimizer(n, learning_rate, optname='Adam'):
    optimizer = None
    if optname == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, n.parameters()), lr=learning_rate)
    elif optname == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, n.parameters()), lr=learning_rate)
    else:
        print("WRONG OPTIMIZER NAME")

    return optimizer


def get_scheduler(optimizer, schedname=None):
    scheduler = None
    if schedname == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, threshold=0.004)

    return scheduler
