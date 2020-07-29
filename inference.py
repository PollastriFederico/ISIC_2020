import argparse
import torch
from torch import nn
import time
from classification_net import ClassifyNet
from data import ImgAugTransform
from torch.utils.data import DataLoader
from torchvision import transforms

from medici_medical_dataset import vidix

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


def eval_inference(class_model, e_loader, with_temp_scal=False, device='cuda'):
    with torch.no_grad():

        class_model.n.eval()
        sofmx = nn.Softmax(dim=-1)

        preds = []
        class_model.n = class_model.n.to(device)
        class_model.temp_scal_model = class_model.temp_scal_model.to(device)
        start_time = time.time()

        for idx, (x, img_name) in enumerate(e_loader):
            # measure data loading time
            # print("data time: " + str(time.time() - start_time))
            # compute output
            if device == 'cuda':
                x = x.to(device)
            out = class_model.n(x)

            if with_temp_scal:
                out = class_model.temp_scal_model(out)
            c = sofmx(out)
            check_output, res = torch.max(c, -1)
            print(img_name[0] + ': ' + class_dict.get(res.item()) + ' | Confidence: ' + str(
                check_output.item()) + ' | computed in: ' + str(time.time() - start_time) + ' seconds')

            preds.append(c)
            start_time = time.time()

        return preds


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    net_parser = argparse.ArgumentParser()
    parser.add_argument('--avg', default=None, help='name of the file with ensemble parameters')
    parser.add_argument('--da_n_iter', type=int, default=0, help='number of iterations for Data Augmentation ensemble')
    parser.add_argument('--calibrated', action='store_true', help='Boolean flag for applying temperature scaling')
    parser.add_argument('--device', default='cuda', help='device on which permorm inference', choices=['cuda', 'cpu'])

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
    net_parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
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

        inference_imgaug_transforms = ImgAugTransform(config_code=-1, size=net_opt.size, SRV=net_opt.SRV)
        test_transforms = transforms.Compose([
            inference_imgaug_transforms,
            transforms.ToTensor(),
            transforms.Normalize((0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595)),
        ])
        dataset = vidix(transform=test_transforms)

        data_loader = DataLoader(dataset,
                                 batch_size=net_opt.batch_size,
                                 shuffle=False,
                                 # num_workers=0,
                                 num_workers=net_opt.workers,
                                 drop_last=False,
                                 pin_memory=True)

        eval_inference(n, data_loader, with_temp_scal=opt.calibrated, device=opt.device)
