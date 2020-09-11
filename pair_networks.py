import argparse
import torch

if torch.__version__ != '1.1.0':
    raise (Exception('Torch version must be 1.1.0'))
import time
from classification_net import ClassifyNet, train_temperature_scaling_decoupled, eval
from data import get_dataloader
from utils import ConfusionMatrix, compute_calibration_measures

parser = argparse.ArgumentParser()
net_parser = argparse.ArgumentParser()

net_parser.add_argument('--network', default='resnet50',
                        choices=['resnet18', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169',
                                 'densenet201', 'densenet161'])
net_parser.add_argument('--save_dir', help='directory where to save model weights')
net_parser.add_argument('--dataset', default='isic2019', help='name of the dataset to use')
net_parser.add_argument('--dropout', action='store_true', help='Boolean flag for DropOut inclusion')
net_parser.add_argument('--classes', '-c', type=int, nargs='+',  # , default=[[0], [1], [2], [3], [4], [5], [6], [7]]
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
net_parser.add_argument('--from_scratch', action='store_true', help='Boolean flag for training a model from scratch')
net_parser.add_argument('--augm_config', type=int, default=1,
                        help='configuration code for augmentation techniques choice')
net_parser.add_argument('--cutout_pad', nargs='+', type=int, default=[0], help='cutout pad.')
net_parser.add_argument('--cutout_holes', nargs='+', type=int, default=[0], help='number of cutout holes.')
net_parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixout coefficient. If 0 is provided no mixup is applied')

net_parser.add_argument('--gpu_device', type=int, default=0,
                        help='Which GPU to use')


if __name__ == '__main__':

    start_time = time.time()
    models_list=[]

    for idx,line in enumerate(open('./file_pair_networks_specifications_ood7.txt','r')):
        if idx==0:
           OOD_class=int(line.split("\n")[0])
           continue
        
        l=line.split("\n")[0]
        net_opt = net_parser.parse_args(l.split(" "))

        n = ClassifyNet(net=net_opt.network, dname=net_opt.dataset, dropout=net_opt.dropout,
                    classes=net_opt.classes, l_r=net_opt.learning_rate, loss=net_opt.loss,
                    optimizer=net_opt.optimizer, scheduler=net_opt.scheduler, size=net_opt.size,
                    batch_size=1, n_workers=net_opt.workers, pretrained=(not net_opt.from_scratch), augm_config=net_opt.augm_config, save_dir=net_opt.save_dir, mixup_coeff=net_opt.mixup,
                    cutout_params=[net_opt.cutout_holes, net_opt.cutout_pad], total_epochs=net_opt.epochs,SRV=net_opt.SRV, no_logs=True, optimize_temp_scal=True)

        if not net_opt.load_epoch == 0:
           n.load_mode_one(net_opt.load_epoch)

        models_list.append(n)

    data_loader, ood_test_data_loader, ood_valid_data_loader = get_dataloader(size=n.size,
                                                                              dataset_classes=[[7]],
                                                                              SRV=n.SRV,
                                                                              batch_size=1,
                                                                              n_workers=n.n_workers,
                                                                              augm_config=n.augm_config,
                                                                              cutout_params=[[0],[0]],
                                                                              drop_last_flag=False)


    baseline_network=models_list[7]
    pair_networks=models_list[0:7]#list of networks in descent order. First position is 0 vs all, second is 1 vs all and so on
    with torch.no_grad():
        baseline_network.n.eval()

        #with pollo function
        acc,w_acc,calib,conf_matrix,_=eval(baseline_network, baseline_network.valid_data_loader, *baseline_network.calibration_variables[1], with_temp_scal=False, compute_separate_metrics_for_errors=False)

        print("VALID DATASET POLLO FUNCTION BASELINE")
        print("ACC {} wACC {}".format(acc,w_acc))

        acc,w_acc,calib,conf_matrix,_=eval(baseline_network, baseline_network.test_data_loader, *baseline_network.calibration_variables[2], with_temp_scal=False, compute_separate_metrics_for_errors=False)

        print("TEST DATASET POLLO FUNCTION BASELINE")
        print("ACC {} wACC {}".format(acc,w_acc))
 
        #analisis of baseline network first
        corr_baseline_network,incorr_baseline_network=[0.0]*2
        for image,t_true,_ in baseline_network.valid_data_loader:
            image,t_true=image.cuda(),t_true.cuda()
            t_baseline=torch.argmax(baseline_network.n(image),dim=1)

            
            if t_true == t_baseline:
                corr_baseline_network+=1
	
            else:
                incorr_baseline_network+=1
           
        print("VALID DATASET BASELINE")
        print("Of total samples {} getting correct {} incorrect {}".format(corr_baseline_network+incorr_baseline_network,corr_baseline_network,incorr_baseline_network))

        #analisis of baseline network first
        corr_baseline_network,incorr_baseline_network=[0.0]*2
        for image,t_true,_ in baseline_network.test_data_loader:
            image,t_true=image.cuda(),t_true.cuda()
            t_baseline=torch.argmax(baseline_network.n(image),dim=1)

            if t_true == t_baseline:
                corr_baseline_network+=1

            else:
                incorr_baseline_network+=1
           
        print("TEST DATASET BASELINE")
        print("Of total samples {} getting correct {} incorrect {}".format(corr_baseline_network+incorr_baseline_network,corr_baseline_network,incorr_baseline_network))
      
        #as the dataloader changes we have to recompute the label for each class. This means that samples whose label is above OOD_label should change its value to target-1. The OOD sample will be change to class 7
        correct_classified,incorrect_classified,as_OOD_but_is_IID_corr,as_OOD_but_is_IID_incorr,as_IID_corr,as_IID_incorr,OOD_as_OOD,OOD_as_IID,total_OOD=[0.0]*9
        for image,t_true,_ in baseline_network.valid_data_loader:
            image,t_true=image.cuda(),t_true.cuda()


            t_baseline=torch.argmax(baseline_network.n(image),dim=1)
            pair_net = pair_networks[t_baseline]
            pair_net.n.eval()
            t_pair_net=torch.argmax(pair_net.n(image),dim=1)


            if t_baseline == t_true:
                correct_classified+=1
 
                if t_pair_net==1:
                    #classified as OOD
                    as_OOD_but_is_IID_corr+=1
                else:
                    as_IID_corr+=1

                continue


            if t_baseline != t_true:
                incorrect_classified+=1

                if t_pair_net==1:
                    #classified as OOD
                    as_OOD_but_is_IID_incorr+=1
                else:
                    as_IID_incorr+=1

                continue

        #on the OOD class
        for image,t_true,_ in ood_valid_data_loader:
            image,t_true=image.cuda(),t_true.cuda()


            t_baseline=torch.argmax(baseline_network.n(image),dim=1)
            pair_net = pair_networks[t_baseline]
            pair_net.n.eval()
            t_pair_net=torch.argmax(pair_net.n(image),dim=1)
                
            if t_pair_net==1:
                OOD_as_OOD +=1

            elif t_pair_net==0:
                OOD_as_IID +=1

            total_OOD+=1

        print("===================") 
        print("===================") 
        print("VALIDATION DATASET")
        print("OOD DETECTION RATE: Total OOD samples {}, detected {}, missclassified {}".format(total_OOD,OOD_as_OOD,OOD_as_IID))
        print("IDD DETECTION RATE on CORRECT CLASSIFIED: Total samples {}, as OOD {}, as IID {}".format(correct_classified,as_OOD_but_is_IID_corr,as_IID_corr))
        print("IDD DETECTION RATE on INCORRECT CLASSIFIED: Total samples {}, as OOD {}, as IID {}".format(incorrect_classified,as_OOD_but_is_IID_incorr,as_IID_incorr))

        correct_classified,incorrect_classified,as_OOD_but_is_IID_corr,as_OOD_but_is_IID_incorr,as_IID_corr,as_IID_incorr,OOD_as_OOD,OOD_as_IID,total_OOD=[0.0]*9
        for image,t_true,_ in baseline_network.test_data_loader:
            image,t_true=image.cuda(),t_true.cuda()

            t_baseline=torch.argmax(baseline_network.n(image),dim=1)
            pair_net = pair_networks[t_baseline]
            pair_net.n.eval()
            t_pair_net=torch.argmax(pair_net.n(image),dim=1)


            if t_baseline == t_true:
                correct_classified+=1
 
                if t_pair_net==1:
                    #classified as OOD
                    as_OOD_but_is_IID_corr+=1
                else:
                    as_IID_corr+=1

                continue


            if t_baseline != t_true:
                incorrect_classified+=1

                if t_pair_net==1:
                    #classified as OOD
                    as_OOD_but_is_IID_incorr+=1
                else:
                    as_IID_incorr+=1

                continue


        for image,t_true,_ in ood_test_data_loader:
            image,t_true=image.cuda(),t_true.cuda()

            t_baseline=torch.argmax(baseline_network.n(image),dim=1)
            pair_net = pair_networks[t_baseline]
            pair_net.n.eval()
            t_pair_net=torch.argmax(pair_net.n(image),dim=1)

                
            if t_pair_net==1:
                OOD_as_OOD +=1

            elif t_pair_net==0:
                OOD_as_IID +=1

            total_OOD += 1

        print("===================") 
        print("===================") 
        print("TEST DATASET")
        print("OOD DETECTION RATE: Total OOD samples {}, detected {}, missclassified {}".format(total_OOD,OOD_as_OOD,OOD_as_IID))
        print("IDD DETECTION RATE on CORRECT CLASSIFIED: Total samples {}, as OOD {}, as IID {}".format(correct_classified,as_OOD_but_is_IID_corr,as_IID_corr))
        print("IDD DETECTION RATE on INCORRECT CLASSIFIED: Total samples {}, as OOD {}, as IID {}".format(incorrect_classified,as_OOD_but_is_IID_incorr,as_IID_incorr))

           

    
