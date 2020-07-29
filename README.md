# ISIC_challenge_2019

###  Baseline Ensemble

|     Network       | augm config |  cut out holes  |  cut out pads   |      Epoch      |    Accuracy    |  Balanced Accuracy  |    Ensemble 0  |    Ensemble 1  |    Ensemble 2  |    Ensemble 3  |    Ensemble 4  |
|:-----------------:|:-----------:|:---------------:|:---------------:|:---------------:|---------------:|:-------------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
|   Densenet 201    |     84      |      1, 2, 3    |   20, 50, 100   |       118       |      0.857     |        0.857        |     &#9745;    |     &#9744;    |     &#9745;    |     &#9745;    |     &#9745;    |
|   Densenet 201    |     84      |      1, 2, 3    |   20, 50, 100   |       115       |      0.856     |        0.850        |     &#9745;    |     &#9744;    |     &#9745;    |     &#9744;    |     &#9744;    |
|   Densenet 201    |     84      |      1, 2, 3    |   20, 50, 100   |        72       |      0.854     |        0.853        |     &#9745;    |     &#9744;    |     &#9744;    |     &#9745;    |     &#9744;    |
|   Densenet 201    |     116     | 0, 0, 1, 2, 3   |   20, 50, 100   |       104       |      0.866     |        0.852        |     &#9745;    |     &#9744;    |     &#9745;    |     &#9745;    |     &#9745;    |
|   Densenet 201    |     16      |       None      |       None      |       102       |      0.859     |        0.858        |     &#9745;    |     &#9745;    |     &#9745;    |     &#9745;    |     &#9745;    |
|   Densenet 201    |     16      |       None      |       None      |        96       |      0.862     |        0.859        |     &#9745;    |     &#9745;    |     &#9745;    |     &#9744;    |     &#9744;    |
|   Densenet 201    |     16      |       None      |       None      |        86       |      0.861     |        0.862        |     &#9745;    |     &#9745;    |     &#9744;    |     &#9745;    |     &#9744;    |
|   Densenet 201    |     16      |       None      |       None      |        70       |      0.864     |        0.850        |     &#9745;    |     &#9745;    |     &#9744;    |     &#9744;    |     &#9744;    |
|   Densenet 201    |     84      |       None      |       None      |        79       |      0.854     |        0.850        |     &#9745;    |     &#9744;    |     &#9745;    |     &#9745;    |     &#9745;    |
|   Densenet 201    |     84      |       None      |       None      |        69       |      0.851     |        0.857        |     &#9745;    |     &#9744;    |     &#9745;    |     &#9745;    |     &#9744;    |
|    Resnet 152     |     16      |       None      |       None      |        83       |      0.862     |        0.851        |     &#9745;    |     &#9745;    |     &#9745;    |     &#9745;    |     &#9744;    |
|    Resnet 152     |     16      |       None      |       None      |        82       |      0.865     |        0.857        |     &#9745;    |     &#9745;    |     &#9745;    |     &#9744;    |     &#9744;    |
|    Resnet 152     |     16      |       None      |       None      |        77       |      0.857     |        0.854        |     &#9745;    |     &#9745;    |     &#9744;    |     &#9745;    |     &#9744;    |
|    Resnet 152     |     16      |       None      |       None      |        76       |      0.863     |        0.852        |     &#9745;    |     &#9745;    |     &#9744;    |     &#9744;    |     &#9744;    |
|                   |             |                 |                 |                 |                |                     |                |                |                |                |                |                                                                                                                               
|                   |             |                 |                 |                 |                |**Balanced Accuracy:**|    **0.886**   |    **0.874**   |    **0.886**   |    **0.885**   |    **0.876**   |


augm config is a *code* obtained as the sum of employed data augmentation strategy following values:

``` python

self.possible_aug_list = [
            None,  # dummy for padding mode                                                         # 1
            None,  # placeholder for future inclusion                                               # 2
            sometimes(ia.augmenters.AdditivePoissonNoise((0, 10), per_channel=True)),               # 4
            sometimes(ia.augmenters.Dropout((0, 0.02), per_channel=False)),                         # 8
            sometimes(ia.augmenters.GaussianBlur((0, 0.8))),                                        # 16
            sometimes(ia.augmenters.AddToHueAndSaturation((-20, 10))),                              # 32
            sometimes(ia.augmenters.GammaContrast((0.5, 1.5))),                                     # 64
            None,  # placeholder for future inclusion                                               # 128
            None,  # placeholder for future inclusion                                               # 256
            sometimes(ia.augmenters.PiecewiseAffine((0, 0.04))),                                    # 512
            sometimes(ia.augmenters.Affine(shear=(-20, 20), mode=self.mode)),                       # 1024
            sometimes(ia.augmenters.CropAndPad(percent=(-0.2, 0.05), pad_mode=self.mode))           # 2048
        ]

```
Random Flipping and Rotations are **always** employed. Images are not simply resized, yet filled to be perfectly sqared and **then** resized to 512x512

Every network in this table was trained with:
- batch size: 8
- starting learning rate: 0.001
- Optimizer: SGD
- Scheduler: Plateau (on Validation Balanced Accuracy)

Networks are trained for a maximum of 120 epochs, checkpoint that obtain a new best weighted accuracy on the **validation set** are saved.

Every ensemble is obtained by calibrating each model with Temp Scal strategy, with **NO** Data Augmentation ensemble employed.

To recalculate the results of " Ensemble 0 ":

``` bash
models_ensemble.py --avg big_ensemble_0.txt --calibrated
```

where big_ensemble_0.txt is a simple txt file containing only the following lines.

``` txt
--network densenet201 --batch_size 8 --save_dir my_dir --SRV --optimizer SGD --scheduler plateau --cutout_holes 1 2 3 --cutout_pad 20 50 100  --augm_config 84 --load_epoch 118 
--network densenet201 --batch_size 8 --save_dir my_dir --SRV --optimizer SGD --scheduler plateau --cutout_holes 1 2 3 --cutout_pad 20 50 100  --augm_config 84 --load_epoch 115 
--network densenet201 --batch_size 8 --save_dir my_dir --SRV --optimizer SGD --scheduler plateau --cutout_holes 1 2 3 --cutout_pad 20 50 100  --augm_config 84 --load_epoch 72 
--network densenet201 --batch_size 8 --save_dir my_dir --SRV --optimizer SGD --scheduler plateau --cutout_holes 0 0 1 2 3 --cutout_pad 20 50 100  --augm_config 116 --load_epoch 104 
--network densenet201 --batch_size 8 --save_dir my_dir --SRV --optimizer SGD --scheduler plateau --augm_config 16 --load_epoch 102
--network densenet201 --batch_size 8 --save_dir my_dir --SRV --optimizer SGD --scheduler plateau --augm_config 16 --load_epoch 96
--network densenet201 --batch_size 8 --save_dir my_dir --SRV --optimizer SGD --scheduler plateau --augm_config 16 --load_epoch 86
--network densenet201 --batch_size 8 --save_dir my_dir --SRV --optimizer SGD --scheduler plateau --augm_config 16 --load_epoch 70
--network densenet201 --batch_size 8 --save_dir my_dir --SRV --optimizer SGD --scheduler plateau --augm_config 84 --load_epoch 79
--network densenet201 --batch_size 8 --save_dir my_dir --SRV --optimizer SGD --scheduler plateau --augm_config 84 --load_epoch 69
--network resnet152 --batch_size 8 --save_dir my_dir --SRV --optimizer SGD --scheduler plateau --augm_config 16 --load_epoch 83
--network resnet152 --batch_size 8 --save_dir my_dir --SRV --optimizer SGD --scheduler plateau --augm_config 16 --load_epoch 82
--network resnet152 --batch_size 8 --save_dir my_dir --SRV --optimizer SGD --scheduler plateau --augm_config 16 --load_epoch 77
--network resnet152 --batch_size 8 --save_dir my_dir --SRV --optimizer SGD --scheduler plateau --augm_config 16 --load_epoch 76
```
