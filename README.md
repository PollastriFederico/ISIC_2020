<!--
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
-->

# SIIM-ISIC Melanoma Classification 2020

This repository contains the python project that we employed to take part in the [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification/) competition, in August 2020.

Our best result is an AUC of **0.9162** on the private portion on the test set.

The ensemble model used inthe final submission is composed of 8 variations of ResNet152 and SEResNeXt101.

|Network|Learning Rate|Optimizer|Augm Config|Loss Function|Cutout Holes|Cutout Pad|Pretrained on ISIC 2019|
|:-----:|:-----------:|:-:|:-:|:-:|:-:|:-:|:-:|
|SEResNeXt101|0.001|SGD|16|Cross-Entropy|0 0 1 2 3|20 50 100|No|
|ResNet152|0.001|SGD|16|Combo|0 0 1 2 3|20 50 100|No|
|SEResNeXt101|0.001|SGD|84|Focal|0 0 1 2 3|20 50 100|No|
|ResNet152|0.001|SGD|84|Cross-Entropy|0 0 1 2 3|20 50 100|No|
|ResNet152|0.0001|SGD|16|Focal|0 0 1 2 3|20 50 100|Yes|
|SEResNeXt101|0.0001|SGD|84|Cross-Entropy|0 0 1 2 3|20 50 100|Yes|
|SEResNeXt101|0.0001|SGD|16|Combo|0 0 1 2 3|20 50 100|Yes|
|ResNet152|0.0001|SGD|84|Cross-Entropy|0 0 1 2 3|20 50 100|Yes|

A brief description of the project follows.

## Dataset

The ISIC 2020 dataset is composed of 33126 training images and 10982 test images, binary labelled as *benignant* or *malignant*.
The dataset root path is specified in the config variable `data_root`, in config.py.
The class `ISIC`, defined in isic_classification_dataset.py, is the subclass of `torch.utils.data.Dataset` used for the training process.
Each instantiation of `ISIC` needs a specific *split*, which is a portion of the total dataset.
The description of a split includes a list of image names and target labels, and comes in the form of a csv file.
The proper binding between splits and csv files is specified in a class variable of `ISIC`, the dictionary `splitsdic`.

### Data loading optimization

The ISIC dataset is composed of a very large amount of relatively small images. Therefore, the training and evaluation processes (evaluation especially) require high reading throughput for a prolonged period of time.
This can cause issues to the underlying hardware responsible for transfering image content from a storage device to central memory.
Two ways to mitigate the problem are Single File Dataset and temporary memory.

#### Single File Dataset

A Single File Dataset (SFD), as the name suggests, is a single file containing a concatenation of all the images of a dataset.
The aim of SFD is to drastically reduce the number of open/close operations on the file system. The use of a SFD is summarized in the folloowing steps:

- Open the SFD
- For each image being processed: move the file pointer to its first byte (*seek*), read *imgsize* bytes
- Close the SFD

A simple SFD file format, used in the project, is described in single_file_dataset.py in the form of a `C struct`, and is reported in the following:

```C
struct Img {
    uint32_t size;                  // image size in bytes
    uint8_t data[size];             // image file data
}

struct Sfd {
    uint32_t length;                // number of images
    uint64_t pos[length];           // position of images, as offset from the beginning
    struct Img[length];             // images
}
```

The content of a SFD file exactly matches a `struct Sfd`.
This format is intentionally very simple. One of its limitations is the lack of image names and labels. In fact, an SFD is (almost) always accompanied to list of image descriptions, usually represented in a csv file.

The use of a SFD, in the class `ISIC`, is compulsory. `ISIC` cannot load single images directly from the file system.
Functions for the creation of SFDs starting from a list of images or from other SFDs are available as static methods of the class `Sfd`, in single_file_dataset.py.
The same file also contains utilities for the creation of SFDs specifically for the ISIC dataset.
When training with multiple workers, the SFD must be opened once for every thread. For this reason, the SFD class needs to know the number of workers.

#### Temporary memory

If the server on which the training/evaluation process is performed is equipped with a dedicated storage device with high reading speed, like an SSD, it is usually appropriate to copy the whole dataset to temporary memory (/tmp on Linux devices). This copy is automatically performed when the `copy_to_tmp` flag of `ISIC` is enabled. Note that it is disabled by default.

## Training process

The training process happens in the framework of the class `ClassifyNet`, defined in classification_net.py.
In order to start training a network, it is sufficient to run the file as a python script.
The list of parameters, along with possible values, is configured near the end of the file.

The training starts instantiating three data loaders, for training, validation and test.
The triplet of splits is determined by parameter `dname`.
For every accepted value of `dname`, the three splits to be used are specified in the function `get_dataloader()`, in data.py, along with the necessary preprocessing operations.

Preprocessing varies between different values of `dname`, but the general structure is the following.

Training:
- Augmentations
- Image to Tensor conversion
- Normalization
- CutOut

Validation and Test:
- Inference Augmentations
- Image to Tensor conversion
- Normalization

After each epoch, metrics are evaluated on the validation set and the scheduler is updated according to the vaidation AUC. 
Network weights and optimizer parameters are periodically saved in the directory specified as `save_dir`.

### Augmentations

Random Flipping and Rotations are always employed. Images are filled to be perfectly squared and then resized to 512x512.

Other transformations can be configured in a bitmask.
Each of 12 bits corresponds to a transformation, expect for some that are reserved for future use.

|Bit|Augmentation|
|:-:|:-|
|0|*None*|
|1|*None*|
|2|AdditivePoissonNoise|
|3|Dropout|
|4|GaussianBlur|
|5|AddToHueAndSaturation|
|6|GammaContrast|
|7|*None*|
|8|*None*|
|9|PiecewiseAffine|
|10|Affine|
|11|CropAndPad|
 
The exact augmentation configuration, and the parameters used for each transformation, are in the class `ImgAugTransform`, defined in data.py.

### Logs and charts

Tensorboard logs are stored, during the training process, in the path specified by the config variable `tensorboard_root`, in config.py. 
Measured metrics are:
- Loss
- AUC on validation and test set
- Fscore on validation and test set
- Recall on validation and test set

## Ensemble

A common way to improve performance just consists of aggregating the predictions of multiple good-performing models.
This approach is followed by the script models_ensemble.py.
In order to use an ensemble for inference, a configuration file must be prepared, with one line for each network.
Each line shall contain a list of parameters for a network that has been already trained. Possible parameters are configured in models_ensemble.py, in the form of a variable of type `argparse.ArgumentParser`, named `net_parser`.
Inference is performed with each network in turn, and predictions are averaged to produce the final output, which is saved in the path set in the configuration variable `ensemble_output_path`.