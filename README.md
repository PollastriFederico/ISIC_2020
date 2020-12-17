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
