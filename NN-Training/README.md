# Pre-Training of neural network classification models

## Dataset Preparation

This paper focus on two In-Distribution (InD) and several Out-of-Distribution (OOD) Datasets. Here we provide the following instructions for downloading relevant datasets.

### InD Dataset

The ImageNet10 dataset that we used is a subset of the well-known ImageNet1k (ILSVRC-2012) dataset, which can be downloaded from [this website](https://www.image-net.org/index.php). The downloaded training and validation dataset should be extracted and placed in `./data/train` and `./data/val`, respectively. The 10 classes are pre-chosen and are hard-coded in our codebase. For more details, we kindly refer the audience to the codebase developed by [Ming et. al (2022)](https://github.com/deeplearning-wisc/MCM/tree/main?tab=readme-ov-file). For simple benchmarking MNIST dataset, it can be easily retrieved from Pytorch dataset library.

### OOD Dataset

For downloading OOD datasets, we kindly refer the audience to the instructions mentioned in [this repository](https://github.com/deeplearning-wisc/cider). However, please keep in mind that all downloaded and extracted data should be placed under the folder `./data/`. For simple OOD datasets used in MNIST benchmarking experiment, including CIFAR10, FashionMNIST, and [mini-ImageNet](https://drive.google.com/file/d/1Kot50VljGnN4exQtxN76_PoJhPrFJTim/view?usp=sharing), they can be either retrieved from the provided link or Pytorch dataset library.

## Benchmark Experiment

### MNIST experiment

For MNIST experiment, as the classification task is not very difficult, we choose to hardcode all parameters in the script; the command is given by:

```
python mnist.py
```

However, those parameters, including learning rate, feature size, batch size, and epochs, can be adjusted easily in `mnist.py`.

### ImageNet experiment

To train a classifier for ImageNet10 dataset and evaluate them on all OOD datasets, one example command is provided as follows:

```
python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=32 --dset_id 0 --train --eval_train
```

Note that the learning rate, batch size, and the number of features in the penultimate layer can be adjusted from the command. To evaluate on the OOD datasets, we can simply specify the dataset by adding `--ood <dset name>` and remove the `--train` and `--eval_train` flag to avoid repetitve training. For instance, one example command is provided as follows:

```
python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=32 --dset_id 0 --ood Places365
```

After training and evaluation, the trained model checkpoint as well as all features are saved under the folder `ckpt/`. These features and logits will then be utilized for training of GP models.
