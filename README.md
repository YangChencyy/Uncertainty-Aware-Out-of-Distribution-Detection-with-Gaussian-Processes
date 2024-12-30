# **Uncertainty-Aware Out-of-Distribution Detection with Gaussian Processes**


## **Description**

This repository contains the official implementation of the paper:  
**"Uncertainty-Aware Out-of-Distribution Detection with Gaussian Processes"**

Deep neural networks (DNNs) often fail to handle out-of-distribution (OOD) data, leading to overconfident and incorrect predictions, especially in safety-critical tasks. Existing OOD detection methods typically rely on curated OOD data or hyperparameter tuning, limiting their effectiveness without exposure to OOD samples during training.

To address this, we propose a Gaussian-process-based OOD detection method that establishes a detection boundary using only in-distribution (InD) data. By quantifying uncertainty in softmax scores with a clustered Gaussian process, our approach defines a score function to separate InD and OOD data based on differences in their posterior predictive distributions. The proposed method consistently outperforms state-of-the-art techniques in detection accuracy while requiring no OOD data during training.


---

## **Getting Started**

### **1. Clone the Repository**
```bash
git clone https://github.com/YangChencyy/Uncertainty-Aware-Out-of-Distribution-Detection-with-Gaussian-Processes.git
cd Uncertainty-Aware-Out-of-Distribution-Detection-with-Gaussian-Processes
```

### **2. Install Dependencies**

This project requires dependencies for both Python and R.

#### **1. Python**
We recommend using a virtual environment to manage Python dependencies. To set up the environment and install the required packages, run:

```bash
python -m venv env
source env/bin/activate  # For Linux/Mac
env\Scripts\activate     # For Windows
pip install -r requirements.txt
```

#### **2. R**
Ensure R is installed on your system. Then, install the required R packages using the provided script:

```bash
Rscript install_packages.R
```

### **3. Download Dataset**
Here we provide the following instructions for downloading relevant datasets.

#### InD Dataset

The ImageNet10 dataset that we used is a subset of the well-known ImageNet1k (ILSVRC-2012) dataset, which can be downloaded from [this website](https://www.image-net.org/index.php). The downloaded training and validation dataset should be extracted and placed in `./NN-Training/data/train` and `./NN-Training/data/val`, respectively. The 10 classes are pre-chosen and are hard-coded in our codebase. For more details, we kindly refer the audience to the codebase developed by [Ming et. al (2022)](https://github.com/deeplearning-wisc/MCM/tree/main?tab=readme-ov-file). For simple benchmarking MNIST dataset, it can be easily retrieved from Pytorch dataset library.

#### OOD Dataset

For downloading OOD datasets, we kindly refer the audience to the instructions mentioned in [this repository](https://github.com/deeplearning-wisc/cider). However, please keep in mind that all downloaded and extracted data should be placed under the folder `./NN-Training/data/`. For simple OOD datasets used in MNIST benchmarking experiment, including CIFAR10, FashionMNIST, and [mini-ImageNet](https://drive.google.com/file/d/1Kot50VljGnN4exQtxN76_PoJhPrFJTim/view?usp=sharing), they can be either retrieved from the provided link or Pytorch dataset library.


## **Usage**
### **1. Trian the Neural Network**
From the repo directory, 
```
cd NN-Training
```

#### MNIST experiment

For MNIST experiment, as the classification task is not very difficult, we choose to hardcode all parameters in the script; the command is given by:

```
python mnist.py
```

However, those parameters, including learning rate, feature size, batch size, and epochs, can be adjusted easily in `mnist.py`.

#### ImageNet experiment

To train a classifier for ImageNet10 dataset and evaluate them on all OOD datasets, one example command is provided as follows:

```
python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=32 --dset_id 0 --train --eval_train
```

Note that the learning rate, batch size, and the number of features in the penultimate layer can be adjusted from the command. To evaluate on the OOD datasets, we can simply specify the dataset by adding `--ood <dset name>` and remove the `--train` and `--eval_train` flag to avoid repetitve training. For instance, one example command is provided as follows:

```
python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=32 --dset_id 0 --ood Places365
```

After training and evaluation, the trained model checkpoint as well as all features are saved under the folder `ckpt/`. These features and logits will then be utilized for training of GP models.


### **2. Gaussian process**
Gaussain process training will utilize checkpoint data we got from previous steps. We recommend putting all data from `./NN-Training/ckpt` to `GP-Fitting/data` （only .csv files are needed）, or you can modify the directory in laGP_ImageNet.R or laGP_MNIST.R directly.

First, from the main directory,
```
cd GP-Fitting
```

Ensure R is installed on your system. Refer to the [R installation guide](https://cran.r-project.org/) for instructions. Then run the following command to install all required R packages:
```
Rscript install_packages.R
```

#### MNIST experiment
Once all dependencies are installed, you can execute the main script for MNIST by running,
```
Rscript laGP_MNIST.R
```
After running laGP_MNIST.R, the outputs will be saved in the directory


#### ImageNet experiment