import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import random
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets

from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import shutil
from tqdm import tqdm
import argparse

np.random.seed(42)

class ImageNetSubset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label

def load_data(data_folder):
    all_data = []
    all_labels = []
    for idx in tqdm(range(1, 11)):  # Assuming there are 10 batches for training
        batch_file = os.path.join(data_folder, f'train_data_batch_{idx}')
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            all_data.append(batch['data'])
            all_labels.extend(batch['labels'])
            # print(batch['labels'][0:20])
    return np.concatenate(all_data), np.array(all_labels)


def select_and_remap_classes(data, labels, num_classes=10):
    unique_classes = np.unique(labels)
    selected_classes = np.random.choice(unique_classes, num_classes, replace=False)
    
    # Create a mapping for the selected classes to [0, num_classes-1]
    class_mapping = {old_class: new_class for new_class, old_class in enumerate(selected_classes)}
    
    # Filter and remap labels
    selected_indices = [i for i, label in enumerate(labels) if label in selected_classes]
    remapped_labels = [class_mapping[label] for label in labels[selected_indices]]
    
    # Apply selection and remapping
    selected_data = data[selected_indices]
    
    return selected_data, np.array(remapped_labels), class_mapping


def load_test_data(data_folder, class_mapping):
    test_file = os.path.join(data_folder, 'val_data')  # Adjust the file name/path as necessary
    with open(test_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    x = batch['data']
    y = batch['labels']

    # Adjust labels to match the training data's remapping and selection
    y = [class_mapping.get(label, -1) for label in y]  # Subtract 1 since original labels start at 1
    valid_indices = [i for i, label in enumerate(y) if label >= 0]
    x = x[valid_indices]
    y = [y[i] for i in valid_indices]

    return x, y


##################################  OOD Datasets   ############################################################

class INaturalistDataLoader:
    def __init__(self, root_dir, version='2021_train_mini', target_type='full', batch_size=32, download=True):
        self.root_dir = root_dir
        self.version = version
        self.target_type = target_type
        self.batch_size = batch_size
        self.download = download
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_data_loader(self, shuffle=True, num_workers=4):
        # Initialize the INaturalist dataset
        dataset = datasets.INaturalist(root=self.root_dir, version=self.version, target_type=self.target_type, transform=self.transform, download=self.download)

        # Create and return a DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers)




class SUN397DataLoader:
    def __init__(self, root_dir, batch_size=32, download=True):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.download = download
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = None  # Define if you need to apply any transformations to the labels

    def get_data_loader(self, shuffle=True, num_workers=4):
        # Initialize the SUN397 dataset
        dataset = datasets.SUN397(root=self.root_dir, transform=self.transform, target_transform=self.target_transform, download=self.download)

        # Create and return a DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers)



class Places365DataLoader:
    def __init__(self, root_dir, split='val', small=True, download=True, batch_size=32):
        self.root_dir = root_dir
        self.split = split
        self.small = small
        self.download = download
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = None  # Optional, define if needed

    def get_data_loader(self, shuffle=True, num_workers=4):
        # Initialize the Places365 dataset
        dataset = datasets.Places365(root=self.root_dir, split=self.split, small=self.small,
                                     download=self.download, transform=self.transform,
                                     target_transform=self.target_transform)

        # Create and return a DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers)


class DTDDataLoader:
    def __init__(self, root_dir, split='train', partition=1, download=False, batch_size=32):
        self.root_dir = root_dir
        self.split = split
        self.partition = partition
        self.download = download
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def get_data_loader(self, shuffle=True, num_workers=4):
        # Initialize the DTD dataset
        train_dataset = datasets.DTD(root=self.root_dir, split=self.split, partition=self.partition,
                               transform=self.transform, download=self.download)
        val_dataset = datasets.DTD(root=self.root_dir, split='val', transform=self.transform, download=True)
        combined_dataset = ConcatDataset([train_dataset, val_dataset])


        # Create and return a DataLoader
        print(len(combined_dataset))
        return DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers)

class SVHNDataLoader:
    def __init__(self, root_dir, split='train', download=True, batch_size=32):

        self.root_dir = root_dir
        self.split = split
        self.download = download
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize the image to 32x32 pixels
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
            # These normalization values are approximations; adjust as necessary.
        ])
        self.target_transform = None  # Optional, define if needed

    def get_data_loader(self, shuffle=True, num_workers=4):

        dataset = datasets.SVHN(root=self.root_dir, split=self.split, download=self.download,
                                transform=self.transform, target_transform=self.target_transform)

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers)
        return data_loader


class ImageSubfolder(DatasetFolder):
    """Extend ImageFolder to support fold subsets
    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        class_to_idx (dict): Dict with items (class_name, class_index).
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        class_to_idx: Optional[Dict] = None,
    ):
        super(DatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        if class_to_idx is not None:
            classes = class_to_idx.keys()
        else:
            classes, class_to_idx = self.find_classes(self.root)
        extensions = IMG_EXTENSIONS if is_valid_file is None else None,
        samples = self.make_dataset(self.root, class_to_idx, extensions[0], is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples


def imagenet100_set_loader(bsz):
    train_transform = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        # trn.RandomResizedCrop(size=(224, 224), scale=(0.5, 1), interpolation=trn.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    root_dir = 'data/'
    train_dir = root_dir + 'val'
    classes, _ = torchvision.datasets.folder.find_classes(train_dir)
    index = [125, 788, 630, 535, 474, 694, 146, 914, 447, 208, 182, 621, 271, 646, 328, 119, 772, 928, 610, 891, 340,
             890, 589, 524, 172, 453, 869, 556, 168, 982, 942, 874, 787, 320, 457, 127, 814, 358, 604, 634, 898, 388,
             618, 306, 150, 508, 702, 323, 822, 63, 445, 927, 266, 298, 255, 44, 207, 151, 666, 868, 992, 843, 436, 131,
             384, 908, 278, 169, 294, 428, 60, 472, 778, 304, 76, 289, 199, 152, 584, 510, 825, 236, 395, 762, 917, 573,
             949, 696, 977, 401, 583, 10, 562, 738, 416, 637, 973, 359, 52, 708]

    classes = [classes[i] for i in index]
    # print(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_data = ImageSubfolder(root_dir + 'train', transform=train_transform, class_to_idx=class_to_idx)
    test_data = ImageSubfolder(root_dir + 'val', transform=test_transform, class_to_idx=class_to_idx)
    return train_data, test_data
    # labeled_trainloader = torch.utils.data.DataLoader(train_data, batch_size=bsz, shuffle=True, num_workers=16,
    #                                                   pin_memory=True, drop_last=True)
    # testloader = torch.utils.data.DataLoader(test_data, batch_size=bsz, shuffle=True, num_workers=16, pin_memory=True)
    
    # return labeled_trainloader, testloader

def imagenet10_set_loader(bsz, dset_id, small=True):
    n = 32 if small else 224
    train_transform = transforms.Compose([
        transforms.Resize(size=(n, n), interpolation=transforms.InterpolationMode.BICUBIC),
        # trn.RandomResizedCrop(size=(224, 224), scale=(0.5, 1), interpolation=trn.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(size=(n, n), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(n, n)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    root_dir = 'data/'
    train_dir = root_dir + 'val'
    classes, _ = torchvision.datasets.folder.find_classes(train_dir)

    # # Choose class
    indices = [[895, 817, 10, 284, 352, 238, 30, 569, 339, 510],
               [648, 506, 608, 640, 539, 548, 446, 183, 809, 127],
               [961, 316, 227, 74, 322, 480, 933, 508, 158, 367],
               [247, 202, 622, 351, 367, 523, 796, 91, 39, 54],
               [114, 183, 841, 870, 730, 756, 554, 799, 97, 150],
               [795, 854, 631, 581, 669, 573, 310, 900, 569, 598],
               [310, 404, 382, 136, 786, 97, 858, 970, 391, 688],
               [744, 437, 606, 909, 96, 951, 384, 43, 461, 247],
               [534, 358, 139, 955, 304, 879, 998, 319, 359, 904],
               [461, 29, 22, 254, 560, 232, 700, 45, 363, 321],
               [8, 641, 417, 181, 813, 64, 396, 437, 7, 178]]
    index = indices[dset_id]

    classes = [classes[i] for i in index]
    # print(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_data = ImageSubfolder(root_dir + 'train', transform=train_transform, class_to_idx=class_to_idx)
    test_data = ImageSubfolder(root_dir + 'val', transform=test_transform, class_to_idx=class_to_idx)
    return train_data, test_data

def imagenet10_32():
    data, labels = load_data('data/ImageNet32_train')
    selected_data, selected_labels, class_mapping = select_and_remap_classes(data, labels, num_classes = 10)
    selected_data = torch.tensor(selected_data, dtype=torch.float) / 255.0
    selected_labels = torch.tensor(selected_labels, dtype=torch.long)
    train_set = ImageNetSubset(selected_data, selected_labels)
    total_size = len(train_set)
    train_ratio = 0.8
    val_ratio = 0.2

    # Calculate sizes for each split
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio) + 1
    print(train_size, val_size)
    train_set, test_set = torch.utils.data.random_split(train_set, [train_size, val_size])
    return train_set, test_set

# DEPRECATED!
def create_imagenet_subset():
    src_dir = os.path.join('data')
    dst_path = os.path.join('data', "ImageNet10")
    os.makedirs(dst_path, exist_ok=True)

    cls_list = ["n04552348", "n04285008", "n01530575", "n02123597", "n02422699", 
                "n02107574", "n01641577", "n03417042", "n02389026", "n03095699"]

    for split in ['train', 'val']:
        for c in tqdm(cls_list):
            shutil.copytree(os.path.join(src_dir, split, c), os.path.join(dst_path, split, c), dirs_exist_ok=True)

def sample_class_with_seed(n=10, seed=2024):
    np.random.seed(seed)
    for _ in range(n):
        print(list(np.random.choice(1000, 10)))
        

if __name__ == '__main__':
    pass
    # create_imagenet_subset()
    # imagenet10_set_loader(512)
    sample_class_with_seed()
    