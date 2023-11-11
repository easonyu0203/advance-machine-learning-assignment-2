from collections import namedtuple
import numpy as np
from utils.plot_utils import display_dataset_images, display_dataset_images_for_all_labels
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Any, Optional

# Custom dataset class for handling image datasets
class ImageDataset(Dataset):
    """
    A dataset class for loading and transforming image datasets.

    Attributes:
        data (np.ndarray): The image data.
        targets (np.ndarray): The target labels.
        transform (Optional[transforms.Compose]): Transformations to be applied to the images.
    """
    def __init__(self, data: np.ndarray, targets: np.ndarray, transform: Optional[transforms.Compose] = None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        x = self.data[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

# Transformations for MNIST and CIFAR datasets
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Namedtuple for holding dataset paths
DatasetFile = namedtuple('DatasetFile', ['X_train', 'y_train', 'X_test', 'y_test'])

# Dataset file paths
CIFAR_DATASET_PATH = "../datasets/CIFAR.npz"
MNIST_DATASET5_PATH = "../datasets/FashionMNIST0.5.npz"
MNIST_DATASET6_PATH = "../datasets/FashionMNIST0.6.npz"

def load_dataset_file(path: str) -> DatasetFile:
    """
    Load a dataset file and return a namedtuple containing train and test data.

    Args:
        path (str): Path to the dataset file.

    Returns:
        DatasetFile: A namedtuple containing train and test data and labels.
    """
    data = np.load(path)
    return DatasetFile(X_train=data['Xtr'], y_train=data['Str'], X_test=data['Xts'], y_test=data['Yts'])

def create_dataset(dataset_name: str, is_train: bool = True) -> Dataset:
    """
    Create a Dataset for a specified dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'CIFAR', 'MNIST5').
        is_train (bool): Flag to indicate whether to load training data (True) or testing data (False).

    Returns:
        Dataset: A Dataset for the specified dataset.
    """
    if dataset_name == 'CIFAR':
        dataset_file = load_dataset_file(CIFAR_DATASET_PATH)
        transform = cifar_transform
    elif dataset_name == 'MNIST5':
        dataset_file = load_dataset_file(MNIST_DATASET5_PATH)
        transform = mnist_transform
    elif dataset_name == 'MNIST6':
        dataset_file = load_dataset_file(MNIST_DATASET6_PATH)
        transform = mnist_transform
    else:
        raise ValueError("Unknown dataset")

    return ImageDataset(dataset_file.X_train if is_train else dataset_file.X_test,
                           dataset_file.y_train if is_train else dataset_file.y_test,
                           transform=transform)

    

