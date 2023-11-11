from collections import namedtuple
import numpy as np
from utils.plot_utils import display_dataset_images, display_dataset_images_for_all_labels
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)

        return x, y
    

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define a namedtuple type for datasets with more descriptive names
DatasetFile = namedtuple('DatasetFile', ['X_train', 'y_train', 'X_test', 'y_test'])

# Paths to the dataset files
CIFAR_DATASET_PATH = "./datasets/CIFAR.npz"
MNIST_DATASET5_PATH = "./datasets/FashionMNIST0.5.npz"
MNIST_DATASET6_PATH = "./datasets/FashionMNIST0.6.npz"

# Function to load a dataset and return a namedtuple
def load_dataset_file(path):
    data = np.load(path)
    return DatasetFile(X_train=data['Xtr'], y_train=data['Str'], X_test=data['Xts'], y_test=data['Yts'])


def create_dataloader(dataset_name, is_train=True, batch_size=64, shuffle=True):
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

    if is_train:
        dataset = CustomDataset(dataset_file.X_train, dataset_file.y_train, transform=transform)
    else:
        dataset = CustomDataset(dataset_file.X_test, dataset_file.y_test, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
