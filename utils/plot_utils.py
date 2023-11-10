import matplotlib.pyplot as plt
from typing import List, Union, Optional
import random
from collections import namedtuple
import numpy as np


Dataset = namedtuple('Dataset', ['X_train', 'y_train', 'X_test', 'y_test'])

def display_dataset_images(dataset: Dataset, 
                           indices: Optional[List[int]] = None, 
                           count: Optional[int] = None) -> None:
    """
    Display images and their labels from the given dataset using subplots.
    If indices are provided, images at those indices are displayed.
    Otherwise, a specified count of random images will be shown.
    
    :param dataset: A namedtuple containing dataset splits (X_train, y_train, X_test, y_test).
    :param indices: A list of indices of the images to display.
    :param count: The number of random images to display if indices are not provided.
    """
    # If no indices or count is provided, raise an error
    if indices is None and count is None:
        raise ValueError("Either indices or count must be provided.")
    
    # If indices are not provided, select random indices based on the count
    if indices is None:
        indices = random.sample(range(len(dataset.X_train)), count)
    
    # Calculate number of rows needed for subplots
    num_images = len(indices)
    num_rows = num_images // 5 + (num_images % 5 > 0)
    
    # Set up the subplot dimensions
    fig, axes = plt.subplots(num_rows, 5, figsize=(15, 3 * num_rows))
    
    # Flatten the axes array for easy iteration if it's 2D
    if num_rows > 1:
        axes = axes.flatten()
    
    # Display each image
    for idx, ax in zip(indices, axes):
        ax.imshow(dataset.X_train[idx], cmap='gray')
        ax.set_title(f'Label: {dataset.y_train[idx]}')
        ax.axis('off')  # Hide the axes
    
    # If there are any empty subplots, hide them
    for ax in axes[num_images:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# display_dataset_images(cifar_dataset, count=10)
# This will display 10 random images from the cifar_dataset


def display_dataset_images_for_all_labels(dataset: Dataset, count: int) -> None:
    """
    Display a specified count of random images for each label in the dataset.

    :param dataset: A namedtuple containing dataset splits (X_train, y_train, X_test, y_test).
    :param count: The number of random images to display for each label.
    """
    # Get unique labels
    unique_labels = np.unique(dataset.y_train)
    
    # For each label, display 'count' number of images
    for label in unique_labels:
        # Get indices of all images with the current label
        label_indices = [i for i, y in enumerate(dataset.y_train) if y == label]

        # Randomly select 'count' indices from these
        selected_indices = random.sample(label_indices, min(count, len(label_indices)))

        # Calculate number of rows needed for subplots
        num_rows = len(selected_indices) // 10 + (len(selected_indices) % 10 > 0)

        # Set up the subplot dimensions
        fig, axes = plt.subplots(num_rows, 10, figsize=(15, 3 * num_rows))
        
        # Flatten the axes array for easy iteration if it's 2D
        if num_rows > 1:
            axes = axes.flatten()
        
        # Display each image
        for idx, ax in zip(selected_indices, axes.flatten()):
            ax.imshow(dataset.X_train[idx], cmap='gray')
            ax.set_title(f'Label: {dataset.y_train[idx]}')
            ax.axis('off')  # Hide the axes
        
        # If there are any empty subplots, hide them
        for ax in axes[len(selected_indices):]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()