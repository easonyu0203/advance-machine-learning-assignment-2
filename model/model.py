from typing import Optional

import torch
from torch import nn

from model.base_cnn import GenericCNN
from model.nal import ClassifierWithNAL


def get_model(model_type: str, transition_matrix: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Factory function to get the specified CNN model.

    Args:
    model_type (str): Type of model to create ('mnist' or 'cifar').

    Returns:
    GenericCNN: Instantiated CNN model for the specified type.

    Raises:
    ValueError: If the model_type is not 'mnist' or 'cifar'.
    """
    if model_type.lower() == "mnist":
        return GenericCNN(input_channels=1, conv_layers=[32, 64], fc_layers=[128], num_classes=3)
    elif model_type.lower() == "cifar":
        return GenericCNN(input_channels=3, conv_layers=[64, 128, 256], fc_layers=[256, 256], num_classes=3)
    elif model_type.lower() == "mnist_nal":
        base_model = get_model('mnist')
        return ClassifierWithNAL(base_model, num_classes=3, transition_matrix=transition_matrix)
    elif model_type.lower() == "cifar_nal":
        base_model = get_model('cifar')
        return ClassifierWithNAL(base_model, num_classes=3, transition_matrix=transition_matrix)
    else:
        raise ValueError("Invalid model type. Please choose 'mnist' or 'cifar'.")
