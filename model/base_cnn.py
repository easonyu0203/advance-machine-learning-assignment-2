import torch
import torch.nn as nn


def conv_bn(in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1) -> nn.Sequential:
    """
    Creates a convolutional block with batch normalization, ReLU activation, max pooling, and dropout.

    Args:
    in_channels (int): Number of channels in the input image.
    out_channels (int): Number of channels produced by the convolution.
    kernel_size (int): Size of the convolving kernel.
    padding (int): Zero-padding added to both sides of the input.

    Returns:
    nn.Sequential: A sequential container with specified layers.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(0.25)
    )

def fc_bn(in_features: int, out_features: int) -> nn.Sequential:
    """
    Creates a fully connected block with batch normalization, ReLU activation, and dropout.

    Args:
    in_features (int): Size of each input sample.
    out_features (int): Size of each output sample.

    Returns:
    nn.Sequential: A sequential container with specified layers.
    """
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    )

class GenericCNN(nn.Module):
    """
    A generic CNN model for image classification.

    Attributes:
    conv_layers (nn.Sequential): Sequential container of convolutional layers.
    fc_layers (nn.Sequential): Sequential container of fully connected layers.
    output_layer (nn.Linear): Final linear layer for classification.
    """
    def __init__(self, input_channels: int, conv_layers: list, fc_layers: list, num_classes: int):
        super(GenericCNN, self).__init__()
        self.conv_layers = nn.Sequential(*[conv_bn(input_channels if i == 0 else conv_layers[i-1], l) for i, l in enumerate(conv_layers)])
        
        conv_output_size = self._calculate_conv_output(input_channels)
        flattened_size = conv_output_size * conv_output_size * conv_layers[-1]
        
        self.fc_layers = nn.Sequential(*[fc_bn(flattened_size if i == 0 else fc_layers[i-1], l) for i, l in enumerate(fc_layers)])
        self.output_layer = nn.Linear(fc_layers[-1], num_classes)

    def _calculate_conv_output(self, input_channels: int) -> int:
        """
        Calculates the size of the output tensor after passing through the convolutional layers.

        Args:
        input_channels (int): Number of channels in the input image.

        Returns:
        int: Size of one dimension of the square output tensor.
        """
        input_size = 32 if input_channels == 3 else 28  # CIFAR or FashionMNIST
        dummy_input = torch.zeros(1, input_channels, input_size, input_size)
        output = self.conv_layers(dummy_input)
        return output.size(-1)  # Assuming square output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the model.
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = self.output_layer(x)
        return x

