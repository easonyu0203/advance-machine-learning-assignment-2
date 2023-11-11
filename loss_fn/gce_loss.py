import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional


class GCELoss(nn.Module):
    """
    Generalized Cross Entropy Loss.

    Attributes:
    alpha (float): Hyperparameter alpha in the range [0, 1] controlling the noise robustness.
    epsilon (float): Small constant for numerical stability.
    """
    def __init__(self, alpha: float, epsilon: float = 1e-6):
        """
        Initialize the GCELoss class.

        Args:
        alpha (float): Hyperparameter alpha in the range [0, 1].
        epsilon (float): Small constant for numerical stability.
        """
        super(GCELoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the GCE loss.

        Args:
        output (torch.Tensor): Logits tensor from the model (N, C), where C is the number of classes.
        targets (torch.Tensor): One-hot encoded targets labels (N, C), where C is the number of classes.

        Returns:
        torch.Tensor: Computed GCE loss.
        """
        one_hot = F.one_hot(targets, num_classes=output.size()[-1]).float()  # Convert targets to one-hot encoding
        probabilities = F.softmax(output, dim=1)  # Apply softmax to convert logits to probabilities
        clipped_probabilities = torch.clamp(probabilities, self.epsilon, 1.0 - self.epsilon)

        gce_loss = torch.mean((1 - torch.pow(torch.sum(one_hot * clipped_probabilities, dim=1), self.alpha)) / self.alpha)

        return gce_loss
