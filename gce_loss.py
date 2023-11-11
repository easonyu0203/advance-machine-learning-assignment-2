import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the GCE loss.

        Args:
        output (torch.Tensor): Logits tensor from the model (N, C), where C is the number of classes.
        target (torch.Tensor): One-hot encoded target labels (N, C), where C is the number of classes.

        Returns:
        torch.Tensor: Computed GCE loss.
        """
        probabilities = F.softmax(output, dim=1)  # Apply softmax to convert logits to probabilities
        clipped_probabilities = torch.clamp(probabilities, self.epsilon, 1.0 - self.epsilon)
        loss = -torch.sum(target * torch.pow(clipped_probabilities, self.alpha)) / output.size(0)
        return loss
