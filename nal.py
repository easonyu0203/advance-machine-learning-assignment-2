import torch
import torch.nn as nn

class NoiseAdaptationLayer(nn.Module):
    """
    A Noise Adaptation Layer to model and compensate for label noise in classification tasks.

    Attributes:
        transition_matrix (torch.Tensor): The transition matrix to model the noise pattern.
    """

    def __init__(self, num_classes: int, known_transition_matrix: Optional[torch.Tensor] = None):
        """
        Initializes the Noise Adaptation Layer.

        Args:
            num_classes (int): Number of classes in the dataset.
            known_transition_matrix (Optional[torch.Tensor]): A predefined transition matrix. 
                                                             If None, the matrix will be initialized as trainable.
        """
        super(NoiseAdaptationLayer, self).__init__()
        if known_transition_matrix is not None:
            self.transition_matrix = nn.Parameter(known_transition_matrix, requires_grad=False)
        else:
            # Initialize as identity matrix and make it trainable if the transition matrix is not known
            self.transition_matrix = nn.Parameter(torch.eye(num_classes), requires_grad=True)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Noise Adaptation Layer.

        Args:
            p (torch.Tensor): The output probabilities from the previous network layer.

        Returns:
            torch.Tensor: Noise-adapted prediction probabilities.
        """
        # Apply softmax to ensure the rows of the transition matrix sum to 1
        T_normalized = torch.softmax(self.transition_matrix, dim=1)
        return torch.matmul(T_normalized, p)



class ClassifierWithNAL(nn.Module):
    """
    A classifier integrated with a Noise Adaptation Layer.
    """
    def __init__(self, base_model: nn.Module, num_classes: int, transition_matrix: torch.Tensor = None):
        """
        Initialize the Classifier with NAL.

        Parameters:
        - base_model (nn.Module): The base classifier model.
        - num_classes (int): Number of classes.
        - transition_matrix (torch.Tensor, optional): Transition matrix for NAL.
        """
        super(ClassifierWithNAL, self).__init__()
        self.base_model = base_model
        self.nal = NoiseAdaptationLayer(num_classes, transition_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.

        Parameters:
        - x (torch.Tensor): Input data.

        Returns:
        - torch.Tensor: Final predictions after noise adaptation.
        """
        x = self.base_model(x)
        p = torch.softmax(x, dim=1)
        return self.nal(p)
