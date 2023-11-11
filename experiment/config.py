from typing import Optional

import torch
from torch import optim as optim, nn as nn
from torch.utils.data import DataLoader, random_split

from trainer.data import create_dataset
from loss_fn.gce_loss import GCELoss
from model.model import get_model


class ExperimentConfig:
    """Class to hold and initialize trainer parameters."""

    def __init__(self, model_name: str, dataset_name: str, learning_rate: float, loss_fn_name: str, batch_size: int,
                 num_epochs: int,
                 gce_alpha: Optional[float] = None,
                 nal_transition_matrix: Optional[torch.Tensor] = None
                 ):
        self.device = "cuda"  # or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.loss_fn_name = loss_fn_name
        self.gce_alpha = gce_alpha
        self.nal_transition_matrix = nal_transition_matrix
        self.train_all_dataset = create_dataset(dataset_name, is_train=True)
        self.train_all_dataloader = DataLoader(self.train_all_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = create_dataset(dataset_name, is_train=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def init(self):
        self.model = get_model(self.model_name, transition_matrix=self.nal_transition_matrix).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = self._get_loss_fn()
        self.split_dataset()

    def split_dataset(self):
        """Split dataset into training and validation sets."""
        train_size = int(0.8 * len(self.train_all_dataset))
        val_size = len(self.train_all_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.train_all_dataset, [train_size, val_size])
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def _get_loss_fn(self):
        if self.loss_fn_name.lower() == "ce":
            return nn.CrossEntropyLoss()
        if self.loss_fn_name.lower() == "gce":
            if self.gce_alpha is None:
                raise ValueError("GCE Loss requires alpha to be specified")
            return GCELoss(self.gce_alpha)

        raise ValueError("Unknown Loss function, should be CE or GCE")
