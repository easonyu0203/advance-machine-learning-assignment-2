import torch.optim as optim
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Union
from torch.utils.data import DataLoader


def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: str) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (str): Device to run the training on.

    Returns:
        Tuple[float, float]: Average loss and accuracy for this training epoch.
    """
    model.train()
    total_loss, total_correct = 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        total_correct += pred.eq(target.view_as(pred)).sum().item()
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * total_correct / len(train_loader.dataset)
    return avg_loss, accuracy

def evaluate(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: str, metrics: Optional[Dict[str, nn.Module]] = None) -> Union[Dict[str, float], Tuple[float, float]]:
    """
    Evaluate the model and calculate specified metrics.

    Args:
        model (nn.Module): The neural network model.
        test_loader (DataLoader): DataLoader for testing data.
        criterion (nn.Module): Loss function.
        device (str): Device to run the evaluation on.
        metrics (Optional[Dict[str, nn.Module]]): Metrics to compute.

    Returns:
        Union[Dict[str, float], Tuple[float, float]]: Dictionary of calculated metrics, or tuple of average loss and accuracy.
    """
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    if metrics is not None:
        # Reset metrics at the start of each evaluation
        for metric in metrics.values():
            metric.reset()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            if metrics is not None:
                # Update metrics
                for metric in metrics.values():
                    metric(output, target)
            else:
                # Compute accuracy if metrics are not provided
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += target.size(0)

    avg_loss = total_loss / len(test_loader)
    if metrics is not None:
        metrics_results = {}
        for name, metric in metrics.items():
            computed_metric = metric.compute()
            if computed_metric.ndim == 0:  # Scalar value
                metrics_results[name] = computed_metric.item()
            else:  # Non-scalar tensor
                metrics_results[name] = {index: value.item() for index, value in enumerate(computed_metric)}
        return metrics_results
    else:
        accuracy = 100. * total_correct / total_samples
        return avg_loss, accuracy
