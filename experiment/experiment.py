from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from experiment.config import ExperimentConfig
from trainer.train_eval import train, evaluate
from typing import Dict
from torchmetrics import F1Score, Precision, Recall, Accuracy
import pandas as pd


class ModelTrainingExperiment:
    """Class to handle training and evaluation of a model."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        # Initialize metrics
        self.metrics = {
            "train": {"accuracy": Accuracy(task="multiclass", num_classes=3, average='weighted').to(self.config.device),
                      "precision": Precision(task="multiclass", num_classes=3, average='weighted').to(
                          self.config.device),
                      "recall": Recall(task="multiclass", num_classes=3, average='weighted').to(self.config.device),
                      "f1": F1Score(task="multiclass", num_classes=3, average='weighted').to(self.config.device)},
            "val": {"accuracy": Accuracy(task="multiclass", num_classes=3, average='weighted').to(self.config.device),
                    "precision": Precision(task="multiclass", num_classes=3, average='weighted').to(self.config.device),
                    "recall": Recall(task="multiclass", num_classes=3, average='weighted').to(self.config.device),
                    "f1": F1Score(task="multiclass", num_classes=3, average='weighted').to(self.config.device)},
            "test": {"accuracy": Accuracy(task="multiclass", num_classes=3, average='weighted').to(self.config.device),
                     "precision": Precision(task="multiclass", num_classes=3, average='weighted').to(
                         self.config.device),
                     "recall": Recall(task="multiclass", num_classes=3, average='weighted').to(self.config.device),
                     "f1": F1Score(task="multiclass", num_classes=3, average='weighted').to(self.config.device)}
        }
        self.metric_values = {key: {metric: [] for metric in self.metrics[key]} for key in ['train', 'test']}

    def get_writer_name(self, iteration: int) -> str:
        """Generate a unique name for TensorBoard writer based on trainer parameters and iteration."""
        model_name = self.config.model.__class__.__name__
        dataset_name = self.config.dataset_name
        criterion_name = self.config.criterion.__class__.__name__
        return f"{model_name}_{dataset_name}_{criterion_name}_iter_{iteration}"

    def run_experiment(self):
        for iteration in range(10):
            self.config.init()
            writer_name = self.get_writer_name(iteration)
            writer = SummaryWriter(log_dir=f"runs/{writer_name}")
            with tqdm(total=self.config.num_epochs, desc=f"Epoch (Iteration {iteration + 1})", unit="epoch") as pbar:
                for epoch in range(self.config.num_epochs):
                    train_loss, train_acc = train(self.config.model, self.config.train_dataloader,
                                                  self.config.criterion, self.config.optimizer, self.config.device)
                    val_loss, val_acc = evaluate(self.config.model, self.config.val_dataloader, self.config.criterion,
                                                 self.config.device)

                    for phase, dataloader in [("train", self.config.train_dataloader),
                                              ("val", self.config.val_dataloader),
                                              ("test", self.config.test_dataloader)]:
                        results = evaluate(self.config.model, dataloader, self.config.criterion, self.config.device,
                                           self.metrics[phase])
                        for metric, value in results.items():
                            writer.add_scalar(f'{metric.capitalize()}/{phase.capitalize()}', value, epoch)

                    pbar.set_postfix({
                        'Train Loss': f"{train_loss:.4f}",
                        'Train Acc': f"{train_acc:.2f}%",
                        'Val Loss': f"{val_loss:.4f}",
                        'Val Acc': f"{val_acc:.2f}%"
                    })
                    pbar.update()

            # Calculate and store metrics for each iteration
            for dataloader, phase in zip([self.config.train_all_dataloader, self.config.test_dataloader],
                                         ["train", "test"]):
                results = evaluate(self.config.model, dataloader, self.config.criterion, self.config.device,
                                   self.metrics[phase])
                for metric, value in results.items():
                    self.metric_values[phase][metric].append(value)

            writer.close()

    def get_metrics_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Create separate DataFrames for train, validation, and test metrics."""
        dataframes = {}

        for phase in self.metric_values:
            # Creating a DataFrame for each phase
            data = {metric: values for metric, values in self.metric_values[phase].items()}
            df = pd.DataFrame(data)
            dataframes[phase] = df

        return dataframes

    def calculate_mean_std_metrics(self) -> dict:
        """
        Calculate the mean and standard deviation for each metric in training and testing.

        Returns:
            dict: A dictionary with the mean and standard deviation for each metric.
        """
        mean_std_metrics = {}

        for phase in ['train', 'test']:
            metrics_data = self.metric_values[phase]
            mean_std_data = {}

            for metric, values in metrics_data.items():
                mean = pd.Series(values).mean()
                std = pd.Series(values).std()
                mean_std_data[metric] = {'mean': mean, 'std': std}

            mean_std_metrics[phase] = mean_std_data

        return mean_std_metrics