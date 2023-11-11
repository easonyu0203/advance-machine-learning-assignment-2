import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models import get_model
from torch.utils.data import DataLoader, random_split
from data import create_dataset
from train_eval import train, evaluate
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from typing import Tuple, Dict, List
from torchmetrics import F1Score, Precision, Recall, Accuracy
import pandas as pd



class ExperimentConfig:
    """Class to hold and initialize experiment parameters."""
    def __init__(self, model_name: str, dataset_name: str, learning_rate: float, batch_size: int, num_epochs: int):
        self.device = "cpu"  # or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.train_all_dataset = create_dataset(dataset_name, is_train=True)
        self.train_all_dataloader = DataLoader(self.train_all_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = create_dataset(dataset_name, is_train=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
    def init(self):
        self.model = get_model(self.model_name).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.split_dataset()

    def split_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Split dataset into training and validation sets."""
        train_size = int(0.8 * len(self.train_all_dataset))
        val_size = len(self.train_all_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.train_all_dataset, [train_size, val_size])
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
class TrainEvalExperiment:
    """Class to handle training and evaluation of a model."""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        # Initialize metrics
        self.metrics = {
            "train": {"accuracy": Accuracy(task="multiclass", num_classes=3, average='weighted').to(self.config.device), "precision": Precision(task="multiclass", num_classes=3, average='weighted').to(self.config.device), "recall": Recall(task="multiclass", num_classes=3, average='weighted').to(self.config.device), "f1": F1Score(task="multiclass", num_classes=3, average='weighted').to(self.config.device)},  
            "val": {"accuracy": Accuracy(task="multiclass", num_classes=3, average='weighted').to(self.config.device), "precision": Precision(task="multiclass", num_classes=3, average='weighted').to(self.config.device), "recall": Recall(task="multiclass", num_classes=3, average='weighted').to(self.config.device), "f1": F1Score(task="multiclass", num_classes=3, average='weighted').to(self.config.device)},  
            "test": {"accuracy": Accuracy(task="multiclass", num_classes=3, average='weighted').to(self.config.device), "precision": Precision(task="multiclass", num_classes=3, average='weighted').to(self.config.device), "recall": Recall(task="multiclass", num_classes=3, average='weighted').to(self.config.device), "f1": F1Score(task="multiclass", num_classes=3, average='weighted').to(self.config.device)}  
        }
        self.metric_values = {key: {metric: [] for metric in self.metrics[key]} for key in ['train', 'test']}


    def get_writer_name(self, iteration: int) -> str:
        """Generate a unique name for TensorBoard writer based on experiment parameters and iteration."""
        model_name = self.config.model.__class__.__name__
        dataset_name = self.config.dataset_name
        criterion_name = self.config.criterion.__class__.__name__
        return f"{model_name}_{dataset_name}_{criterion_name}_iter_{iteration}"

    def run_experiment(self):
        for iteration in range(1):
            self.config.init()
            writer_name = self.get_writer_name(iteration)
            writer = SummaryWriter(log_dir=f"runs/{writer_name}")
            with tqdm(total=self.config.num_epochs, desc=f"Epoch (Iteration {iteration+1})", unit="epoch") as pbar:
                for epoch in range(self.config.num_epochs):
                    train_loss, train_acc = train(self.config.model, self.config.train_dataloader, self.config.criterion, self.config.optimizer, self.config.device)
                    val_loss, val_acc = evaluate(self.config.model, self.config.val_dataloader, self.config.criterion, self.config.device)
                    
                    for phase, dataloader in [("train", self.config.train_dataloader), ("val", self.config.val_dataloader), ("test", self.config.test_dataloader)]:
                        results = evaluate(self.config.model, dataloader, self.config.criterion, self.config.device, self.metrics[phase])
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
            for dataloader, phase in zip([self.config.train_all_dataloader, self.config.test_dataloader], ["train", "test"]):
                results = evaluate(self.config.model, dataloader, self.config.criterion, self.config.device, self.metrics[phase])
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