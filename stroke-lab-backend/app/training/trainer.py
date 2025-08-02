"""
Training loop implementation for stroke classification models

Provides comprehensive training functionality with progress reporting,
early stopping, learning rate scheduling, and detailed metrics tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Callable, List, Tuple
import logging
import time
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from app.training.model import create_model, count_parameters, get_model_size_mb
from app.training.dataset import create_data_loaders
from app.core.config import get_settings, get_training_settings
from app.db.models import StrokeData

settings = get_settings()
training_settings = get_training_settings()
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    epoch_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    model_type: str = "cnn"
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Optimizer settings
    optimizer: str = "adam"  # adam, sgd, adamw
    momentum: float = 0.9  # for SGD
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # step, cosine, reduce_on_plateau
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_probability: float = 0.5
    
    # Regularization
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    
    # Model saving
    save_best_model: bool = True
    save_last_model: bool = True
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StrokeTrainer:
    """
    Comprehensive trainer for stroke classification models.
    
    Handles training, validation, progress reporting, and model management.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model_config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[TrainingMetrics], None]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            model_config: Model-specific configuration
            progress_callback: Callback function for progress updates
        """
        self.config = config
        self.model_config = model_config or {}
        self.progress_callback = progress_callback
        
        # Setup device
        self.device = self._setup_device()
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None  # For mixed precision
        
        # Metrics tracking
        self.training_history: List[TrainingMetrics] = []
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        self.early_stopping_counter = 0
        
        # Paths
        self.model_save_dir = settings.MODELS_DIR
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"StrokeTrainer initialized with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using MPS device")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device = torch.device(self.config.device)
            logger.info(f"Using specified device: {device}")
        
        return device
    
    def setup_model(self, input_dim: int, num_classes: int) -> nn.Module:
        """
        Setup and initialize the model.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            
        Returns:
            Initialized model
        """
        # Create model
        model_config = {
            "dropout_rate": self.config.dropout_rate,
            "use_batch_norm": self.config.use_batch_norm,
            **self.model_config
        }
        
        self.model = create_model(
            model_type=self.config.model_type,
            input_dim=input_dim,
            num_classes=num_classes,
            model_config=model_config
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup loss function
        self._setup_criterion()
        
        # Setup learning rate scheduler
        if self.config.use_scheduler:
            self._setup_scheduler()
        
        # Setup mixed precision
        if self.config.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")
        
        # Log model info
        num_params = count_parameters(self.model)
        model_size_mb = get_model_size_mb(self.model)
        logger.info(f"Model parameters: {num_params:,}")
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        
        return self.model
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        if self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        logger.info(f"Optimizer: {self.config.optimizer}")
    
    def _setup_criterion(self):
        """Setup loss function."""
        # Use cross-entropy loss for classification
        self.criterion = nn.CrossEntropyLoss()
        logger.info("Loss function: CrossEntropyLoss")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_patience,
                gamma=self.config.scheduler_factor
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler_type == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Monitor validation accuracy
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler_type}")
        
        logger.info(f"Learning rate scheduler: {self.config.scheduler_type}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(1, self.config.num_epochs + 1):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_accuracy = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_accuracy = self._validate_epoch(val_loader)
            
            # Update learning rate scheduler
            if self.scheduler:
                if self.config.scheduler_type == "reduce_on_plateau":
                    self.scheduler.step(val_accuracy)
                else:
                    self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Create metrics object
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                learning_rate=current_lr,
                epoch_time=epoch_time
            )
            
            # Store metrics
            self.training_history.append(metrics)
            
            # Progress callback
            if self.progress_callback:
                self.progress_callback(metrics)
            
            # Log progress
            logger.info(
                f"Epoch {epoch:3d}/{self.config.num_epochs} | "
                f"Train: Loss={train_loss:.4f}, Acc={train_accuracy:.3f} | "
                f"Val: Loss={val_loss:.4f}, Acc={val_accuracy:.3f} | "
                f"LR={current_lr:.6f} | Time={epoch_time:.1f}s"
            )
            
            # Save best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_model_state = self.model.state_dict().copy()
                self.early_stopping_counter = 0
                
                if self.config.save_best_model:
                    self._save_model("best")
                
                logger.info(f"New best validation accuracy: {val_accuracy:.4f}")
            else:
                self.early_stopping_counter += 1
            
            # Early stopping check
            if (self.config.use_early_stopping and 
                self.early_stopping_counter >= self.config.early_stopping_patience):
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Save checkpoint
            if (self.config.save_checkpoints and 
                epoch % self.config.checkpoint_frequency == 0):
                self._save_model(f"checkpoint_epoch_{epoch}")
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")
        
        # Save final model
        if self.config.save_last_model:
            self._save_model("final")
        
        # Evaluate on test set if provided
        test_results = None
        if test_loader:
            test_results = self._evaluate_model(test_loader)
            logger.info(f"Test accuracy: {test_results['accuracy']:.4f}")
        
        # Compile results
        results = {
            "training_completed": True,
            "total_epochs": len(self.training_history),
            "total_time_seconds": total_time,
            "best_val_accuracy": self.best_val_accuracy,
            "final_train_accuracy": self.training_history[-1].train_accuracy,
            "final_val_accuracy": self.training_history[-1].val_accuracy,
            "training_history": [m.to_dict() for m in self.training_history],
            "model_info": {
                "type": self.config.model_type,
                "parameters": count_parameters(self.model),
                "size_mb": get_model_size_mb(self.model)
            },
            "config": self.config.to_dict()
        }
        
        if test_results:
            results["test_results"] = test_results
        
        return results
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def _evaluate_model(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                total_loss += loss.item()
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            all_targets, all_predictions, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_loss": total_loss / len(test_loader),
            "per_class_metrics": {
                "precision": precision_per_class.tolist(),
                "recall": recall_per_class.tolist(),
                "f1_score": f1_per_class.tolist(),
                "support": support.tolist()
            },
            "confusion_matrix": cm.tolist()
        }
        
        return results
    
    def _save_model(self, suffix: str):
        """Save model checkpoint."""
        model_path = self.model_save_dir / f"stroke_model_{suffix}.pth"
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "model_config": self.model_config,
            "training_history": [m.to_dict() for m in self.training_history],
            "best_val_accuracy": self.best_val_accuracy
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, model_path)
        logger.info(f"Model saved: {model_path}")
    
    def load_model(self, model_path: Path, input_dim: int, num_classes: int):
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Setup model with saved config
        self.config = TrainingConfig(**checkpoint["config"])
        self.model_config = checkpoint["model_config"]
        
        self.setup_model(input_dim, num_classes)
        
        # Load state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Restore training state
        self.training_history = [
            TrainingMetrics(**m) for m in checkpoint["training_history"]
        ]
        self.best_val_accuracy = checkpoint["best_val_accuracy"]
        
        logger.info(f"Model loaded from: {model_path}")
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """Plot training history."""
        if not self.training_history:
            logger.warning("No training history to plot")
            return
        
        epochs = [m.epoch for m in self.training_history]
        train_losses = [m.train_loss for m in self.training_history]
        val_losses = [m.val_loss for m in self.training_history]
        train_accs = [m.train_accuracy for m in self.training_history]
        val_accs = [m.val_accuracy for m in self.training_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plots
        ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
        ax1.plot(epochs, val_losses, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plots
        ax2.plot(epochs, train_accs, label='Train Acc', color='blue')
        ax2.plot(epochs, val_accs, label='Val Acc', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        lrs = [m.learning_rate for m in self.training_history]
        ax3.plot(epochs, lrs, color='green')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Epoch time
        times = [m.epoch_time for m in self.training_history]
        ax4.plot(epochs, times, color='orange')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Epoch Training Time')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved: {save_path}")
        
        return fig


def train_stroke_model(
    stroke_data: List[StrokeData],
    config: TrainingConfig,
    model_config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[TrainingMetrics], None]] = None
) -> Dict[str, Any]:
    """
    High-level function to train a stroke classification model.
    
    Args:
        stroke_data: List of stroke data for training
        config: Training configuration
        model_config: Model-specific configuration
        progress_callback: Progress callback function
        
    Returns:
        Training results
    """
    # Create data loaders
    data_loaders = create_data_loaders(
        stroke_data=stroke_data,
        batch_size=config.batch_size,
        dataset_type="features",  # Use features for CNN/MLP models
        augment_train=config.use_augmentation
    )
    
    # Get dataset info
    dataset_info = data_loaders["info"]
    input_dim = dataset_info["feature_dim"]
    num_classes = dataset_info["num_classes"]
    
    logger.info(f"Dataset: {dataset_info['total_size']} samples, {num_classes} classes")
    
    # Create trainer
    trainer = StrokeTrainer(
        config=config,
        model_config=model_config,
        progress_callback=progress_callback
    )
    
    # Setup model
    trainer.setup_model(input_dim, num_classes)
    
    # Train model
    results = trainer.train(
        train_loader=data_loaders["train"],
        val_loader=data_loaders["val"],
        test_loader=data_loaders["test"]
    )
    
    # Add dataset info to results
    results["dataset_info"] = dataset_info
    
    return results


# Example usage
if __name__ == "__main__":
    # Example training configuration
    config = TrainingConfig(
        model_type="cnn",
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001,
        use_early_stopping=True,
        early_stopping_patience=10
    )
    
    # This would be used with actual stroke data
    logger.info("Training configuration created")
    logger.info(f"Config: {config.to_dict()}")
