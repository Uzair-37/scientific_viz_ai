"""
Training utilities for scientific multi-modal models.

This module provides training functionality specialized for scientific domains,
including domain adaptation, uncertainty quantification, and evaluation metrics.
"""
import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .multimodal_model import MultiModalTransformer, DiffusionModel
from .discovery import PatternDiscovery
from .scientific_adapters.materials_science import MaterialsScienceAdapter
from .scientific_adapters.bioinformatics import BioinformaticsAdapter

logger = logging.getLogger(__name__)


class ScientificTrainer:
    """
    Trainer for scientific multi-modal models.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 10,
        use_amp: bool = True,
        uncertainty_aware: bool = True
    ):
        """
        Initialize the scientific trainer.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_interval: How often to log training progress
            use_amp: Whether to use automatic mixed precision
            uncertainty_aware: Whether to use uncertainty-aware training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.use_amp = use_amp and torch.cuda.is_available()
        self.uncertainty_aware = uncertainty_aware
        
        # Move model to device
        self.model.to(device)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": {},
            "val_metrics": {}
        }
        
    def train_epoch(
        self,
        epoch: int,
        loss_fn: Callable,
        metrics: Dict[str, Callable] = {}
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            loss_fn: Loss function
            metrics: Dictionary of metric functions
            
        Returns:
            Dictionary of average loss and metrics
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        metric_values = {name: 0.0 for name in metrics}
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(**batch)
                loss = loss_fn(outputs, batch)
            
            # Backward pass with AMP
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            batch_size = next(iter(batch.values())).shape[0] if isinstance(next(iter(batch.values())), torch.Tensor) else 1
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            for name, metric_fn in metrics.items():
                metric_values[name] += metric_fn(outputs, batch) * batch_size
            
            # Log progress
            if batch_idx % self.log_interval == 0:
                logger.info(f"Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] "
                           f"Loss: {loss.item():.6f}")
        
        # Compute averages
        avg_loss = total_loss / total_samples
        avg_metrics = {name: value / total_samples for name, value in metric_values.items()}
        
        # Update history
        self.history["train_loss"].append(avg_loss)
        for name, value in avg_metrics.items():
            if name not in self.history["train_metrics"]:
                self.history["train_metrics"][name] = []
            self.history["train_metrics"][name].append(value)
        
        # Log summary
        elapsed = time.time() - start_time
        logger.info(f"Train Epoch: {epoch} completed in {elapsed:.2f}s "
                   f"Avg loss: {avg_loss:.6f}")
        
        return {"loss": avg_loss, **avg_metrics}
    
    def validate(
        self,
        loss_fn: Callable,
        metrics: Dict[str, Callable] = {}
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            loss_fn: Loss function
            metrics: Dictionary of metric functions
            
        Returns:
            Dictionary of average loss and metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        metric_values = {name: 0.0 for name in metrics}
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = loss_fn(outputs, batch)
                
                # Update metrics
                batch_size = next(iter(batch.values())).shape[0] if isinstance(next(iter(batch.values())), torch.Tensor) else 1
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                for name, metric_fn in metrics.items():
                    metric_values[name] += metric_fn(outputs, batch) * batch_size
        
        # Compute averages
        avg_loss = total_loss / total_samples
        avg_metrics = {name: value / total_samples for name, value in metric_values.items()}
        
        # Update history
        self.history["val_loss"].append(avg_loss)
        for name, value in avg_metrics.items():
            if name not in self.history["val_metrics"]:
                self.history["val_metrics"][name] = []
            self.history["val_metrics"][name].append(value)
        
        # Log summary
        logger.info(f"Validation - Avg loss: {avg_loss:.6f}")
        
        return {"loss": avg_loss, **avg_metrics}
    
    def train(
        self,
        epochs: int,
        loss_fn: Callable,
        metrics: Dict[str, Callable] = {},
        early_stopping_patience: int = 10,
        save_best_only: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train
            loss_fn: Loss function
            metrics: Dictionary of metric functions
            early_stopping_patience: Patience for early stopping
            save_best_only: Whether to save only the best model
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch, loss_fn, metrics)
            
            # Validate
            val_metrics = self.validate(loss_fn, metrics)
            
            # Update learning rate
            self.scheduler.step(val_metrics["loss"])
            
            # Save checkpoint if improved
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                patience_counter = 0
                
                # Save model
                self.save_checkpoint(epoch, val_metrics, best=True)
                logger.info(f"Epoch {epoch}: New best model with val_loss {best_val_loss:.6f}")
            else:
                patience_counter += 1
                
                # Save model if not using save_best_only
                if not save_best_only:
                    self.save_checkpoint(epoch, val_metrics)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch} epochs. Best epoch was {best_epoch} "
                           f"with val_loss {best_val_loss:.6f}")
                break
        
        logger.info(f"Training finished. Best epoch was {best_epoch} with val_loss {best_val_loss:.6f}")
        
        return self.history
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        best: bool = False
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Validation metrics
            best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "history": self.history
        }
        
        if best:
            checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint["history"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
        
        return checkpoint
    
    def plot_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        
        # Plot metrics
        if self.history["train_metrics"] and self.history["val_metrics"]:
            plt.subplot(1, 2, 2)
            
            # Get first metric
            metric_name = next(iter(self.history["train_metrics"].keys()))
            
            plt.plot(self.history["train_metrics"][metric_name], label=f"Train {metric_name}")
            plt.plot(self.history["val_metrics"][metric_name], label=f"Val {metric_name}")
            plt.xlabel("Epoch")
            plt.ylabel(metric_name)
            plt.legend()
            plt.title(f"Training and Validation {metric_name}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved training history plot to {save_path}")
        
        plt.show()


class UncertaintyLoss:
    """
    Loss function for uncertainty-aware training.
    """
    def __init__(self, base_loss_fn: str = "mse", beta: float = 1.0):
        """
        Initialize the uncertainty loss.
        
        Args:
            base_loss_fn: Base loss function ("mse" or "nll")
            beta: Weight for the uncertainty regularization term
        """
        self.base_loss_fn = base_loss_fn
        self.beta = beta
        
    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute uncertainty-aware loss.
        
        Args:
            outputs: Model outputs including mean and variance predictions
            targets: Target values
            
        Returns:
            Loss value
        """
        # Extract predictions and targets
        if "prediction_mean" in outputs and "prediction_var" in outputs:
            mean = outputs["prediction_mean"]
            var = outputs["prediction_var"]
            target = targets["target"]
        elif "property_mean" in outputs and "property_var" in outputs:
            mean = outputs["property_mean"]
            var = outputs["property_var"]
            target = targets["target"]
        else:
            raise ValueError("Outputs must contain mean and variance predictions")
        
        # Compute loss based on the specified function
        if self.base_loss_fn == "mse":
            # Uncertainty-weighted MSE loss
            precision = 1.0 / (var + 1e-8)
            loss = (precision * (mean - target) ** 2 + torch.log(var + 1e-8)) / 2.0
            loss = loss.mean()
        elif self.base_loss_fn == "nll":
            # Negative log-likelihood for regression with heteroscedastic noise
            loss = torch.log(var + 1e-8) / 2.0 + (mean - target) ** 2 / (2.0 * var + 1e-8)
            loss = loss.mean()
        else:
            raise ValueError(f"Unsupported base loss function: {self.base_loss_fn}")
        
        return loss


class ScientificMetrics:
    """
    Metrics for scientific model evaluation.
    """
    @staticmethod
    def mean_absolute_error(
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute mean absolute error.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            MAE value
        """
        # Extract predictions and targets
        if "prediction_mean" in outputs:
            pred = outputs["prediction_mean"]
        elif "property_mean" in outputs:
            pred = outputs["property_mean"]
        elif "predictions" in outputs:
            pred = outputs["predictions"]
        elif "properties" in outputs:
            pred = outputs["properties"]
        else:
            raise ValueError("Outputs must contain predictions")
        
        target = targets["target"]
        
        # Compute MAE
        return torch.mean(torch.abs(pred - target))
    
    @staticmethod
    def root_mean_squared_error(
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute root mean squared error.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            RMSE value
        """
        # Extract predictions and targets
        if "prediction_mean" in outputs:
            pred = outputs["prediction_mean"]
        elif "property_mean" in outputs:
            pred = outputs["property_mean"]
        elif "predictions" in outputs:
            pred = outputs["predictions"]
        elif "properties" in outputs:
            pred = outputs["properties"]
        else:
            raise ValueError("Outputs must contain predictions")
        
        target = targets["target"]
        
        # Compute RMSE
        return torch.sqrt(torch.mean((pred - target) ** 2))
    
    @staticmethod
    def r2_score(
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute coefficient of determination (R²).
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            R² value
        """
        # Extract predictions and targets
        if "prediction_mean" in outputs:
            pred = outputs["prediction_mean"]
        elif "property_mean" in outputs:
            pred = outputs["property_mean"]
        elif "predictions" in outputs:
            pred = outputs["predictions"]
        elif "properties" in outputs:
            pred = outputs["properties"]
        else:
            raise ValueError("Outputs must contain predictions")
        
        target = targets["target"]
        
        # Compute R²
        target_mean = torch.mean(target, dim=0)
        ss_tot = torch.sum((target - target_mean) ** 2, dim=0)
        ss_res = torch.sum((target - pred) ** 2, dim=0)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        return torch.mean(r2)
    
    @staticmethod
    def calibration_error(
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        num_bins: int = 10
    ) -> torch.Tensor:
        """
        Compute calibration error for uncertainty estimates.
        
        Args:
            outputs: Model outputs including mean and variance
            targets: Target values
            num_bins: Number of bins for calibration curve
            
        Returns:
            Calibration error
        """
        # Extract predictions and targets
        if "prediction_mean" in outputs and "prediction_var" in outputs:
            mean = outputs["prediction_mean"]
            var = outputs["prediction_var"]
        elif "property_mean" in outputs and "property_var" in outputs:
            mean = outputs["property_mean"]
            var = outputs["property_var"]
        else:
            raise ValueError("Outputs must contain mean and variance predictions")
        
        target = targets["target"]
        
        # Compute standardized residuals
        # z = (target - mean) / sqrt(var)
        std = torch.sqrt(var + 1e-8)
        z = (target - mean) / std
        
        # Compute expected proportion in each bin
        expected_prop = 1.0 / num_bins
        
        # Compute actual proportions
        bins = torch.linspace(-3, 3, num_bins + 1, device=z.device)
        bin_indices = torch.bucketize(z, bins)
        
        # Count occurrences in each bin
        bin_counts = torch.zeros(num_bins, device=z.device)
        for i in range(num_bins):
            bin_counts[i] = torch.sum((bin_indices == i + 1).float())
        
        # Normalize to get proportions
        actual_prop = bin_counts / z.numel()
        
        # Compute calibration error (mean absolute difference from expected)
        cal_error = torch.mean(torch.abs(actual_prop - expected_prop))
        
        return cal_error


def create_scientific_trainer(
    model_type: str,
    model_config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 1e-4,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "./checkpoints",
    uncertainty_aware: bool = True
) -> ScientificTrainer:
    """
    Create a trainer for a scientific model.
    
    Args:
        model_type: Type of model to train (e.g., "multimodal", "diffusion", "materials", "bio")
        model_config: Configuration for the model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        learning_rate: Learning rate for optimizer
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        uncertainty_aware: Whether to use uncertainty-aware training
        
    Returns:
        Configured trainer
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model based on type
    if model_type == "multimodal":
        model = MultiModalTransformer(**model_config)
    elif model_type == "diffusion":
        model = DiffusionModel(**model_config)
    elif model_type == "materials":
        model = MaterialsScienceAdapter(**model_config)
    elif model_type == "bio":
        model = BioinformaticsAdapter(**model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create trainer
    trainer = ScientificTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        device=device,
        checkpoint_dir=checkpoint_dir,
        uncertainty_aware=uncertainty_aware
    )
    
    return trainer


def main():
    """
    Main function for testing the training utilities.
    """
    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s [%(levelname)s] %(message)s")
    
    logger.info("Testing scientific trainer")
    
    # This would be replaced with actual code
    # ...


if __name__ == "__main__":
    main()