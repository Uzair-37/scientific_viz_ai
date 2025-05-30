"""
Training utilities for multi-modal scientific models.
"""
import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Trainer for multi-modal scientific models.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: Callable,
        device: torch.device,
        scheduler: Optional[Any] = None,
        checkpoint_dir: str = "./checkpoints",
        tensorboard_dir: str = "./logs",
        max_grad_norm: float = 1.0
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for training
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler
            checkpoint_dir: Directory to save checkpoints
            tensorboard_dir: Directory for TensorBoard logs
            max_grad_norm: Maximum gradient norm for gradient clipping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.max_grad_norm = max_grad_norm
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set up TensorBoard
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # Move model to device
        self.model.to(device)
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        
        start_time = time.time()
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            
            # Calculate loss
            loss = self.criterion(outputs, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Log batch metrics
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | "
                           f"Loss: {loss.item():.4f}")
                
                # Log to TensorBoard
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar("train/batch_loss", loss.item(), global_step)
        
        # Calculate epoch metrics
        epoch_loss /= len(self.train_loader)
        epoch_metrics["loss"] = epoch_loss
        
        # Update learning rate
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(epoch_loss)
            else:
                self.scheduler.step()
        
        # Log epoch metrics
        logger.info(f"Epoch {epoch} | Training Loss: {epoch_loss:.4f} | "
                   f"Time: {time.time() - start_time:.2f}s")
        
        self.writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
        
        return epoch_metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        val_metrics = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Calculate loss
                loss = self.criterion(outputs, batch)
                
                # Update metrics
                val_loss += loss.item()
        
        # Calculate validation metrics
        val_loss /= len(self.val_loader)
        val_metrics["loss"] = val_loss
        
        # Log validation metrics
        logger.info(f"Epoch {epoch} | Validation Loss: {val_loss:.4f}")
        
        self.writer.add_scalar("val/loss", val_loss, epoch)
        
        return val_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> str:
        """
        Save a model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
            
        Returns:
            Epoch number of the checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
        
        return checkpoint["epoch"]
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
        save_every: int = 1,
        resume_from: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait for improvement
            save_every: Save checkpoint every N epochs
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Dictionary of training and validation metrics
        """
        start_epoch = 0
        best_val_loss = float("inf")
        patience_counter = 0
        
        metrics_history = {
            "train_loss": [],
            "val_loss": []
        }
        
        # Resume from checkpoint if provided
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from) + 1
            logger.info(f"Resuming training from epoch {start_epoch}")
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            logger.info(f"Starting epoch {epoch}/{num_epochs}")
            
            # Train one epoch
            train_metrics = self.train_epoch(epoch)
            metrics_history["train_loss"].append(train_metrics["loss"])
            
            # Validate
            val_metrics = self.validate(epoch)
            metrics_history["val_loss"].append(val_metrics["loss"])
            
            # Check for improvement
            is_best = val_metrics["loss"] < best_val_loss
            
            if is_best:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch} after {patience_counter} epochs without improvement")
                break
        
        logger.info("Training completed")
        return metrics_history