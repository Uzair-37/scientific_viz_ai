"""
Base adapter for domain-specific models with uncertainty quantification.

This module defines the BaseAdapter abstract class that all domain-specific
adapters (Finance, Climate) must implement. It enforces a consistent interface
for training, prediction, and uncertainty quantification.
"""

import abc
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional, Any


class BaseAdapter(nn.Module, abc.ABC):
    """
    Abstract base class for all domain adapters.
    
    This class defines the interface that all domain-specific adapters must implement,
    ensuring consistent behavior across different domains while allowing for
    domain-specific customization.
    """
    
    def __init__(self, 
                 embed_dim: int,
                 uncertainty_type: str = "none",
                 dropout_rate: float = 0.1,
                 activation: str = "relu"):
        """
        Initialize the base adapter.
        
        Args:
            embed_dim: Dimension of the embedding space
            uncertainty_type: Type of uncertainty quantification to use
                ('none', 'heteroscedastic', 'mc_dropout', 'bayesian', 'ensemble')
            dropout_rate: Dropout rate for regularization and MC dropout
            activation: Activation function to use ('relu', 'gelu', 'silu')
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.uncertainty_type = uncertainty_type
        self.dropout_rate = dropout_rate
        
        # Set activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    @abc.abstractmethod
    def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode domain-specific inputs into the shared embedding space.
        
        Args:
            inputs: Dictionary of input tensors specific to the domain
            
        Returns:
            Tensor in the shared embedding space
        """
        pass
    
    @abc.abstractmethod
    def predict(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict domain-specific outputs from the embedding.
        
        Args:
            embedding: Tensor in the shared embedding space
            
        Returns:
            Dictionary of predictions specific to the domain
        """
        pass
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the adapter.
        
        Args:
            inputs: Dictionary of input tensors
            
        Returns:
            Dictionary of output tensors
        """
        embedding = self.encode(inputs)
        return self.predict(embedding)
    
    @abc.abstractmethod
    def predict_with_uncertainty(self, 
                                inputs: Dict[str, torch.Tensor], 
                                num_samples: int = 30) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            inputs: Dictionary of input tensors
            num_samples: Number of samples for Monte Carlo methods
            
        Returns:
            Dictionary containing predictions and uncertainty estimates
        """
        pass
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Calculate KL divergence for Bayesian methods.
        
        Returns:
            KL divergence as a tensor
        """
        # Default implementation for non-Bayesian methods
        return torch.tensor(0.0, device=next(self.parameters()).device)
    
    @abc.abstractmethod
    def get_metrics(self) -> List[str]:
        """
        Get the list of metrics that this adapter can compute.
        
        Returns:
            List of metric names
        """
        pass
    
    @abc.abstractmethod
    def compute_loss(self, 
                    predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the loss for training.
        
        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of target values
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        pass
    
    @staticmethod
    def heteroscedastic_loss(mean: torch.Tensor, 
                           var: torch.Tensor, 
                           target: torch.Tensor) -> torch.Tensor:
        """
        Compute heteroscedastic loss that accounts for uncertainty.
        
        Args:
            mean: Predicted mean values
            var: Predicted variance values
            target: Target values
            
        Returns:
            Loss value that accounts for uncertainty
        """
        # Gaussian negative log likelihood with variance prediction
        return torch.mean(0.5 * torch.log(var) + 0.5 * (target - mean)**2 / var)
    
    def get_uncertainty_type(self) -> str:
        """
        Get the type of uncertainty quantification being used.
        
        Returns:
            Uncertainty type as a string
        """
        return self.uncertainty_type