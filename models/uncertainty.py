"""
Uncertainty quantification module for scientific predictions.

This module implements various uncertainty quantification methods, including:
1. Heteroscedastic aleatoric uncertainty (data uncertainty)
2. Monte Carlo Dropout for epistemic uncertainty (model uncertainty)
3. Deep Ensembles for combined uncertainty estimation
4. Calibration metrics and recalibration techniques
"""
import logging
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

logger = logging.getLogger(__name__)


class UncertaintyLoss:
    """
    Uncertainty-aware loss functions for training models.
    """
    def __init__(
        self,
        base_loss_fn: str = "mse",
        beta: float = 1.0,
        reduction: str = "mean"
    ):
        """
        Initialize the uncertainty loss.
        
        Args:
            base_loss_fn: Base loss function type ('mse', 'mae', 'nll', etc.)
            beta: Weight for the uncertainty regularization term
            reduction: How to reduce the loss ('mean', 'sum', 'none')
        """
        self.base_loss_fn = base_loss_fn
        self.beta = beta
        self.reduction = reduction
        
    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate uncertainty-aware loss.
        
        Args:
            outputs: Model outputs including predictions and uncertainties
            targets: Target values
            
        Returns:
            Loss value
        """
        # Extract predictions and targets
        mean = outputs["mean"]
        var = outputs["var"]
        target = targets["target"]
        
        if self.base_loss_fn == "mse":
            # Uncertainty-weighted MSE loss
            precision = 1.0 / (var + 1e-8)
            loss = precision * (mean - target)**2 + torch.log(var + 1e-8)
            loss = 0.5 * loss
        
        elif self.base_loss_fn == "nll":
            # Negative log-likelihood loss with Gaussian assumption
            dist = Normal(mean, torch.sqrt(var + 1e-8))
            loss = -dist.log_prob(target)
        
        elif self.base_loss_fn == "laplace":
            # Laplace negative log-likelihood
            b = var  # Using var as scale parameter
            loss = torch.abs(mean - target) / (b + 1e-8) + torch.log(2 * (b + 1e-8))
        
        else:
            raise ValueError(f"Unsupported base loss function: {self.base_loss_fn}")
        
        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout layer for epistemic uncertainty estimation.
    """
    def __init__(self, p: float = 0.1, activate_in_eval: bool = True):
        """
        Initialize MC Dropout layer.
        
        Args:
            p: Dropout probability
            activate_in_eval: Whether to activate in evaluation mode
        """
        super().__init__()
        self.p = p
        self.activate_in_eval = activate_in_eval
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Monte Carlo dropout.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if self.training or self.activate_in_eval:
            return F.dropout(x, p=self.p, training=True)
        else:
            return x


class UncertaintyMLP(nn.Module):
    """
    MLP with uncertainty quantification capabilities.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        dropout_rate: float = 0.1,
        uncertainty_type: str = "heteroscedastic",
        activation: str = "relu"
    ):
        """
        Initialize uncertainty-aware MLP.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout_rate: Dropout probability
            uncertainty_type: Type of uncertainty ('heteroscedastic', 'mc_dropout', 'ensemble')
            activation: Activation function ('relu', 'silu', 'gelu', etc.)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.uncertainty_type = uncertainty_type
        
        # Create activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(self.activation)
            
            if uncertainty_type == "mc_dropout":
                layers.append(MCDropout(p=dropout_rate))
            else:
                layers.append(nn.Dropout(p=dropout_rate))
                
            prev_dim = h_dim
        
        self.layers = nn.Sequential(*layers)
        
        # Output heads
        if uncertainty_type == "heteroscedastic":
            # Separate heads for mean and variance
            self.mean_head = nn.Linear(prev_dim, output_dim)
            self.var_head = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Softplus()  # Ensure positive variance
            )
        else:
            # Single head for mean prediction
            self.head = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty quantification.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with mean and variance predictions
        """
        # Process through shared layers
        features = self.layers(x)
        
        if self.uncertainty_type == "heteroscedastic":
            # Predict both mean and variance
            mean = self.mean_head(features)
            var = self.var_head(features) + 1e-6  # Add small constant for numerical stability
            
            return {"mean": mean, "var": var}
        else:
            # Return only mean with dummy variance
            mean = self.head(features)
            
            return {"mean": mean, "var": torch.ones_like(mean) * 1e-6}
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 30
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimation.
        
        Args:
            x: Input tensor
            num_samples: Number of MC samples (only used for mc_dropout)
            
        Returns:
            Dictionary with mean and variance predictions
        """
        if self.uncertainty_type == "heteroscedastic":
            # Direct prediction from the model
            with torch.no_grad():
                pred = self(x)
            return pred
            
        elif self.uncertainty_type == "mc_dropout":
            # Multiple forward passes with dropout
            means = []
            self.train()  # Activate dropout
            
            with torch.no_grad():
                for _ in range(num_samples):
                    pred = self(x)
                    means.append(pred["mean"])
                    
            self.eval()
            
            # Calculate mean and variance across samples
            means = torch.stack(means, dim=0)
            mean = means.mean(dim=0)
            var = means.var(dim=0)
            
            return {"mean": mean, "var": var}
            
        else:
            # Default behavior
            with torch.no_grad():
                pred = self(x)
            return pred


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0
    ):
        """
        Initialize Bayesian linear layer.
        
        Args:
            in_features: Input features
            out_features: Output features
            prior_mu: Mean of weight prior
            prior_sigma: Standard deviation of weight prior
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight means
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        
        # Weight variances (parametrized as log-variance)
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self.weight_mu.data.normal_(prior_mu, 0.1)
        self.bias_mu.data.normal_(prior_mu, 0.1)
        self.weight_rho.data.fill_(-5.0)
        self.bias_rho.data.fill_(-5.0)
        
        # Prior distribution
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)
        
        # Store sampled weights
        self.weight = None
        self.bias = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sampled weights.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Sample weights and biases
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        
        if self.training:
            # Generate random variables
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            
            # Reparameterization trick
            self.weight = self.weight_mu + weight_eps * weight_std
            self.bias = self.bias_mu + bias_eps * bias_std
        else:
            # Use mean weights during evaluation
            self.weight = self.weight_mu
            self.bias = self.bias_mu
        
        # Perform linear transformation
        return F.linear(x, self.weight, self.bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Calculate KL divergence between weight posterior and prior.
        
        Returns:
            KL divergence
        """
        # Weight KL
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        kl_weight = self._kl_normal(
            self.weight_mu, weight_std,
            self.prior_mu, self.prior_sigma
        )
        
        # Bias KL
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        kl_bias = self._kl_normal(
            self.bias_mu, bias_std,
            self.prior_mu, self.prior_sigma
        )
        
        return kl_weight + kl_bias
    
    def _kl_normal(
        self,
        mu1: torch.Tensor,
        sigma1: torch.Tensor,
        mu2: float,
        sigma2: float
    ) -> torch.Tensor:
        """Calculate KL divergence between two normal distributions."""
        sigma2_tensor = torch.tensor(sigma2, device=mu1.device)
        mu2_tensor = torch.tensor(mu2, device=mu1.device)
        
        kl = (torch.log(sigma2_tensor) - torch.log(sigma1) + 
              (sigma1**2 + (mu1 - mu2_tensor)**2) / (2 * sigma2_tensor**2) - 0.5)
        return kl.sum()


class BayesianMLP(nn.Module):
    """
    Bayesian MLP with weight uncertainty.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        activation: str = "relu"
    ):
        """
        Initialize Bayesian MLP.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            prior_mu: Mean of weight prior
            prior_sigma: Standard deviation of weight prior
            activation: Activation function ('relu', 'silu', 'gelu', etc.)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Create Bayesian layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            self.layers.append(BayesianLinear(
                prev_dim, h_dim,
                prior_mu=prior_mu, prior_sigma=prior_sigma
            ))
            prev_dim = h_dim
            
        # Output layer
        self.output_layer = BayesianLinear(
            prev_dim, output_dim,
            prior_mu=prior_mu, prior_sigma=prior_sigma
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with mean prediction
        """
        # Process through Bayesian layers
        features = x
        for layer in self.layers:
            features = layer(features)
            features = self.activation(features)
            
        # Output layer
        output = self.output_layer(features)
        
        return {"mean": output, "var": torch.ones_like(output) * 1e-6}
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Calculate total KL divergence for all layers.
        
        Returns:
            Total KL divergence
        """
        kl = 0.0
        for layer in self.layers:
            kl += layer.kl_divergence()
        kl += self.output_layer.kl_divergence()
        return kl
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 30
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimation.
        
        Args:
            x: Input tensor
            num_samples: Number of MC samples
            
        Returns:
            Dictionary with mean and variance predictions
        """
        # Multiple forward passes
        means = []
        self.train()  # Enable sampling
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self(x)
                means.append(pred["mean"])
                
        self.eval()
        
        # Calculate mean and variance across samples
        means = torch.stack(means, dim=0)
        mean = means.mean(dim=0)
        var = means.var(dim=0)
        
        return {"mean": mean, "var": var}


def calibration_error(
    mean: torch.Tensor,
    var: torch.Tensor,
    target: torch.Tensor,
    num_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate expected calibration error.
    
    Args:
        mean: Predicted means
        var: Predicted variances
        target: Target values
        num_bins: Number of bins for calibration
        
    Returns:
        Tuple of (calibration error, confidence levels, observed frequencies)
    """
    # Convert to numpy
    if isinstance(mean, torch.Tensor):
        mean = mean.detach().cpu().numpy()
    if isinstance(var, torch.Tensor):
        var = var.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    # Calculate standardized error
    std = np.sqrt(var)
    error = np.abs(mean - target) / (std + 1e-8)
    
    # Create bins
    bin_edges = np.linspace(0, 3, num_bins + 1)  # 0 to 3 standard deviations
    bin_indices = np.digitize(error, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    # Calculate expected fraction (from normal distribution)
    expected_fractions = []
    for i in range(num_bins):
        if i == num_bins - 1:
            expected_frac = 1.0 - (2 * (1 - scipy.stats.norm.cdf(bin_edges[i])))
        else:
            expected_frac = 2 * (scipy.stats.norm.cdf(bin_edges[i+1]) - scipy.stats.norm.cdf(bin_edges[i]))
        expected_fractions.append(expected_frac)
    
    # Calculate observed fractions
    observed_fractions = []
    for i in range(num_bins):
        bin_mask = (bin_indices == i)
        if np.sum(bin_mask) == 0:
            observed_fractions.append(0.0)
        else:
            observed_fractions.append(np.mean(bin_mask))
    
    # Calculate calibration error
    cal_error = np.mean(np.abs(np.array(expected_fractions) - np.array(observed_fractions)))
    
    return cal_error, np.array(expected_fractions), np.array(observed_fractions)


def recalibrate_uncertainty(
    mean: torch.Tensor,
    var: torch.Tensor,
    scaling_factor: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recalibrate uncertainty predictions.
    
    Args:
        mean: Predicted means
        var: Predicted variances
        scaling_factor: Scaling factor for variance
        
    Returns:
        Tuple of (mean, recalibrated variance)
    """
    return mean, var * scaling_factor


def train_with_uncertainty(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    lr: float = 1e-3,
    epochs: int = 100,
    beta: float = 1.0,
    kl_weight: float = 1e-3,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    early_stopping_patience: int = 10
) -> Dict[str, List[float]]:
    """
    Train model with uncertainty quantification.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        lr: Learning rate
        epochs: Number of epochs
        beta: Weight for uncertainty term
        kl_weight: Weight for KL divergence term (for Bayesian models)
        device: Device to use
        early_stopping_patience: Number of epochs to wait before early stopping
        
    Returns:
        Dictionary with training history
    """
    # Move model to device
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create uncertainty loss
    uncertainty_loss = UncertaintyLoss(beta=beta)
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [] if val_loader else None,
        "calibration_error": [] if val_loader else None
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            # Get batch
            inputs = batch["inputs"].to(device)
            targets = {"target": batch["targets"].to(device)}
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = uncertainty_loss(outputs, targets)
            
            # Add KL divergence term for Bayesian models
            if hasattr(model, "kl_divergence"):
                kl_div = model.kl_divergence()
                loss += kl_weight * kl_div
                
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record loss
            train_losses.append(loss.item())
            
        # Record average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        history["train_loss"].append(avg_train_loss)
        
        # Validation
        if val_loader:
            model.eval()
            val_losses = []
            all_means = []
            all_vars = []
            all_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    # Get batch
                    inputs = batch["inputs"].to(device)
                    targets = {"target": batch["targets"].to(device)}
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Calculate loss
                    loss = uncertainty_loss(outputs, targets)
                    
                    # Record values
                    val_losses.append(loss.item())
                    all_means.append(outputs["mean"])
                    all_vars.append(outputs["var"])
                    all_targets.append(targets["target"])
            
            # Record average validation loss
            avg_val_loss = sum(val_losses) / len(val_losses)
            history["val_loss"].append(avg_val_loss)
            
            # Calculate calibration error
            all_means = torch.cat(all_means, dim=0)
            all_vars = torch.cat(all_vars, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            cal_error, _, _ = calibration_error(all_means, all_vars, all_targets)
            history["calibration_error"].append(cal_error)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
        # Print progress
        if val_loader:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Cal Error: {cal_error:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
    
    return history


def generate_confidence_intervals(
    mean: torch.Tensor,
    var: torch.Tensor,
    confidence: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate confidence intervals for predictions.
    
    Args:
        mean: Predicted means
        var: Predicted variances
        confidence: Confidence level (e.g., 0.95 for 95% confidence)
        
    Returns:
        Tuple of (lower bound, upper bound)
    """
    # Calculate z-score for the given confidence level
    z_score = scipy.stats.norm.ppf((1 + confidence) / 2)
    
    # Calculate standard deviation
    std = torch.sqrt(var)
    
    # Calculate bounds
    lower_bound = mean - z_score * std
    upper_bound = mean + z_score * std
    
    return lower_bound, upper_bound