"""
Finance domain adapter for time series forecasting with uncertainty quantification.

This module implements the FinanceAdapter class for financial time series analysis,
including forecasting, volatility estimation, and risk assessment with various
uncertainty quantification methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional, Any

from .base_adapter import BaseAdapter


class TimeSeriesEncoder(nn.Module):
    """Encoder for financial time series data."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 embed_dim: int,
                 num_layers: int = 2,
                 dropout_rate: float = 0.1):
        """
        Initialize the time series encoder.
        
        Args:
            input_dim: Number of features in the input time series
            hidden_dim: Hidden dimension of the LSTM
            embed_dim: Output embedding dimension
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Final projection to embedding space
        self.projection = nn.Linear(hidden_dim * 2, embed_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the time series encoder.
        
        Args:
            x: Input time series [batch_size, seq_len, input_dim]
            mask: Optional mask for padding [batch_size, seq_len]
            
        Returns:
            Encoded representation [batch_size, embed_dim]
        """
        # Apply LSTM
        outputs, (hidden, _) = self.lstm(x)  # outputs: [batch_size, seq_len, hidden_dim*2]
        
        # Apply attention mechanism
        attention_scores = self.attention(outputs).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]
        
        # Apply attention weights to get context vector
        context = torch.bmm(attention_weights, outputs).squeeze(1)  # [batch_size, hidden_dim*2]
        
        # Project to embedding space
        embedding = self.projection(self.dropout(context))  # [batch_size, embed_dim]
        
        return embedding


class FundamentalEncoder(nn.Module):
    """Encoder for financial fundamental data."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 embed_dim: int,
                 dropout_rate: float = 0.1):
        """
        Initialize the fundamental encoder.
        
        Args:
            input_dim: Number of features in the fundamental data
            hidden_dims: List of hidden dimensions for the MLP
            embed_dim: Output embedding dimension
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Final projection to embedding space
        layers.append(nn.Linear(prev_dim, embed_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the fundamental encoder.
        
        Args:
            x: Input fundamental data [batch_size, input_dim]
            
        Returns:
            Encoded representation [batch_size, embed_dim]
        """
        return self.mlp(x)


class ReturnForecaster(nn.Module):
    """Forecaster for financial returns with uncertainty quantification."""
    
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 horizon: int,
                 uncertainty_type: str = "none",
                 dropout_rate: float = 0.1):
        """
        Initialize the return forecaster.
        
        Args:
            embed_dim: Dimension of the input embedding
            hidden_dim: Hidden dimension of the forecaster
            output_dim: Number of asset returns to forecast
            horizon: Forecast horizon (number of time steps)
            uncertainty_type: Type of uncertainty quantification
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.uncertainty_type = uncertainty_type
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output heads
        if uncertainty_type == "heteroscedastic":
            # Mean prediction
            self.mean_head = nn.Linear(hidden_dim, output_dim * horizon)
            # Log variance prediction
            self.logvar_head = nn.Linear(hidden_dim, output_dim * horizon)
        else:
            # Single output head for point predictions
            self.forecast_head = nn.Linear(hidden_dim, output_dim * horizon)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the return forecaster.
        
        Args:
            x: Input embedding [batch_size, embed_dim]
            
        Returns:
            Forecasted returns [batch_size, horizon, output_dim] or
            Dict with mean and variance
        """
        shared_features = self.shared(x)
        
        if self.uncertainty_type == "heteroscedastic":
            # Predict mean and log variance
            mean = self.mean_head(shared_features)
            logvar = self.logvar_head(shared_features)
            
            # Reshape outputs to [batch_size, horizon, output_dim]
            mean = mean.view(-1, self.horizon, self.output_dim)
            logvar = logvar.view(-1, self.horizon, self.output_dim)
            
            # Convert log variance to variance
            var = torch.exp(logvar)
            
            return {"mean": mean, "var": var}
        else:
            # Point forecast
            forecast = self.forecast_head(shared_features)
            
            # Reshape to [batch_size, horizon, output_dim]
            forecast = forecast.view(-1, self.horizon, self.output_dim)
            
            return forecast


class VolatilityEstimator(nn.Module):
    """Volatility estimator for financial time series."""
    
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 horizon: int,
                 uncertainty_type: str = "none",
                 dropout_rate: float = 0.1):
        """
        Initialize the volatility estimator.
        
        Args:
            embed_dim: Dimension of the input embedding
            hidden_dim: Hidden dimension of the estimator
            output_dim: Number of assets for volatility estimation
            horizon: Forecast horizon (number of time steps)
            uncertainty_type: Type of uncertainty quantification
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.uncertainty_type = uncertainty_type
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output heads depend on uncertainty type
        if uncertainty_type == "heteroscedastic":
            # Mean volatility prediction
            self.mean_head = nn.Linear(hidden_dim, output_dim * horizon)
            # Log variance of volatility prediction
            self.logvar_head = nn.Linear(hidden_dim, output_dim * horizon)
        else:
            # Single output head for volatility point estimates
            self.vol_head = nn.Linear(hidden_dim, output_dim * horizon)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the volatility estimator.
        
        Args:
            x: Input embedding [batch_size, embed_dim]
            
        Returns:
            Estimated volatility [batch_size, horizon, output_dim] or
            Dict with mean and variance of volatility estimates
        """
        shared_features = self.shared(x)
        
        if self.uncertainty_type == "heteroscedastic":
            # Predict mean and log variance of volatility
            mean = self.mean_head(shared_features)
            logvar = self.logvar_head(shared_features)
            
            # Reshape outputs to [batch_size, horizon, output_dim]
            mean = mean.view(-1, self.horizon, self.output_dim)
            logvar = logvar.view(-1, self.horizon, self.output_dim)
            
            # Ensure volatility is positive with softplus
            mean = F.softplus(mean)
            
            # Convert log variance to variance
            var = torch.exp(logvar)
            
            return {"mean": mean, "var": var}
        else:
            # Point estimate of volatility
            volatility = self.vol_head(shared_features)
            
            # Reshape to [batch_size, horizon, output_dim]
            volatility = volatility.view(-1, self.horizon, self.output_dim)
            
            # Ensure volatility is positive with softplus
            volatility = F.softplus(volatility)
            
            return volatility


class FinanceAdapter(BaseAdapter):
    """Finance domain adapter for time series forecasting with uncertainty quantification."""
    
    def __init__(self,
                 time_series_dim: int,
                 fundamental_dim: int,
                 embed_dim: int = 256,
                 hidden_dim: int = 128,
                 output_dim: int = 1,
                 horizon: int = 5,
                 uncertainty_type: str = "none",
                 dropout_rate: float = 0.1,
                 activation: str = "relu"):
        """
        Initialize the finance adapter.
        
        Args:
            time_series_dim: Dimension of the time series input
            fundamental_dim: Dimension of the fundamental data input
            embed_dim: Dimension of the shared embedding space
            hidden_dim: Hidden dimension for components
            output_dim: Number of assets to forecast
            horizon: Forecast horizon (number of time steps)
            uncertainty_type: Type of uncertainty quantification
            dropout_rate: Dropout rate for regularization
            activation: Activation function to use
        """
        super().__init__(embed_dim, uncertainty_type, dropout_rate, activation)
        
        self.time_series_dim = time_series_dim
        self.fundamental_dim = fundamental_dim
        self.output_dim = output_dim
        self.horizon = horizon
        
        # Time series encoder
        self.time_series_encoder = TimeSeriesEncoder(
            input_dim=time_series_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            dropout_rate=dropout_rate
        )
        
        # Fundamental encoder (if fundamental data is provided)
        if fundamental_dim > 0:
            self.fundamental_encoder = FundamentalEncoder(
                input_dim=fundamental_dim,
                hidden_dims=[hidden_dim, hidden_dim],
                embed_dim=embed_dim,
                dropout_rate=dropout_rate
            )
            
            # Fusion layer for combining time series and fundamental embeddings
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            )
        
        # Return forecaster
        self.return_forecaster = ReturnForecaster(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            horizon=horizon,
            uncertainty_type=uncertainty_type,
            dropout_rate=dropout_rate
        )
        
        # Volatility estimator
        self.volatility_estimator = VolatilityEstimator(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            horizon=horizon,
            uncertainty_type=uncertainty_type,
            dropout_rate=dropout_rate
        )
    
    def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode finance inputs into the shared embedding space.
        
        Args:
            inputs: Dictionary containing:
                - 'time_series': Time series data [batch_size, seq_len, time_series_dim]
                - 'mask': Optional mask for time series [batch_size, seq_len]
                - 'fundamentals': Optional fundamental data [batch_size, fundamental_dim]
            
        Returns:
            Tensor in the shared embedding space [batch_size, embed_dim]
        """
        # Encode time series
        time_series = inputs["time_series"]
        mask = inputs.get("mask", None)
        time_series_embedding = self.time_series_encoder(time_series, mask)
        
        # If fundamentals are provided and we have a fundamental encoder
        if "fundamentals" in inputs and hasattr(self, "fundamental_encoder"):
            fundamentals = inputs["fundamentals"]
            fundamental_embedding = self.fundamental_encoder(fundamentals)
            
            # Concatenate and fuse the embeddings
            combined = torch.cat([time_series_embedding, fundamental_embedding], dim=1)
            embedding = self.fusion(combined)
        else:
            embedding = time_series_embedding
        
        return embedding
    
    def predict(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict financial outputs from the embedding.
        
        Args:
            embedding: Tensor in the shared embedding space [batch_size, embed_dim]
            
        Returns:
            Dictionary containing:
                - 'returns': Forecasted returns or dict with mean and variance
                - 'volatility': Estimated volatility or dict with mean and variance
        """
        # Get return forecasts
        returns = self.return_forecaster(embedding)
        
        # Get volatility estimates
        volatility = self.volatility_estimator(embedding)
        
        # Structure the output based on uncertainty type
        if self.uncertainty_type == "heteroscedastic":
            return {
                "returns_mean": returns["mean"],
                "returns_var": returns["var"],
                "volatility_mean": volatility["mean"],
                "volatility_var": volatility["var"]
            }
        else:
            return {
                "returns": returns,
                "volatility": volatility
            }
    
    def predict_with_uncertainty(self, 
                                inputs: Dict[str, torch.Tensor], 
                                num_samples: int = 30) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        
        The implementation depends on the uncertainty type:
        - heteroscedastic: direct variance prediction
        - mc_dropout: Monte Carlo dropout sampling
        - bayesian: Bayesian neural network sampling
        - ensemble: handled externally
        
        Args:
            inputs: Dictionary of input tensors
            num_samples: Number of samples for Monte Carlo methods
            
        Returns:
            Dictionary containing predictions and uncertainty estimates
        """
        if self.uncertainty_type == "heteroscedastic":
            # For heteroscedastic models, we already output mean and variance
            return self.forward(inputs)
        
        elif self.uncertainty_type == "mc_dropout":
            # For MC dropout, we need to sample with dropout enabled
            self.train()  # Enable dropout
            
            returns_samples = []
            volatility_samples = []
            
            for _ in range(num_samples):
                embedding = self.encode(inputs)
                predictions = self.predict(embedding)
                
                returns_samples.append(predictions["returns"])
                volatility_samples.append(predictions["volatility"])
            
            # Stack samples and compute statistics
            returns_samples = torch.stack(returns_samples, dim=0)
            volatility_samples = torch.stack(volatility_samples, dim=0)
            
            returns_mean = returns_samples.mean(dim=0)
            returns_var = returns_samples.var(dim=0)
            
            volatility_mean = volatility_samples.mean(dim=0)
            volatility_var = volatility_samples.var(dim=0)
            
            return {
                "returns_mean": returns_mean,
                "returns_var": returns_var,
                "volatility_mean": volatility_mean,
                "volatility_var": volatility_var
            }
        
        elif self.uncertainty_type == "bayesian":
            # For Bayesian neural networks, we sample from the posterior
            returns_samples = []
            volatility_samples = []
            
            for _ in range(num_samples):
                embedding = self.encode(inputs)
                predictions = self.predict(embedding)
                
                returns_samples.append(predictions["returns"])
                volatility_samples.append(predictions["volatility"])
            
            # Stack samples and compute statistics
            returns_samples = torch.stack(returns_samples, dim=0)
            volatility_samples = torch.stack(volatility_samples, dim=0)
            
            returns_mean = returns_samples.mean(dim=0)
            returns_var = returns_samples.var(dim=0)
            
            volatility_mean = volatility_samples.mean(dim=0)
            volatility_var = volatility_samples.var(dim=0)
            
            return {
                "returns_mean": returns_mean,
                "returns_var": returns_var,
                "volatility_mean": volatility_mean,
                "volatility_var": volatility_var
            }
        
        else:
            # For no uncertainty or ensemble (handled externally)
            predictions = self.forward(inputs)
            
            # Convert to mean/var format for consistency
            return {
                "returns_mean": predictions["returns"],
                "returns_var": torch.zeros_like(predictions["returns"]),
                "volatility_mean": predictions["volatility"],
                "volatility_var": torch.zeros_like(predictions["volatility"])
            }
    
    def get_metrics(self) -> List[str]:
        """
        Get the list of metrics that this adapter can compute.
        
        Returns:
            List of metric names
        """
        return [
            "MSE", "MAE", "RMSE", "MAPE", "Directional Accuracy", 
            "Sharpe Ratio", "Sortino Ratio", "Maximum Drawdown",
            "Calibration Error"
        ]
    
    def compute_loss(self, 
                    predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the loss for training.
        
        Args:
            predictions: Dictionary containing model predictions
            targets: Dictionary containing target values
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        loss_components = {}
        
        # Extract predictions and targets
        if self.uncertainty_type == "heteroscedastic":
            returns_mean = predictions["returns_mean"]
            returns_var = predictions["returns_var"]
            volatility_mean = predictions["volatility_mean"]
            volatility_var = predictions["volatility_var"]
            
            returns_target = targets["returns"]
            volatility_target = targets["volatility"]
            
            # Compute heteroscedastic losses
            returns_loss = self.heteroscedastic_loss(returns_mean, returns_var, returns_target)
            volatility_loss = self.heteroscedastic_loss(volatility_mean, volatility_var, volatility_target)
            
            loss_components["returns_loss"] = returns_loss
            loss_components["volatility_loss"] = volatility_loss
            
        else:
            returns_pred = predictions["returns"]
            volatility_pred = predictions["volatility"]
            
            returns_target = targets["returns"]
            volatility_target = targets["volatility"]
            
            # Compute MSE losses
            returns_loss = F.mse_loss(returns_pred, returns_target)
            volatility_loss = F.mse_loss(volatility_pred, volatility_target)
            
            loss_components["returns_loss"] = returns_loss
            loss_components["volatility_loss"] = volatility_loss
        
        # Add KL divergence term for Bayesian neural networks
        if self.uncertainty_type == "bayesian":
            kl_div = self.kl_divergence()
            loss_components["kl_div"] = kl_div
            total_loss = returns_loss + volatility_loss + 0.01 * kl_div  # Weight KL term
        else:
            total_loss = returns_loss + volatility_loss
        
        return total_loss, loss_components