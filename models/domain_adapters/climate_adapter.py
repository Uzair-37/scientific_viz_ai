"""
Climate domain adapter for environmental modeling with uncertainty quantification.

This module implements the ClimateAdapter class for climate and environmental data analysis,
including temperature forecasting, CO2 prediction, and spatial-temporal modeling with
various uncertainty quantification methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional, Any

from .base_adapter import BaseAdapter


class SpatialTemporalEncoder(nn.Module):
    """Encoder for spatial-temporal climate data."""
    
    def __init__(self,
                 input_channels: int,
                 hidden_channels: List[int],
                 temporal_dim: int,
                 spatial_dim: Tuple[int, int],
                 embed_dim: int,
                 dropout_rate: float = 0.1):
        """
        Initialize the spatial-temporal encoder.
        
        Args:
            input_channels: Number of input channels (variables)
            hidden_channels: List of hidden channels for CNN layers
            temporal_dim: Length of time dimension
            spatial_dim: Tuple of (height, width) for spatial dimensions
            embed_dim: Output embedding dimension
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        
        # Convolutional layers for spatial features
        cnn_layers = []
        in_channels = input_channels
        
        for out_channels in hidden_channels:
            cnn_layers.extend([
                nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Global pooling to reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool3d((temporal_dim, 1, 1))
        
        # LSTM for temporal features
        self.lstm = nn.LSTM(
            input_size=hidden_channels[-1],
            hidden_size=hidden_channels[-1],
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # Final projection to embedding space
        self.projection = nn.Linear(hidden_channels[-1] * 2, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the spatial-temporal encoder.
        
        Args:
            x: Input climate data [batch_size, channels, time, height, width]
            
        Returns:
            Encoded representation [batch_size, embed_dim]
        """
        # Apply CNN layers
        x = self.cnn(x)  # [batch_size, hidden_channels[-1], time, height, width]
        
        # Global pooling
        x = self.pool(x)  # [batch_size, hidden_channels[-1], time, 1, 1]
        
        # Reshape for LSTM
        x = x.squeeze(-1).squeeze(-1)  # [batch_size, hidden_channels[-1], time]
        x = x.transpose(1, 2)  # [batch_size, time, hidden_channels[-1]]
        
        # Apply LSTM
        x, (hidden, _) = self.lstm(x)  # x: [batch_size, time, hidden_channels[-1]*2]
        
        # Get last time step
        x = x[:, -1, :]  # [batch_size, hidden_channels[-1]*2]
        
        # Project to embedding space
        x = self.projection(x)  # [batch_size, embed_dim]
        
        return x


class TimeSeriesEncoder(nn.Module):
    """Encoder for climate time series data."""
    
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


class MetadataEncoder(nn.Module):
    """Encoder for climate metadata and geographical information."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 embed_dim: int,
                 dropout_rate: float = 0.1):
        """
        Initialize the metadata encoder.
        
        Args:
            input_dim: Number of features in the metadata
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
        Forward pass through the metadata encoder.
        
        Args:
            x: Input metadata [batch_size, input_dim]
            
        Returns:
            Encoded representation [batch_size, embed_dim]
        """
        return self.mlp(x)


class TemperaturePredictor(nn.Module):
    """Predictor for temperature forecasting with uncertainty quantification."""
    
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 horizon: int,
                 uncertainty_type: str = "none",
                 dropout_rate: float = 0.1):
        """
        Initialize the temperature predictor.
        
        Args:
            embed_dim: Dimension of the input embedding
            hidden_dim: Hidden dimension of the predictor
            output_dim: Number of locations for temperature prediction
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
            # Mean prediction
            self.mean_head = nn.Linear(hidden_dim, output_dim * horizon)
            # Log variance prediction
            self.logvar_head = nn.Linear(hidden_dim, output_dim * horizon)
        else:
            # Single output head for point predictions
            self.forecast_head = nn.Linear(hidden_dim, output_dim * horizon)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the temperature predictor.
        
        Args:
            x: Input embedding [batch_size, embed_dim]
            
        Returns:
            Forecasted temperatures [batch_size, horizon, output_dim] or
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


class CO2Predictor(nn.Module):
    """Predictor for CO2 levels with uncertainty quantification."""
    
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 horizon: int,
                 uncertainty_type: str = "none",
                 dropout_rate: float = 0.1):
        """
        Initialize the CO2 predictor.
        
        Args:
            embed_dim: Dimension of the input embedding
            hidden_dim: Hidden dimension of the predictor
            output_dim: Number of locations for CO2 prediction
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
            # Mean prediction
            self.mean_head = nn.Linear(hidden_dim, output_dim * horizon)
            # Log variance prediction
            self.logvar_head = nn.Linear(hidden_dim, output_dim * horizon)
        else:
            # Single output head for point predictions
            self.forecast_head = nn.Linear(hidden_dim, output_dim * horizon)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the CO2 predictor.
        
        Args:
            x: Input embedding [batch_size, embed_dim]
            
        Returns:
            Forecasted CO2 levels [batch_size, horizon, output_dim] or
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
            
            # Ensure CO2 is positive
            mean = F.softplus(mean)
            
            # Convert log variance to variance
            var = torch.exp(logvar)
            
            return {"mean": mean, "var": var}
        else:
            # Point forecast
            forecast = self.forecast_head(shared_features)
            
            # Reshape to [batch_size, horizon, output_dim]
            forecast = forecast.view(-1, self.horizon, self.output_dim)
            
            # Ensure CO2 is positive
            forecast = F.softplus(forecast)
            
            return forecast


class ClimateAdapter(BaseAdapter):
    """Climate domain adapter for environmental modeling with uncertainty quantification."""
    
    def __init__(self,
                 input_channels: int = 5,
                 time_series_dim: int = 10,
                 metadata_dim: int = 8,
                 spatial_dim: Tuple[int, int] = (32, 32),
                 temporal_dim: int = 48,
                 embed_dim: int = 256,
                 hidden_dim: int = 128,
                 output_dim: int = 1,
                 horizon: int = 12,
                 uncertainty_type: str = "none",
                 dropout_rate: float = 0.1,
                 activation: str = "relu",
                 data_type: str = "gridded"):
        """
        Initialize the climate adapter.
        
        Args:
            input_channels: Number of input channels for spatial-temporal data
            time_series_dim: Dimension of the time series input
            metadata_dim: Dimension of the metadata input
            spatial_dim: Tuple of (height, width) for spatial dimensions
            temporal_dim: Length of time dimension
            embed_dim: Dimension of the shared embedding space
            hidden_dim: Hidden dimension for components
            output_dim: Number of locations or variables to predict
            horizon: Forecast horizon (number of time steps)
            uncertainty_type: Type of uncertainty quantification
            dropout_rate: Dropout rate for regularization
            activation: Activation function to use
            data_type: Type of climate data ('gridded' or 'station')
        """
        super().__init__(embed_dim, uncertainty_type, dropout_rate, activation)
        
        self.data_type = data_type
        self.output_dim = output_dim
        self.horizon = horizon
        
        # Different encoders based on data type
        if data_type == "gridded":
            # For gridded data (e.g., satellite, reanalysis)
            self.spatial_temporal_encoder = SpatialTemporalEncoder(
                input_channels=input_channels,
                hidden_channels=[32, 64, 128],
                temporal_dim=temporal_dim,
                spatial_dim=spatial_dim,
                embed_dim=embed_dim,
                dropout_rate=dropout_rate
            )
        else:
            # For station data (e.g., weather stations)
            self.time_series_encoder = TimeSeriesEncoder(
                input_dim=time_series_dim,
                hidden_dim=hidden_dim,
                embed_dim=embed_dim,
                dropout_rate=dropout_rate
            )
        
        # Metadata encoder for both data types
        self.metadata_encoder = MetadataEncoder(
            input_dim=metadata_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            embed_dim=embed_dim,
            dropout_rate=dropout_rate
        )
        
        # Fusion layer for combining embeddings
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            self.activation,
            nn.Dropout(dropout_rate)
        )
        
        # Temperature predictor
        self.temperature_predictor = TemperaturePredictor(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            horizon=horizon,
            uncertainty_type=uncertainty_type,
            dropout_rate=dropout_rate
        )
        
        # CO2 predictor
        self.co2_predictor = CO2Predictor(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            horizon=horizon,
            uncertainty_type=uncertainty_type,
            dropout_rate=dropout_rate
        )
    
    def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode climate inputs into the shared embedding space.
        
        Args:
            inputs: Dictionary containing inputs based on data_type:
                For gridded data:
                    - 'spatial_temporal': Gridded data [batch_size, channels, time, height, width]
                    - 'metadata': Metadata [batch_size, metadata_dim]
                For station data:
                    - 'time_series': Time series data [batch_size, seq_len, time_series_dim]
                    - 'mask': Optional mask for time series [batch_size, seq_len]
                    - 'metadata': Metadata [batch_size, metadata_dim]
            
        Returns:
            Tensor in the shared embedding space [batch_size, embed_dim]
        """
        # Encode based on data type
        if self.data_type == "gridded":
            # For gridded data
            spatial_temporal = inputs["spatial_temporal"]
            metadata = inputs["metadata"]
            
            # Encode spatial-temporal data
            spatial_temporal_embedding = self.spatial_temporal_encoder(spatial_temporal)
            
            # Encode metadata
            metadata_embedding = self.metadata_encoder(metadata)
            
            # Concatenate and fuse the embeddings
            combined = torch.cat([spatial_temporal_embedding, metadata_embedding], dim=1)
            embedding = self.fusion(combined)
            
        else:
            # For station data
            time_series = inputs["time_series"]
            mask = inputs.get("mask", None)
            metadata = inputs["metadata"]
            
            # Encode time series
            time_series_embedding = self.time_series_encoder(time_series, mask)
            
            # Encode metadata
            metadata_embedding = self.metadata_encoder(metadata)
            
            # Concatenate and fuse the embeddings
            combined = torch.cat([time_series_embedding, metadata_embedding], dim=1)
            embedding = self.fusion(combined)
        
        return embedding
    
    def predict(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict climate outputs from the embedding.
        
        Args:
            embedding: Tensor in the shared embedding space [batch_size, embed_dim]
            
        Returns:
            Dictionary containing:
                - 'temperature': Forecasted temperatures or dict with mean and variance
                - 'co2': Predicted CO2 levels or dict with mean and variance
        """
        # Get temperature forecasts
        temperature = self.temperature_predictor(embedding)
        
        # Get CO2 predictions
        co2 = self.co2_predictor(embedding)
        
        # Structure the output based on uncertainty type
        if self.uncertainty_type == "heteroscedastic":
            return {
                "temperature_mean": temperature["mean"],
                "temperature_var": temperature["var"],
                "co2_mean": co2["mean"],
                "co2_var": co2["var"]
            }
        else:
            return {
                "temperature": temperature,
                "co2": co2
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
            
            temperature_samples = []
            co2_samples = []
            
            for _ in range(num_samples):
                embedding = self.encode(inputs)
                predictions = self.predict(embedding)
                
                temperature_samples.append(predictions["temperature"])
                co2_samples.append(predictions["co2"])
            
            # Stack samples and compute statistics
            temperature_samples = torch.stack(temperature_samples, dim=0)
            co2_samples = torch.stack(co2_samples, dim=0)
            
            temperature_mean = temperature_samples.mean(dim=0)
            temperature_var = temperature_samples.var(dim=0)
            
            co2_mean = co2_samples.mean(dim=0)
            co2_var = co2_samples.var(dim=0)
            
            return {
                "temperature_mean": temperature_mean,
                "temperature_var": temperature_var,
                "co2_mean": co2_mean,
                "co2_var": co2_var
            }
        
        elif self.uncertainty_type == "bayesian":
            # For Bayesian neural networks, we sample from the posterior
            temperature_samples = []
            co2_samples = []
            
            for _ in range(num_samples):
                embedding = self.encode(inputs)
                predictions = self.predict(embedding)
                
                temperature_samples.append(predictions["temperature"])
                co2_samples.append(predictions["co2"])
            
            # Stack samples and compute statistics
            temperature_samples = torch.stack(temperature_samples, dim=0)
            co2_samples = torch.stack(co2_samples, dim=0)
            
            temperature_mean = temperature_samples.mean(dim=0)
            temperature_var = temperature_samples.var(dim=0)
            
            co2_mean = co2_samples.mean(dim=0)
            co2_var = co2_samples.var(dim=0)
            
            return {
                "temperature_mean": temperature_mean,
                "temperature_var": temperature_var,
                "co2_mean": co2_mean,
                "co2_var": co2_var
            }
        
        else:
            # For no uncertainty or ensemble (handled externally)
            predictions = self.forward(inputs)
            
            # Convert to mean/var format for consistency
            return {
                "temperature_mean": predictions["temperature"],
                "temperature_var": torch.zeros_like(predictions["temperature"]),
                "co2_mean": predictions["co2"],
                "co2_var": torch.zeros_like(predictions["co2"])
            }
    
    def get_metrics(self) -> List[str]:
        """
        Get the list of metrics that this adapter can compute.
        
        Returns:
            List of metric names
        """
        return [
            "MSE", "MAE", "RMSE", "MAPE", 
            "Spatial IoU", "Temporal Correlation",
            "Calibration Error", "Seasonal Skill Score"
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
            temperature_mean = predictions["temperature_mean"]
            temperature_var = predictions["temperature_var"]
            co2_mean = predictions["co2_mean"]
            co2_var = predictions["co2_var"]
            
            temperature_target = targets["temperature"]
            co2_target = targets["co2"]
            
            # Compute heteroscedastic losses
            temperature_loss = self.heteroscedastic_loss(temperature_mean, temperature_var, temperature_target)
            co2_loss = self.heteroscedastic_loss(co2_mean, co2_var, co2_target)
            
            loss_components["temperature_loss"] = temperature_loss
            loss_components["co2_loss"] = co2_loss
            
        else:
            temperature_pred = predictions["temperature"]
            co2_pred = predictions["co2"]
            
            temperature_target = targets["temperature"]
            co2_target = targets["co2"]
            
            # Compute MSE losses
            temperature_loss = F.mse_loss(temperature_pred, temperature_target)
            co2_loss = F.mse_loss(co2_pred, co2_target)
            
            loss_components["temperature_loss"] = temperature_loss
            loss_components["co2_loss"] = co2_loss
        
        # Add KL divergence term for Bayesian neural networks
        if self.uncertainty_type == "bayesian":
            kl_div = self.kl_divergence()
            loss_components["kl_div"] = kl_div
            total_loss = temperature_loss + co2_loss + 0.01 * kl_div  # Weight KL term
        else:
            total_loss = temperature_loss + co2_loss
        
        return total_loss, loss_components