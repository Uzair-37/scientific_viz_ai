"""
Materials science domain adapter for multi-modal scientific analysis.

This module provides specialized components for processing materials science data,
including crystal structures, spectroscopy data, and property predictions.
"""
import logging
import math
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class CrystalStructureEncoder(nn.Module):
    """
    Encoder for crystal structure data (atoms, positions, and lattice).
    """
    def __init__(
        self,
        atom_embedding_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 3,
        use_attention: bool = True,
        max_atoms: int = 100
    ):
        """
        Initialize the crystal structure encoder.
        
        Args:
            atom_embedding_dim: Embedding dimension for atom types
            hidden_dim: Hidden dimension for the network
            output_dim: Output embedding dimension
            num_layers: Number of interaction layers
            use_attention: Whether to use attention mechanism
            max_atoms: Maximum number of atoms to consider
        """
        super().__init__()
        self.atom_embedding_dim = atom_embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_atoms = max_atoms
        
        # Atom type embedding (atomic number to vector)
        self.atom_embedding = nn.Embedding(119, atom_embedding_dim)  # 119 for up to element 118 (Og)
        
        # Position encoder
        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Lattice encoder (unit cell parameters)
        self.lattice_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),  # 6 parameters: a, b, c, alpha, beta, gamma
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Interaction layers
        self.interaction_layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_attention:
                # Self-attention based interaction
                self.interaction_layers.append(
                    nn.MultiheadAttention(
                        embed_dim=hidden_dim,
                        num_heads=4,
                        batch_first=True
                    )
                )
            else:
                # MLP-based interaction
                self.interaction_layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.SiLU(),
                        nn.Linear(hidden_dim * 2, hidden_dim)
                    )
                )
                
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(
        self,
        atom_types: torch.Tensor,  # [batch_size, num_atoms]
        positions: torch.Tensor,   # [batch_size, num_atoms, 3]
        lattice: torch.Tensor,     # [batch_size, 6]
        mask: Optional[torch.Tensor] = None  # [batch_size, num_atoms] - mask for padding
    ) -> torch.Tensor:
        """
        Encode crystal structure data.
        
        Args:
            atom_types: Atomic numbers of atoms
            positions: Cartesian coordinates of atoms
            lattice: Unit cell parameters [a, b, c, alpha, beta, gamma]
            mask: Mask for padding (1 for atom, 0 for padding)
            
        Returns:
            Encoded representation [batch_size, output_dim]
        """
        batch_size, num_atoms = atom_types.shape
        
        # Create mask if not provided
        if mask is None:
            mask = (atom_types > 0).float()  # 0 is used for padding
        
        # Embed atom types
        atom_features = self.atom_embedding(atom_types)  # [batch_size, num_atoms, atom_embedding_dim]
        
        # Encode positions
        position_features = self.position_encoder(positions)  # [batch_size, num_atoms, hidden_dim]
        
        # Encode lattice
        lattice_features = self.lattice_encoder(lattice)  # [batch_size, hidden_dim]
        lattice_features = lattice_features.unsqueeze(1).expand(-1, num_atoms, -1)  # [batch_size, num_atoms, hidden_dim]
        
        # Combine features
        features = position_features + lattice_features  # [batch_size, num_atoms, hidden_dim]
        
        # Apply interaction layers
        for layer in self.interaction_layers:
            if isinstance(layer, nn.MultiheadAttention):
                # Apply attention with masking for padding
                attn_mask = mask.unsqueeze(1).expand(-1, num_atoms, -1) * mask.unsqueeze(2).expand(-1, -1, num_atoms)
                attn_mask = attn_mask.bool()
                attn_output, _ = layer(
                    query=features,
                    key=features,
                    value=features,
                    key_padding_mask=~mask.bool()  # False for atom, True for padding
                )
                features = features + attn_output
            else:
                # Apply MLP
                features = features + layer(features)
        
        # Global pooling (sum over atoms, weighted by mask)
        mask_expanded = mask.unsqueeze(2).expand(-1, -1, self.hidden_dim)
        pooled_features = (features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        
        # Final projection
        output = self.projection(pooled_features)
        normalized = self.layer_norm(output)
        
        return normalized


class SpectroscopyEncoder(nn.Module):
    """
    Encoder for spectroscopy data (XRD, XPS, Raman, IR, etc.).
    """
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 3,
        use_attention: bool = True
    ):
        """
        Initialize the spectroscopy data encoder.
        
        Args:
            input_dim: Input dimension (length of spectrum)
            hidden_dim: Hidden dimension for the network
            output_dim: Output embedding dimension
            num_layers: Number of layers in the network
            use_attention: Whether to use self-attention for long-range correlations
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initial projection
        self.initial_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Attention-based processing if requested
        self.use_attention = use_attention
        if use_attention:
            self.positional_encoding = PositionalEncoding(hidden_dim, max_len=input_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers
            )
        else:
            # Standard MLP layers
            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.SiLU(),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.SiLU()
                    )
                )
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode spectroscopy data.
        
        Args:
            x: Input spectrum [batch_size, input_dim]
            
        Returns:
            Encoded representation [batch_size, output_dim]
        """
        # Reshape input to ensure 2D, handling both single spectra and batches
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, input_dim]
            
        # Project to hidden dimension
        h = self.initial_projection(x)  # [batch_size, hidden_dim]
        
        if self.use_attention:
            # Reshape for transformer if needed (it expects a sequence)
            if h.dim() == 2:
                h = h.unsqueeze(1)  # [batch_size, 1, hidden_dim]
                
            # Add positional encoding
            h = self.positional_encoding(h)
            
            # Apply transformer
            h = self.transformer(h)
            
            # Global pooling if needed
            if h.dim() == 3:
                h = h.mean(dim=1)  # [batch_size, hidden_dim]
        else:
            # Apply MLP layers
            for layer in self.layers:
                h = layer(h) + h  # Residual connection
        
        # Final projection
        output = self.projection(h)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-based models.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return x


class MaterialPropertyPredictor(nn.Module):
    """
    Predictor for material properties from encoded representations.
    Supports multiple uncertainty quantification methods.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.1,
        uncertainty_type: str = "heteroscedastic",
        activation: str = "silu"
    ):
        """
        Initialize the property predictor.
        
        Args:
            input_dim: Input dimension (size of encoded representation)
            hidden_dim: Hidden dimension for the network
            output_dim: Number of properties to predict
            num_layers: Number of layers in the network
            dropout: Dropout rate
            uncertainty_type: Type of uncertainty to model
                - "none": No uncertainty modeling
                - "heteroscedastic": Aleatoric uncertainty with separate variance head
                - "mc_dropout": Epistemic uncertainty with Monte Carlo dropout
                - "bayesian": Epistemic uncertainty with Bayesian neural network
            activation: Activation function to use ('relu', 'silu', 'gelu')
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.uncertainty_type = uncertainty_type
        
        # Import uncertainty components
        from models.uncertainty import UncertaintyMLP, MCDropout, BayesianMLP
        
        # Choose model based on uncertainty type
        if uncertainty_type == "bayesian":
            self.model = BayesianMLP(
                input_dim=input_dim,
                hidden_dims=[hidden_dim] * num_layers,
                output_dim=output_dim,
                prior_sigma=1.0,
                activation=activation
            )
        else:
            self.model = UncertaintyMLP(
                input_dim=input_dim,
                hidden_dims=[hidden_dim] * num_layers,
                output_dim=output_dim,
                dropout_rate=dropout,
                uncertainty_type=uncertainty_type if uncertainty_type != "none" else "heteroscedastic",
                activation=activation
            )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict material properties.
        
        Args:
            x: Encoded representation [batch_size, input_dim]
            
        Returns:
            Dictionary containing:
                - "mean": Mean predictions [batch_size, output_dim]
                - "var": Variance predictions [batch_size, output_dim] (if uncertainty is enabled)
        """
        return self.model(x)
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor,
        num_samples: int = 30
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimation.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            num_samples: Number of Monte Carlo samples (for MC dropout and Bayesian methods)
            
        Returns:
            Dictionary with mean and variance predictions
        """
        if hasattr(self.model, "predict_with_uncertainty"):
            return self.model.predict_with_uncertainty(x, num_samples)
        else:
            return self.model(x)
            
    def kl_divergence(self) -> torch.Tensor:
        """
        Calculate KL divergence term for Bayesian models.
        
        Returns:
            KL divergence loss term
        """
        if hasattr(self.model, "kl_divergence"):
            return self.model.kl_divergence()
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)


class MaterialsScienceAdapter(nn.Module):
    """
    Complete adapter for materials science domain with advanced uncertainty quantification.
    """
    def __init__(
        self,
        embed_dim: int = 512,
        crystal_hidden_dim: int = 256,
        spectral_hidden_dim: int = 256,
        property_hidden_dim: int = 256,
        num_properties: int = 5,
        uncertainty_type: str = "heteroscedastic",
        dropout_rate: float = 0.1,
        activation: str = "silu"
    ):
        """
        Initialize the materials science adapter.
        
        Args:
            embed_dim: Common embedding dimension
            crystal_hidden_dim: Hidden dimension for crystal structure encoder
            spectral_hidden_dim: Hidden dimension for spectroscopy encoder
            property_hidden_dim: Hidden dimension for property predictor
            num_properties: Number of material properties to predict
            uncertainty_type: Type of uncertainty quantification
                - "none": No uncertainty modeling
                - "heteroscedastic": Aleatoric uncertainty with separate variance head
                - "mc_dropout": Epistemic uncertainty with Monte Carlo dropout
                - "bayesian": Epistemic uncertainty with Bayesian neural network
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'silu', 'gelu')
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.uncertainty_type = uncertainty_type
        
        # Domain-specific encoders
        self.crystal_encoder = CrystalStructureEncoder(
            hidden_dim=crystal_hidden_dim,
            output_dim=embed_dim
        )
        
        self.spectral_encoder = SpectroscopyEncoder(
            hidden_dim=spectral_hidden_dim,
            output_dim=embed_dim
        )
        
        # Cross-attention for fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Property predictor with uncertainty quantification
        self.property_predictor = MaterialPropertyPredictor(
            input_dim=embed_dim,
            hidden_dim=property_hidden_dim,
            output_dim=num_properties,
            dropout=dropout_rate,
            uncertainty_type=uncertainty_type,
            activation=activation
        )
        
    def encode(
        self, 
        crystal_data: Optional[Dict[str, torch.Tensor]] = None,
        spectral_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode materials science data.
        
        Args:
            crystal_data: Dictionary with crystal structure data
                - atom_types: Atomic numbers [batch_size, num_atoms]
                - positions: Atom positions [batch_size, num_atoms, 3]
                - lattice: Unit cell parameters [batch_size, 6]
                - mask: Optional mask for padding [batch_size, num_atoms]
            spectral_data: Spectroscopy data [batch_size, input_dim]
            
        Returns:
            Encoded representation [batch_size, embed_dim]
        """
        encodings = []
        
        # Encode crystal structure if provided
        if crystal_data is not None:
            crystal_encoding = self.crystal_encoder(
                atom_types=crystal_data["atom_types"],
                positions=crystal_data["positions"],
                lattice=crystal_data["lattice"],
                mask=crystal_data.get("mask", None)
            )
            encodings.append(crystal_encoding)
            
        # Encode spectroscopy data if provided
        if spectral_data is not None:
            spectral_encoding = self.spectral_encoder(spectral_data)
            encodings.append(spectral_encoding)
            
        # If only one modality is provided, return its encoding
        if len(encodings) == 1:
            return encodings[0]
        
        # Fuse modalities using cross-attention if multiple are provided
        # Using first modality as query and second as key/value
        fused_encoding, _ = self.cross_attention(
            query=encodings[0].unsqueeze(1),
            key=encodings[1].unsqueeze(1),
            value=encodings[1].unsqueeze(1)
        )
        
        return fused_encoding.squeeze(1)
    
    def predict_properties(
        self, 
        encoding: torch.Tensor, 
        num_samples: int = 30
    ) -> Dict[str, torch.Tensor]:
        """
        Predict material properties from encoding with uncertainty quantification.
        
        Args:
            encoding: Encoded representation [batch_size, embed_dim]
            num_samples: Number of Monte Carlo samples (for MC dropout and Bayesian models)
            
        Returns:
            Dictionary with property predictions including uncertainty
        """
        if hasattr(self.property_predictor, "predict_with_uncertainty") and self.uncertainty_type != "none":
            # Use advanced uncertainty estimation
            return self.property_predictor.predict_with_uncertainty(encoding, num_samples)
        else:
            # Use standard forward pass
            return self.property_predictor(encoding)
    
    def forward(
        self,
        crystal_data: Optional[Dict[str, torch.Tensor]] = None,
        spectral_data: Optional[torch.Tensor] = None,
        predict_with_samples: bool = False,
        num_samples: int = 30
    ) -> Dict[str, Any]:
        """
        Full forward pass with uncertainty quantification.
        
        Args:
            crystal_data: Dictionary with crystal structure data
            spectral_data: Spectroscopy data
            predict_with_samples: Whether to use Monte Carlo sampling for predictions
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with encodings and property predictions
        """
        # Encode data
        encoding = self.encode(crystal_data, spectral_data)
        
        # Predict properties with uncertainty
        if predict_with_samples and self.uncertainty_type in ["mc_dropout", "bayesian"]:
            predictions = self.predict_properties(encoding, num_samples)
        else:
            predictions = self.property_predictor(encoding)
        
        # Prepare output dictionary
        result = {"encoding": encoding}
        result.update(predictions)
        
        return result
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Calculate KL divergence for Bayesian models.
        
        Returns:
            KL divergence loss term
        """
        if hasattr(self.property_predictor, "kl_divergence"):
            return self.property_predictor.kl_divergence()
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)