"""
Advanced multi-modal architecture for financial analysis and economic forecasting.

This module implements a novel neural architecture that combines multiple data modalities
(time series, text, fundamentals, network data) for financial analysis tasks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any


class TemporalAttention(nn.Module):
    """
    Multi-scale temporal attention mechanism for financial time series.
    
    This module captures patterns at different time scales (daily, weekly, monthly)
    and attends to the most relevant signals for the current prediction task.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 252,  # Typical trading days in a year
        time_scales: List[int] = [5, 21, 63]  # Day, week, month in trading days
    ):
        """
        Initialize the temporal attention module.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            time_scales: List of time scales to model (in days)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.time_scales = time_scales
        
        # Position encoding
        self.position_encoding = nn.Parameter(
            torch.zeros(1, max_seq_length, hidden_dim)
        )
        nn.init.trunc_normal_(self.position_encoding, std=0.02)
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Scale-specific attention blocks
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in time_scales
        ])
        
        # Scale weights (learnable)
        self.scale_weights = nn.Parameter(torch.ones(len(time_scales)) / len(time_scales))
        self.softmax = nn.Softmax(dim=0)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply temporal attention to input sequence.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            mask: Optional attention mask
            
        Returns:
            Tensor of shape (batch_size, seq_length, hidden_dim)
        """
        batch_size, seq_length, _ = x.shape
        
        # Apply input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        positions = self.position_encoding[:, :seq_length, :]
        x = x + positions
        
        # Apply attention at different time scales
        scale_outputs = []
        normalized_weights = self.softmax(self.scale_weights)
        
        for i, scale in enumerate(self.time_scales):
            # Create scale-specific attention mask for looking back `scale` days
            if mask is None:
                scale_mask = torch.zeros(seq_length, seq_length, device=x.device)
                for j in range(seq_length):
                    scale_mask[j, max(0, j - scale):j + 1] = 1
                scale_mask = scale_mask.bool()
            else:
                # Combine with provided mask
                scale_mask = mask & create_scale_mask(seq_length, scale, x.device)
            
            # Apply attention
            attended, _ = self.scale_attentions[i](
                query=x,
                key=x,
                value=x,
                attn_mask=scale_mask if mask is not None else None,
                key_padding_mask=None
            )
            
            scale_outputs.append(normalized_weights[i] * attended)
        
        # Combine outputs from different scales
        combined = sum(scale_outputs)
        
        # Apply output projection
        output = self.output_projection(combined)
        output = self.dropout(output)
        output = self.layer_norm(output + x)  # Residual connection
        
        return output


class FinancialTextEncoder(nn.Module):
    """
    Specialized text encoder for financial documents with domain-specific features.
    
    This module enhances a pre-trained language model with finance-specific
    capabilities like entity recognition, sentiment analysis, and numerical reasoning.
    """
    
    def __init__(
        self,
        base_model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 768,
        output_dim: int = 512,
        dropout: float = 0.1,
        max_length: int = 512,
        numerical_feature_dim: int = 64
    ):
        """
        Initialize the financial text encoder.
        
        Args:
            base_model_name: Name of the pre-trained language model
            hidden_dim: Dimension of hidden representations
            output_dim: Dimension of output embeddings
            dropout: Dropout rate
            max_length: Maximum sequence length
            numerical_feature_dim: Dimension for numerical feature extraction
        """
        super().__init__()
        self.output_dim = output_dim
        
        # Import is done here to avoid requiring transformers for the whole module
        from transformers import AutoModel, AutoConfig
        
        # Load pre-trained language model
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.transformer = AutoModel.from_pretrained(base_model_name)
        
        # Finance-specific entity embedding
        self.entity_embedding = nn.Embedding(100, hidden_dim)  # 100 entity types
        
        # Numerical feature extraction
        self.num_feature_extractor = nn.Sequential(
            nn.Linear(1, numerical_feature_dim),
            nn.ReLU(),
            nn.Linear(numerical_feature_dim, numerical_feature_dim)
        )
        
        # Entity-aware attention
        self.entity_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim + numerical_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_ids: Optional[torch.Tensor] = None,
        numerical_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode financial text.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding
            entity_ids: Optional entity type IDs for financial entities
            numerical_values: Optional extracted numerical values from text
            
        Returns:
            Encoded representation of shape (batch_size, output_dim)
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Process entity information if provided
        if entity_ids is not None:
            entity_embeds = self.entity_embedding(entity_ids)
            
            # Apply entity-aware attention
            attended_states, _ = self.entity_attention(
                query=hidden_states,
                key=entity_embeds,
                value=entity_embeds,
                key_padding_mask=(entity_ids == 0)
            )
            
            hidden_states = hidden_states + attended_states
        
        # Get sequence representation (mean pooling)
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.sum(mask_expanded, 1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        text_embedding = sum_embeddings / sum_mask
        
        # Process numerical features if provided
        if numerical_values is not None:
            num_features = self.num_feature_extractor(numerical_values.unsqueeze(-1))
            
            # Combine with text embedding
            combined = torch.cat([text_embedding, num_features], dim=1)
        else:
            # Use zero features if no numerical values provided
            batch_size = text_embedding.size(0)
            num_features = torch.zeros(batch_size, self.numerical_feature_dim, 
                                      device=text_embedding.device)
            combined = torch.cat([text_embedding, num_features], dim=1)
        
        # Project to output dimension
        output = self.projection(combined)
        
        return output


class FundamentalAnalysisModule(nn.Module):
    """
    Financial fundamental analysis module for processing structured financial data.
    
    This module processes company fundamentals (balance sheet, income statement, etc.)
    and extracts meaningful financial indicators and relationships.
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize the fundamental analysis module.
        
        Args:
            input_dims: Dictionary mapping data types to dimensions
            hidden_dim: Dimension of hidden representations
            output_dim: Dimension of output embeddings
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dims = input_dims
        
        # Feature extractors for different data types
        self.feature_extractors = nn.ModuleDict({
            data_type: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for data_type, dim in input_dims.items()
        })
        
        # Cross-feature attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Ratio computation module (learnable transformations of financial ratios)
        self.ratio_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Growth rate computation module
        self.growth_module = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, output_dim),  # 3 sources: features, ratios, growth
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        temporal_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Process financial fundamentals.
        
        Args:
            inputs: Dictionary mapping data types to input tensors
            temporal_data: Optional historical data for growth computation
            
        Returns:
            Encoded representation of shape (batch_size, output_dim)
        """
        # Extract features from each data type
        features = {}
        for data_type, tensor in inputs.items():
            if data_type in self.feature_extractors:
                features[data_type] = self.feature_extractors[data_type](tensor)
        
        # Combine features
        if len(features) > 0:
            feature_list = list(features.values())
            combined_features = torch.stack(feature_list, dim=1)  # [batch_size, num_types, hidden_dim]
            
            # Apply cross-feature attention
            attended_features, _ = self.cross_attention(
                query=combined_features,
                key=combined_features,
                value=combined_features
            )
            
            # Mean pooling across feature types
            feature_embedding = attended_features.mean(dim=1)  # [batch_size, hidden_dim]
            
            # Compute financial ratios (learnable combinations)
            ratio_embedding = self.ratio_module(feature_embedding)
            
            # Compute growth rates if temporal data is provided
            if temporal_data is not None:
                temporal_features = []
                for data_type, time_series in temporal_data.items():
                    if data_type in self.feature_extractors:
                        # Process each time step
                        batch_size, time_steps, feat_dim = time_series.shape
                        reshaped = time_series.reshape(batch_size * time_steps, feat_dim)
                        extracted = self.feature_extractors[data_type](reshaped)
                        extracted = extracted.reshape(batch_size, time_steps, hidden_dim)
                        temporal_features.append(extracted)
                
                if temporal_features:
                    # Combine temporal features and compute growth
                    temporal_combined = torch.stack(temporal_features, dim=2).mean(dim=2)
                    _, growth_embedding = self.growth_module(temporal_combined)
                    growth_embedding = growth_embedding.squeeze(0)  # [batch_size, hidden_dim]
                else:
                    # Default growth embedding if no valid temporal features
                    growth_embedding = torch.zeros_like(feature_embedding)
            else:
                # Default growth embedding if no temporal data
                growth_embedding = torch.zeros_like(feature_embedding)
            
            # Combine all embeddings
            combined = torch.cat([feature_embedding, ratio_embedding, growth_embedding], dim=1)
            output = self.projection(combined)
            
            return output
        else:
            # Return zeros if no valid inputs
            return torch.zeros(inputs[list(inputs.keys())[0]].size(0), self.output_dim)


class FinancialNetworkModule(nn.Module):
    """
    Financial network analysis module for processing relationship data.
    
    This module handles relationship networks like company supply chains,
    investment relationships, and industry connections.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize the financial network module.
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Dimension of hidden representations
            output_dim: Dimension of output embeddings
            num_layers: Number of message-passing layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Node feature embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Edge feature embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Global pooling
        self.global_attention_pool = GlobalAttentionPooling(hidden_dim)
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process financial network data.
        
        Args:
            node_features: Node feature tensor
            edge_index: Edge connectivity tensor
            edge_features: Optional edge feature tensor
            batch: Optional batch assignment for nodes
            
        Returns:
            Encoded representation of shape (batch_size, output_dim)
        """
        # Embed node features
        x = self.node_embedding(node_features)
        
        # Embed edge features if provided
        edge_attr = None
        if edge_features is not None:
            edge_attr = self.edge_embedding(edge_features)
        
        # Apply message passing layers
        for layer in self.message_layers:
            x = layer(x, edge_index, edge_attr)
        
        # Global pooling
        if batch is None:
            batch = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)
        
        pooled = self.global_attention_pool(x, batch)
        
        # Output projection
        output = self.projection(pooled)
        
        return output


class MessagePassingLayer(nn.Module):
    """
    Message passing layer for graph neural networks.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """Initialize the message passing layer."""
        super().__init__()
        
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply message passing."""
        # Extract source and target nodes
        src, dst = edge_index
        
        # Prepare messages
        src_features = x[src]
        dst_features = x[dst]
        
        if edge_attr is not None:
            # Include edge features in messages
            messages = torch.cat([src_features, dst_features, edge_attr], dim=1)
        else:
            # No edge features
            messages = torch.cat([src_features, dst_features], dim=1)
        
        # Compute message values
        messages = self.message_mlp(messages)
        
        # Aggregate messages
        msg_by_node = {}
        for i, node_idx in enumerate(dst.tolist()):
            if node_idx in msg_by_node:
                msg_by_node[node_idx].append(messages[i])
            else:
                msg_by_node[node_idx] = [messages[i]]
        
        # Update node representations
        new_x = x.clone()
        for node_idx, msgs in msg_by_node.items():
            agg_msg = torch.stack(msgs).mean(dim=0)
            new_x[node_idx] = self.update_gru(agg_msg.unsqueeze(0), x[node_idx].unsqueeze(0)).squeeze(0)
        
        # Apply normalization
        new_x = self.layer_norm(new_x)
        
        return new_x


class GlobalAttentionPooling(nn.Module):
    """
    Global attention pooling for graph-level representations.
    """
    
    def __init__(self, hidden_dim: int):
        """Initialize the global attention pooling."""
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Apply global attention pooling."""
        # Compute attention scores
        scores = self.attention(x)
        
        # Apply softmax over nodes in each graph
        batch_size = batch.max().item() + 1
        
        pooled_outputs = []
        for i in range(batch_size):
            # Get nodes for this graph
            graph_mask = (batch == i)
            graph_x = x[graph_mask]
            graph_scores = scores[graph_mask]
            
            # Apply softmax and weighted sum
            weights = F.softmax(graph_scores, dim=0)
            pooled = (graph_x * weights).sum(dim=0)
            pooled_outputs.append(pooled)
        
        # Combine outputs
        if pooled_outputs:
            return torch.stack(pooled_outputs)
        else:
            return torch.zeros(batch_size, x.size(1), device=x.device)


class UncertaintyAwareForecasting(nn.Module):
    """
    Uncertainty-aware forecasting module for financial predictions.
    
    This module provides calibrated uncertainty estimates for financial
    forecasts using quantile regression and ensemble techniques.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        forecast_steps: int = 5,
        num_quantiles: int = 9,
        dropout: float = 0.1,
        num_heads: int = 4
    ):
        """
        Initialize the uncertainty-aware forecasting module.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden representations
            forecast_steps: Number of steps to forecast
            num_quantiles: Number of quantiles to predict (including median)
            dropout: Dropout rate
            num_heads: Number of ensemble heads
        """
        super().__init__()
        self.forecast_steps = forecast_steps
        self.num_quantiles = num_quantiles
        self.num_heads = num_heads
        
        # Calculate quantile levels
        self.quantile_levels = torch.tensor(
            [i / (num_quantiles - 1) for i in range(num_quantiles)]
        )
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Ensemble heads for quantile regression
        self.forecast_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, forecast_steps * num_quantiles)
            )
            for _ in range(num_heads)
        ])
        
        # Calibration network
        self.calibration_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate probabilistic forecasts.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary with forecast results
        """
        batch_size = x.size(0)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Generate ensemble predictions
        ensemble_forecasts = []
        for head in self.forecast_heads:
            # Generate predictions for all steps and quantiles
            forecast = head(features)
            forecast = forecast.view(batch_size, self.forecast_steps, self.num_quantiles)
            ensemble_forecasts.append(forecast)
        
        # Stack ensemble forecasts
        ensemble_forecasts = torch.stack(ensemble_forecasts, dim=1)  # [batch, num_heads, steps, quantiles]
        
        # Generate calibration weights (attention over ensemble heads)
        calibration_weights = self.calibration_network(features)
        calibration_weights = F.softmax(calibration_weights, dim=1)  # [batch, num_heads]
        
        # Apply calibration weights
        calibration_weights = calibration_weights.unsqueeze(-1).unsqueeze(-1)  # [batch, num_heads, 1, 1]
        calibrated_forecast = (ensemble_forecasts * calibration_weights).sum(dim=1)  # [batch, steps, quantiles]
        
        # Extract prediction intervals
        median_forecast = calibrated_forecast[:, :, self.num_quantiles // 2]  # [batch, steps]
        
        lower_idx = 0
        upper_idx = self.num_quantiles - 1
        prediction_interval_90 = torch.stack(
            [calibrated_forecast[:, :, lower_idx], calibrated_forecast[:, :, upper_idx]],
            dim=-1
        )
        
        # Prepare return values
        return {
            "median_forecast": median_forecast,
            "calibrated_forecast": calibrated_forecast,
            "prediction_interval_90": prediction_interval_90,
            "ensemble_forecasts": ensemble_forecasts,
            "calibration_weights": calibration_weights.squeeze(-1).squeeze(-1)
        }


class FinancialMultiModalFusion(nn.Module):
    """
    Multi-modal fusion module for financial data analysis.
    
    This module combines representations from different financial data modalities
    (time series, text, fundamentals, network) for integrated analysis.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize the multi-modal fusion module.
        
        Args:
            modality_dims: Dictionary mapping modality names to dimensions
            hidden_dim: Dimension of hidden representations
            output_dim: Dimension of output embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.modality_dims = modality_dims
        
        # Modality-specific projections to common dimension
        self.projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for modality, dim in modality_dims.items()
        })
        
        # Cross-modal attention
        self.cross_attentions = nn.ModuleDict({
            f"{mod1}_to_{mod2}": nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for i, mod1 in enumerate(modality_dims.keys())
            for j, mod2 in enumerate(modality_dims.keys())
            if i != j
        })
        
        # Modality importance prediction
        self.modality_importance = nn.Sequential(
            nn.Linear(hidden_dim * len(modality_dims), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(modality_dims))
        )
        
        # Gated fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * len(modality_dims), hidden_dim),
            nn.Sigmoid()
        )
        
        self.fusion_content = nn.Sequential(
            nn.Linear(hidden_dim * len(modality_dims), hidden_dim),
            nn.Tanh()
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse multi-modal financial data.
        
        Args:
            inputs: Dictionary mapping modality names to input tensors
            
        Returns:
            Dictionary with fusion results
        """
        # Project each modality to common dimension
        projected = {}
        for modality, tensor in inputs.items():
            if modality in self.projections:
                projected[modality] = self.projections[modality](tensor)
        
        # Apply cross-modal attention
        attended = {modality: tensor.clone() for modality, tensor in projected.items()}
        
        for i, mod1 in enumerate(self.modality_dims.keys()):
            if mod1 not in projected:
                continue
                
            for j, mod2 in enumerate(self.modality_dims.keys()):
                if i == j or mod2 not in projected:
                    continue
                    
                key = f"{mod1}_to_{mod2}"
                if key in self.cross_attentions:
                    attention_output, _ = self.cross_attentions[key](
                        query=attended[mod1].unsqueeze(1),
                        key=projected[mod2].unsqueeze(1),
                        value=projected[mod2].unsqueeze(1)
                    )
                    attended[mod1] = attended[mod1] + attention_output.squeeze(1)
        
        # Concatenate attended representations
        if attended:
            modality_list = sorted(self.modality_dims.keys())
            modality_tensors = []
            
            for modality in modality_list:
                if modality in attended:
                    modality_tensors.append(attended[modality])
                else:
                    # Zero tensor for missing modalities
                    batch_size = next(iter(attended.values())).size(0)
                    modality_tensors.append(
                        torch.zeros(batch_size, hidden_dim, device=attended[list(attended.keys())[0]].device)
                    )
            
            concatenated = torch.cat(modality_tensors, dim=1)
            
            # Predict modality importance
            importance = self.modality_importance(concatenated)
            importance_weights = F.softmax(importance, dim=1)
            
            # Apply gated fusion
            gate = self.fusion_gate(concatenated)
            content = self.fusion_content(concatenated)
            fused = gate * content
            
            # Project to output dimension
            output = self.output_projection(fused)
            
            # Prepare return values
            return {
                "fused_representation": output,
                "modality_representations": attended,
                "modality_importance": importance_weights
            }
        else:
            # Handle case with no valid inputs
            return {
                "fused_representation": torch.zeros(1, output_dim),
                "modality_representations": {},
                "modality_importance": torch.zeros(1, len(self.modality_dims))
            }


class FinancialMultiTaskOutput(nn.Module):
    """
    Multi-task output module for financial predictions.
    
    This module generates predictions for multiple financial tasks
    (e.g., returns prediction, risk assessment, anomaly detection).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        task_config: Dict[str, Dict] = None,
        dropout: float = 0.1
    ):
        """
        Initialize the multi-task output module.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden representations
            task_config: Configuration for different prediction tasks
            dropout: Dropout rate
        """
        super().__init__()
        
        # Default task configuration if none provided
        if task_config is None:
            task_config = {
                "return_prediction": {"type": "regression", "output_dim": 1},
                "volatility_prediction": {"type": "regression", "output_dim": 1},
                "trend_classification": {"type": "classification", "output_dim": 3},
                "risk_assessment": {"type": "ordinal", "output_dim": 5},
                "anomaly_detection": {"type": "binary", "output_dim": 1}
            }
        
        self.task_config = task_config
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        for task_name, config in task_config.items():
            task_type = config["type"]
            output_dim = config["output_dim"]
            
            if task_type == "regression":
                # For regression tasks (e.g., return prediction)
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, output_dim)
                )
                
            elif task_type == "classification":
                # For classification tasks (e.g., trend classification)
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, output_dim)
                )
                
            elif task_type == "ordinal":
                # For ordinal regression tasks (e.g., risk assessment)
                self.task_heads[task_name] = OrdinalRegressionHead(
                    input_dim=hidden_dim,
                    num_classes=output_dim,
                    hidden_dim=hidden_dim // 2
                )
                
            elif task_type == "binary":
                # For binary classification tasks (e.g., anomaly detection)
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, output_dim),
                    nn.Sigmoid()
                )
    
    def forward(
        self,
        x: torch.Tensor,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions for multiple financial tasks.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            tasks: Optional list of task names to generate predictions for
                   If None, predictions for all tasks will be generated
            
        Returns:
            Dictionary mapping task names to predictions
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Generate predictions for each task
        predictions = {}
        
        for task_name, head in self.task_heads.items():
            if tasks is None or task_name in tasks:
                task_type = self.task_config[task_name]["type"]
                
                if task_type == "classification":
                    # Apply softmax for classification tasks
                    logits = head(features)
                    predictions[task_name] = {
                        "logits": logits,
                        "probabilities": F.softmax(logits, dim=1)
                    }
                elif task_type == "ordinal":
                    # Get outputs from ordinal head
                    outputs = head(features)
                    predictions[task_name] = outputs
                else:
                    # Direct output for regression and binary tasks
                    predictions[task_name] = head(features)
        
        return predictions


class OrdinalRegressionHead(nn.Module):
    """
    Ordinal regression head for tasks with ordered categorical outputs.
    
    This implements the ordinal regression approach where the model predicts
    cumulative probabilities for each class threshold.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize the ordinal regression head.
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of ordinal classes
            hidden_dim: Dimension of hidden layer
            dropout: Dropout rate
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Shared feature network
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Threshold predictors (one fewer than number of classes)
        self.threshold_predictors = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_classes - 1)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate ordinal predictions.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary with ordinal regression outputs
        """
        # Extract features
        features = self.feature_net(x)
        
        # Predict thresholds
        thresholds = []
        for predictor in self.threshold_predictors:
            thresholds.append(predictor(features))
        
        # Stack thresholds
        thresholds = torch.cat(thresholds, dim=1)  # [batch_size, num_classes-1]
        
        # Apply sigmoid to get cumulative probabilities
        cum_probs = torch.sigmoid(thresholds)
        
        # Calculate class probabilities
        probs = torch.zeros(x.size(0), self.num_classes, device=x.device)
        
        # First class: 1 - P(y > 0)
        probs[:, 0] = 1 - cum_probs[:, 0]
        
        # Middle classes: P(y > k-1) - P(y > k)
        for k in range(1, self.num_classes - 1):
            probs[:, k] = cum_probs[:, k-1] - cum_probs[:, k]
        
        # Last class: P(y > K-2)
        probs[:, -1] = cum_probs[:, -1]
        
        # Predict most likely class
        predicted_class = torch.argmax(probs, dim=1)
        
        return {
            "cumulative_probabilities": cum_probs,
            "probabilities": probs,
            "predicted_class": predicted_class
        }


class FinancialMultiModalModel(nn.Module):
    """
    Complete multi-modal model for financial analysis and forecasting.
    
    This module combines all components:
    - Time series processing with temporal attention
    - Financial text processing
    - Fundamental analysis
    - Network analysis
    - Multi-modal fusion
    - Uncertainty-aware forecasting
    - Multi-task outputs
    """
    
    def __init__(
        self,
        time_series_config: Dict[str, Any],
        text_config: Dict[str, Any],
        fundamental_config: Dict[str, Any],
        network_config: Dict[str, Any],
        fusion_dim: int = 512,
        forecast_steps: int = 5,
        task_config: Dict[str, Dict] = None,
        dropout: float = 0.1
    ):
        """
        Initialize the complete financial multi-modal model.
        
        Args:
            time_series_config: Configuration for time series module
            text_config: Configuration for text module
            fundamental_config: Configuration for fundamental analysis module
            network_config: Configuration for network analysis module
            fusion_dim: Dimension for fused representations
            forecast_steps: Number of steps to forecast
            task_config: Configuration for prediction tasks
            dropout: Dropout rate
        """
        super().__init__()
        
        # Time series module
        self.time_series_module = TemporalAttention(
            input_dim=time_series_config["input_dim"],
            hidden_dim=time_series_config.get("hidden_dim", 256),
            num_heads=time_series_config.get("num_heads", 8),
            dropout=dropout,
            max_seq_length=time_series_config.get("max_seq_length", 252),
            time_scales=time_series_config.get("time_scales", [5, 21, 63])
        )
        
        # Text module
        self.text_module = FinancialTextEncoder(
            base_model_name=text_config.get("base_model_name", "distilbert-base-uncased"),
            hidden_dim=text_config.get("hidden_dim", 768),
            output_dim=fusion_dim,
            dropout=dropout,
            max_length=text_config.get("max_length", 512),
            numerical_feature_dim=text_config.get("numerical_feature_dim", 64)
        )
        
        # Fundamental analysis module
        self.fundamental_module = FundamentalAnalysisModule(
            input_dims=fundamental_config["input_dims"],
            hidden_dim=fundamental_config.get("hidden_dim", 256),
            output_dim=fusion_dim,
            dropout=dropout
        )
        
        # Network analysis module
        self.network_module = FinancialNetworkModule(
            node_feature_dim=network_config["node_feature_dim"],
            edge_feature_dim=network_config["edge_feature_dim"],
            hidden_dim=network_config.get("hidden_dim", 256),
            output_dim=fusion_dim,
            num_layers=network_config.get("num_layers", 3),
            dropout=dropout
        )
        
        # Time series aggregation
        self.time_series_aggregation = nn.Sequential(
            nn.Linear(time_series_config.get("hidden_dim", 256), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout)
        )
        
        # Multi-modal fusion
        self.fusion_module = FinancialMultiModalFusion(
            modality_dims={
                "time_series": fusion_dim,
                "text": fusion_dim,
                "fundamental": fusion_dim,
                "network": fusion_dim
            },
            hidden_dim=fusion_dim,
            output_dim=fusion_dim,
            num_heads=8,
            dropout=dropout
        )
        
        # Uncertainty-aware forecasting
        self.forecasting_module = UncertaintyAwareForecasting(
            input_dim=fusion_dim,
            hidden_dim=fusion_dim // 2,
            forecast_steps=forecast_steps,
            dropout=dropout
        )
        
        # Multi-task output
        self.output_module = FinancialMultiTaskOutput(
            input_dim=fusion_dim,
            hidden_dim=fusion_dim // 2,
            task_config=task_config,
            dropout=dropout
        )
    
    def forward(
        self,
        inputs: Dict[str, Any],
        tasks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process multi-modal financial data.
        
        Args:
            inputs: Dictionary with inputs for different modalities
            tasks: Optional list of tasks to generate predictions for
            
        Returns:
            Dictionary with model outputs
        """
        modality_representations = {}
        
        # Process time series data if available
        if "time_series" in inputs:
            time_series_data = inputs["time_series"]
            time_series_repr = self.time_series_module(time_series_data)
            
            # Aggregate time series representation
            if time_series_repr.dim() == 3:  # [batch_size, seq_length, hidden_dim]
                # Use the last time step
                time_series_repr = time_series_repr[:, -1, :]
            
            modality_representations["time_series"] = self.time_series_aggregation(time_series_repr)
        
        # Process text data if available
        if "text" in inputs:
            text_data = inputs["text"]
            text_repr = self.text_module(
                input_ids=text_data["input_ids"],
                attention_mask=text_data["attention_mask"],
                entity_ids=text_data.get("entity_ids"),
                numerical_values=text_data.get("numerical_values")
            )
            modality_representations["text"] = text_repr
        
        # Process fundamental data if available
        if "fundamental" in inputs:
            fundamental_data = inputs["fundamental"]
            fundamental_repr = self.fundamental_module(
                inputs=fundamental_data,
                temporal_data=inputs.get("fundamental_temporal")
            )
            modality_representations["fundamental"] = fundamental_repr
        
        # Process network data if available
        if "network" in inputs:
            network_data = inputs["network"]
            network_repr = self.network_module(
                node_features=network_data["node_features"],
                edge_index=network_data["edge_index"],
                edge_features=network_data.get("edge_features"),
                batch=network_data.get("batch")
            )
            modality_representations["network"] = network_repr
        
        # Apply multi-modal fusion
        fusion_result = self.fusion_module(modality_representations)
        fused_representation = fusion_result["fused_representation"]
        
        # Generate forecasts
        forecasts = self.forecasting_module(fused_representation)
        
        # Generate task-specific predictions
        predictions = self.output_module(fused_representation, tasks)
        
        # Combine all outputs
        outputs = {
            "modality_representations": modality_representations,
            "fused_representation": fused_representation,
            "modality_importance": fusion_result["modality_importance"],
            "forecasts": forecasts,
            "predictions": predictions
        }
        
        return outputs


def create_scale_mask(seq_length: int, scale: int, device: torch.device) -> torch.Tensor:
    """
    Create an attention mask for a specific time scale.
    
    Args:
        seq_length: Length of the sequence
        scale: Time scale in days
        device: Computation device
        
    Returns:
        Boolean attention mask
    """
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        mask[i, max(0, i - scale):i + 1] = 1
    return mask.bool()