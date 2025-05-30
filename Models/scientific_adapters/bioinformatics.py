"""
Bioinformatics domain adapter for multi-modal scientific analysis.

This module provides specialized components for processing biological data,
including protein structures, sequences, and genomic data.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class ProteinSequenceEncoder(nn.Module):
    """
    Encoder for protein sequence data using transformers.
    """
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_length: int = 1000
    ):
        """
        Initialize the protein sequence encoder.
        
        Args:
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension for feed-forward networks
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            max_length: Maximum sequence length
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Amino acid embedding (20 standard + special tokens)
        self.aa_embedding = nn.Embedding(26, embed_dim, padding_idx=0)
        
        # Positional encoding
        self.register_buffer("position_ids", torch.arange(max_length).expand((1, -1)))
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        sequences: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode protein sequences.
        
        Args:
            sequences: Tensor of amino acid indices [batch_size, seq_length]
            attention_mask: Mask for padding (1 for aa, 0 for padding)
            
        Returns:
            Sequence encodings [batch_size, embed_dim]
        """
        batch_size, seq_length = sequences.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (sequences > 0).float()
        
        # Get embeddings
        aa_embeddings = self.aa_embedding(sequences)
        position_embeddings = self.position_embedding(self.position_ids[:, :seq_length])
        
        # Combine embeddings
        embeddings = aa_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        
        # Apply transformer with masking
        key_padding_mask = (attention_mask == 0)
        transformed = self.transformer(
            src=embeddings,
            src_key_padding_mask=key_padding_mask
        )
        
        # Global pooling (mean of non-padding tokens)
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(transformed)
        sum_embeddings = torch.sum(transformed * mask_expanded, dim=1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        pooled = sum_embeddings / sum_mask.clamp(min=1e-9)
        
        return pooled


class ProteinStructureEncoder(nn.Module):
    """
    Encoder for 3D protein structure data (backbone and side chains).
    """
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize the protein structure encoder.
        
        Args:
            embed_dim: Embedding dimension for residues
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            num_layers: Number of graph layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Residue embedding (20 standard + special tokens)
        self.residue_embedding = nn.Embedding(26, embed_dim, padding_idx=0)
        
        # Process residue features (coordinates + properties)
        self.residue_features = nn.Sequential(
            nn.Linear(embed_dim + 12, hidden_dim),  # 12 = 3 (coord) + 9 (properties)
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Edge feature network (distance-based)
        self.edge_network = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),  # 1 = distance
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU()
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.message_layers.append(
                MessagePassingLayer(
                    hidden_dim=hidden_dim,
                    dropout=dropout
                )
            )
            
        # Global pooling
        self.pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(
        self,
        residues: torch.Tensor,  # [batch_size, num_residues]
        coordinates: torch.Tensor,  # [batch_size, num_residues, 3]
        features: Optional[torch.Tensor] = None,  # [batch_size, num_residues, feat_dim]
        mask: Optional[torch.Tensor] = None,  # [batch_size, num_residues]
        edge_index: Optional[torch.Tensor] = None,  # [batch_size, 2, num_edges]
        distances: Optional[torch.Tensor] = None  # [batch_size, num_edges]
    ) -> torch.Tensor:
        """
        Encode protein structure.
        
        Args:
            residues: Residue type indices
            coordinates: 3D coordinates of residues (alpha carbon or centroid)
            features: Additional residue features (optional)
            mask: Mask for padding (1 for residue, 0 for padding)
            edge_index: Indices of connected residues for each batch (optional)
            distances: Distances between connected residues (optional)
            
        Returns:
            Structure encoding [batch_size, output_dim]
        """
        batch_size, num_residues = residues.shape
        device = residues.device
        
        # Create mask if not provided
        if mask is None:
            mask = (residues > 0).float()
        
        # Embed residues
        residue_embeds = self.residue_embedding(residues)  # [batch_size, num_residues, embed_dim]
        
        # Combine with features if provided, otherwise use zeros
        if features is None:
            features = torch.zeros(batch_size, num_residues, 9, device=device)
            
        node_features = torch.cat([residue_embeds, coordinates, features], dim=-1)
        node_features = self.residue_features(node_features)  # [batch_size, num_residues, hidden_dim]
        
        # Create edges and distances if not provided
        if edge_index is None or distances is None:
            # Compute pairwise distances for each batch
            expanded_coords1 = coordinates.unsqueeze(2)  # [batch_size, num_residues, 1, 3]
            expanded_coords2 = coordinates.unsqueeze(1)  # [batch_size, 1, num_residues, 3]
            
            pairwise_distances = torch.norm(expanded_coords1 - expanded_coords2, dim=-1)  # [batch_size, num_residues, num_residues]
            
            # Create edges for residues within cutoff (e.g., 10 Angstroms)
            cutoff = 10.0
            adjacency = (pairwise_distances < cutoff).float() * mask.unsqueeze(2) * mask.unsqueeze(1)
            
            # Remove self-connections
            self_mask = torch.eye(num_residues, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            adjacency = adjacency * (1 - self_mask)
            
            # Convert adjacency to edge index
            edge_index = []
            distances = []
            
            for b in range(batch_size):
                edges = torch.nonzero(adjacency[b])  # [num_edges, 2]
                edge_index.append(edges.t())  # [2, num_edges]
                
                # Get distances for these edges
                edge_distances = pairwise_distances[b, edges[:, 0], edges[:, 1]]  # [num_edges]
                distances.append(edge_distances)
                
        # Process each graph in the batch
        node_encodings = []
        
        for b in range(batch_size):
            # Get node features for this batch
            nodes = node_features[b]  # [num_residues, hidden_dim]
            
            # Get edges for this batch
            batch_edges = edge_index[b] if isinstance(edge_index, list) else edge_index[b]
            batch_distances = distances[b] if isinstance(distances, list) else distances[b]
            
            # Compute edge features
            edge_features = self.edge_network(batch_distances.unsqueeze(1))  # [num_edges, hidden_dim]
            
            # Apply message passing
            for layer in self.message_layers:
                nodes = layer(nodes, batch_edges, edge_features)
                
            # Apply mask to zero out padding
            batch_mask = mask[b].unsqueeze(1)  # [num_residues, 1]
            nodes = nodes * batch_mask
            
            # Average pooling (accounting for mask)
            node_sum = torch.sum(nodes, dim=0)  # [hidden_dim]
            mask_sum = torch.sum(batch_mask)
            pooled = node_sum / mask_sum.clamp(min=1.0)
            
            node_encodings.append(pooled)
            
        # Stack results
        stacked_encodings = torch.stack(node_encodings)  # [batch_size, hidden_dim]
        
        # Final projection and normalization
        output = self.pooling(stacked_encodings)  # [batch_size, output_dim]
        normalized = self.layer_norm(output)
        
        return normalized


class MessagePassingLayer(nn.Module):
    """
    Message passing layer for graph-based processing.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        nodes: torch.Tensor,  # [num_nodes, hidden_dim]
        edge_index: torch.Tensor,  # [2, num_edges]
        edge_features: torch.Tensor  # [num_edges, hidden_dim]
    ) -> torch.Tensor:
        """
        Apply message passing.
        
        Args:
            nodes: Node features
            edge_index: Indices of connected nodes
            edge_features: Edge features
            
        Returns:
            Updated node features
        """
        # Get node features for each edge
        source, target = edge_index
        source_features = nodes[source]  # [num_edges, hidden_dim]
        
        # Compute messages
        messages = torch.cat([source_features, edge_features], dim=1)  # [num_edges, hidden_dim*2]
        messages = self.message_mlp(messages)  # [num_edges, hidden_dim]
        
        # Aggregate messages (sum) for each target node
        num_nodes = nodes.size(0)
        aggregated = torch.zeros_like(nodes)  # [num_nodes, hidden_dim]
        aggregated.index_add_(0, target, messages)
        
        # Update node features
        updated = torch.cat([nodes, aggregated], dim=1)  # [num_nodes, hidden_dim*2]
        updated = self.update_mlp(updated)  # [num_nodes, hidden_dim]
        
        # Residual connection
        nodes = nodes + updated
        
        return nodes


class GenomicDataEncoder(nn.Module):
    """
    Encoder for genomic data (DNA sequences, variants, gene expression).
    """
    def __init__(
        self,
        input_dim: int = 4,  # For one-hot encoded DNA or gene expression values
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1
    ):
        """
        Initialize the genomic data encoder.
        
        Args:
            input_dim: Input dimension (4 for one-hot DNA, variable for gene expression)
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            num_layers: Number of convolutional layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initial projection
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # CNN for sequence processing
        self.conv_blocks = nn.ModuleList()
        current_dim = embed_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i > 0 else hidden_dim // 2
            
            block = nn.Sequential(
                nn.Conv1d(current_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            )
            
            self.conv_blocks.append(block)
            current_dim = out_dim
            
        # Global aggregation
        self.aggregation = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(
        self,
        x: torch.Tensor,  # [batch_size, seq_length, input_dim]
        mask: Optional[torch.Tensor] = None  # [batch_size, seq_length]
    ) -> torch.Tensor:
        """
        Encode genomic data.
        
        Args:
            x: Input data (one-hot encoded DNA or gene expression)
            mask: Mask for padding (1 for data, 0 for padding)
            
        Returns:
            Encoded representation [batch_size, output_dim]
        """
        batch_size, seq_length, _ = x.shape
        
        # Create mask if not provided
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=x.device)
            
        # Embed input
        embedded = self.embedding(x)  # [batch_size, seq_length, embed_dim]
        
        # Apply mask
        mask_expanded = mask.unsqueeze(-1)  # [batch_size, seq_length, 1]
        embedded = embedded * mask_expanded
        
        # Transpose for CNN
        embedded = embedded.transpose(1, 2)  # [batch_size, embed_dim, seq_length]
        
        # Apply CNN blocks
        features = embedded
        for block in self.conv_blocks:
            features = block(features)
            
        # Global aggregation
        output = self.aggregation(features)  # [batch_size, output_dim]
        normalized = self.layer_norm(output)
        
        return normalized


class BioinformaticsAdapter(nn.Module):
    """
    Complete adapter for bioinformatics domain.
    """
    def __init__(
        self,
        embed_dim: int = 512,
        protein_seq_hidden_dim: int = 512,
        protein_struct_hidden_dim: int = 256,
        genomic_hidden_dim: int = 256,
        num_outputs: int = 1,
        uncertainty: bool = True
    ):
        """
        Initialize the bioinformatics adapter.
        
        Args:
            embed_dim: Common embedding dimension
            protein_seq_hidden_dim: Hidden dimension for protein sequence encoder
            protein_struct_hidden_dim: Hidden dimension for protein structure encoder
            genomic_hidden_dim: Hidden dimension for genomic data encoder
            num_outputs: Number of prediction outputs
            uncertainty: Whether to predict uncertainty in outputs
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Domain-specific encoders
        self.protein_seq_encoder = ProteinSequenceEncoder(
            embed_dim=embed_dim,
            hidden_dim=protein_seq_hidden_dim
        )
        
        self.protein_struct_encoder = ProteinStructureEncoder(
            embed_dim=embed_dim,
            hidden_dim=protein_struct_hidden_dim,
            output_dim=embed_dim
        )
        
        self.genomic_encoder = GenomicDataEncoder(
            hidden_dim=genomic_hidden_dim,
            output_dim=embed_dim
        )
        
        # Cross-attention for multi-modal fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
        # Output layers
        self.uncertainty = uncertainty
        if uncertainty:
            self.mean_predictor = nn.Linear(embed_dim, num_outputs)
            self.var_predictor = nn.Sequential(
                nn.Linear(embed_dim, num_outputs),
                nn.Softplus()  # Ensure positive variance
            )
        else:
            self.predictor = nn.Linear(embed_dim, num_outputs)
            
    def encode(
        self,
        protein_seq: Optional[Dict[str, torch.Tensor]] = None,
        protein_struct: Optional[Dict[str, torch.Tensor]] = None,
        genomic_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Encode bioinformatics data.
        
        Args:
            protein_seq: Dictionary with protein sequence data
                - sequences: Sequence indices [batch_size, seq_length]
                - attention_mask: Optional mask [batch_size, seq_length]
            protein_struct: Dictionary with protein structure data
                - residues: Residue indices [batch_size, num_residues]
                - coordinates: 3D coordinates [batch_size, num_residues, 3]
                - features: Optional features [batch_size, num_residues, feat_dim]
                - mask: Optional mask [batch_size, num_residues]
            genomic_data: Dictionary with genomic data
                - sequences: DNA or expression data [batch_size, seq_length, input_dim]
                - mask: Optional mask [batch_size, seq_length]
            
        Returns:
            Encoded representation [batch_size, embed_dim]
        """
        encodings = []
        
        # Encode protein sequences if provided
        if protein_seq is not None:
            seq_encoding = self.protein_seq_encoder(
                sequences=protein_seq["sequences"],
                attention_mask=protein_seq.get("attention_mask", None)
            )
            encodings.append(seq_encoding)
            
        # Encode protein structures if provided
        if protein_struct is not None:
            struct_encoding = self.protein_struct_encoder(
                residues=protein_struct["residues"],
                coordinates=protein_struct["coordinates"],
                features=protein_struct.get("features", None),
                mask=protein_struct.get("mask", None),
                edge_index=protein_struct.get("edge_index", None),
                distances=protein_struct.get("distances", None)
            )
            encodings.append(struct_encoding)
            
        # Encode genomic data if provided
        if genomic_data is not None:
            genomic_encoding = self.genomic_encoder(
                x=genomic_data["sequences"],
                mask=genomic_data.get("mask", None)
            )
            encodings.append(genomic_encoding)
            
        # If only one modality is provided, return its encoding
        if len(encodings) == 1:
            return encodings[0]
        
        # If multiple modalities, use attention for fusion
        # Reshape encodings for attention
        stacked_encodings = torch.stack(encodings, dim=1)  # [batch_size, num_modalities, embed_dim]
        
        # Self-attention across modalities
        fused_encoding, _ = self.cross_attention(
            query=stacked_encodings,
            key=stacked_encodings,
            value=stacked_encodings
        )
        
        # Average across modalities
        fused_encoding = torch.mean(fused_encoding, dim=1)  # [batch_size, embed_dim]
        
        return fused_encoding
    
    def predict(
        self,
        encoding: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Make predictions from encoded representation.
        
        Args:
            encoding: Encoded representation [batch_size, embed_dim]
            
        Returns:
            Predictions, with uncertainty if enabled
        """
        features = self.prediction_head(encoding)
        
        if self.uncertainty:
            mean = self.mean_predictor(features)
            var = self.var_predictor(features)
            return mean, var
        else:
            return self.predictor(features)
    
    def forward(
        self,
        protein_seq: Optional[Dict[str, torch.Tensor]] = None,
        protein_struct: Optional[Dict[str, torch.Tensor]] = None,
        genomic_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Full forward pass.
        
        Args:
            protein_seq: Dictionary with protein sequence data
            protein_struct: Dictionary with protein structure data
            genomic_data: Dictionary with genomic data
            
        Returns:
            Dictionary with encodings and predictions
        """
        # Encode data
        encoding = self.encode(protein_seq, protein_struct, genomic_data)
        
        # Make predictions
        predictions = self.predict(encoding)
        
        # Prepare output dictionary
        if isinstance(predictions, tuple):
            mean, var = predictions
            return {
                "encoding": encoding,
                "prediction_mean": mean,
                "prediction_var": var
            }
        else:
            return {
                "encoding": encoding,
                "predictions": predictions
            }