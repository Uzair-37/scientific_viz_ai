"""
Domain adaptation modules for applying the financial model architecture to other scientific domains.

This module provides adapters for using the multi-modal architecture with data from
other scientific domains such as bioinformatics and materials science.
"""
import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional, Any

from .financial_architecture import (
    TemporalAttention,
    FinancialTextEncoder,
    FundamentalAnalysisModule,
    FinancialNetworkModule,
    FinancialMultiModalFusion,
    UncertaintyAwareForecasting,
    FinancialMultiTaskOutput,
    FinancialMultiModalModel
)

logger = logging.getLogger(__name__)

class BioinformaticsAdapter:
    """
    Adapter for applying the multi-modal architecture to bioinformatics data.
    
    Maps various bioinformatics data types (sequences, structures, expression data,
    interaction networks) to the architecture's modalities.
    """
    
    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Initialize the bioinformatics adapter.
        
        Args:
            device: Device to run computations on
        """
        self.device = device
        logger.info(f"Initialized BioinformaticsAdapter (device: {device})")
        
    def create_model(
        self,
        sequence_length: int = 1000,
        expression_features: int = 100,
        structure_features: int = 64,
        network_node_features: int = 32,
        network_edge_features: int = 16,
        fusion_dim: int = 512,
        forecast_steps: int = 5,
        dropout: float = 0.1
    ) -> nn.Module:
        """
        Create a domain-adapted multi-modal model for bioinformatics.
        
        Args:
            sequence_length: Maximum sequence length for genomic/protein sequences
            expression_features: Number of gene expression features
            structure_features: Number of structural features
            network_node_features: Number of node features in biological networks
            network_edge_features: Number of edge features in biological networks
            fusion_dim: Dimension for fused representations
            forecast_steps: Number of steps for forecasting (e.g., time points in expression data)
            dropout: Dropout rate
            
        Returns:
            Domain-adapted multi-modal model
        """
        # Configure time series module (for expression data)
        time_series_config = {
            "input_dim": expression_features,
            "hidden_dim": 256,
            "num_heads": 8,
            "max_seq_length": 100,  # Time points in expression data
            "time_scales": [1, 5, 10]  # Different time scales in expression data
        }
        
        # Configure text module (for sequences)
        text_config = {
            "base_model_name": "distilbert-base-uncased",  # Will be modified for sequences
            "hidden_dim": 768,
            "max_length": sequence_length,
            "numerical_feature_dim": 64
        }
        
        # Configure fundamental analysis module (for structural data)
        fundamental_config = {
            "input_dims": {
                "secondary_structure": 8,  # 8 secondary structure classes
                "solvent_accessibility": 20,  # Solvent accessibility features
                "folding_energy": 8,  # Folding energy components
                "physical_properties": 28  # Physical/chemical properties
            },
            "hidden_dim": 256
        }
        
        # Configure network analysis module (for biological networks)
        network_config = {
            "node_feature_dim": network_node_features,
            "edge_feature_dim": network_edge_features,
            "hidden_dim": 256,
            "num_layers": 3
        }
        
        # Configure task outputs for bioinformatics
        task_config = {
            "function_prediction": {"type": "classification", "output_dim": 10},  # GO terms or function classes
            "structure_prediction": {"type": "regression", "output_dim": structure_features},
            "interaction_prediction": {"type": "binary", "output_dim": 1},
            "expression_forecast": {"type": "regression", "output_dim": forecast_steps}
        }
        
        # Create the model
        model = FinancialMultiModalModel(
            time_series_config=time_series_config,
            text_config=text_config,
            fundamental_config=fundamental_config,
            network_config=network_config,
            fusion_dim=fusion_dim,
            forecast_steps=forecast_steps,
            task_config=task_config,
            dropout=dropout
        )
        
        # Domain-specific modifications
        
        # 1. Replace text encoder with sequence encoder
        model.text_module = SequenceEncoder(
            max_length=sequence_length,
            embedding_dim=256,
            hidden_dim=512,
            output_dim=fusion_dim,
            dropout=dropout
        )
        
        # 2. Modify time series module for expression data
        # (The original module works well for time series, so minimal changes needed)
        
        # 3. Replace fundamental module with structure module
        model.fundamental_module = StructureModule(
            input_dims=fundamental_config["input_dims"],
            hidden_dim=fundamental_config["hidden_dim"],
            output_dim=fusion_dim,
            dropout=dropout
        )
        
        # 4. Modify network module for biological networks
        # (The original module works well for networks, so minimal changes needed)
        
        # 5. Move model to device
        model = model.to(self.device)
        
        logger.info(f"Created domain-adapted model for bioinformatics with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model
    
    def preprocess_data(
        self,
        sequences: Optional[List[str]] = None,
        expression_data: Optional[np.ndarray] = None,
        structure_data: Optional[Dict[str, np.ndarray]] = None,
        network_data: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Preprocess bioinformatics data for the model.
        
        Args:
            sequences: List of genomic/protein sequences
            expression_data: Gene expression data (samples × genes × time)
            structure_data: Dictionary of structural features
            network_data: Dictionary of network data
            
        Returns:
            Dictionary of preprocessed data ready for the model
        """
        batch_size = self._get_batch_size(sequences, expression_data, structure_data, network_data)
        inputs = {}
        
        # Process sequences
        if sequences is not None:
            inputs["text"] = self._process_sequences(sequences, batch_size)
            
        # Process expression data
        if expression_data is not None:
            inputs["time_series"] = torch.tensor(expression_data, dtype=torch.float32).to(self.device)
            
        # Process structure data
        if structure_data is not None:
            inputs["fundamental"] = self._process_structure_data(structure_data, batch_size)
            
        # Process network data
        if network_data is not None:
            inputs["network"] = self._process_network_data(network_data, batch_size)
        
        logger.info(f"Preprocessed bioinformatics data with {batch_size} samples")
        
        return inputs
    
    def _get_batch_size(
        self,
        sequences: Optional[List[str]],
        expression_data: Optional[np.ndarray],
        structure_data: Optional[Dict[str, np.ndarray]],
        network_data: Optional[Dict[str, np.ndarray]]
    ) -> int:
        """Determine the batch size from available data."""
        if sequences is not None:
            return len(sequences)
        elif expression_data is not None:
            return expression_data.shape[0]
        elif structure_data is not None:
            return next(iter(structure_data.values())).shape[0]
        elif network_data is not None:
            return network_data["node_features"].shape[0] if "node_features" in network_data else 1
        else:
            return 1
    
    def _process_sequences(self, sequences: List[str], batch_size: int) -> Dict[str, torch.Tensor]:
        """Process genomic/protein sequences for the model."""
        # Convert sequences to one-hot encoding
        # For simplicity, we'll use a basic approach here
        # In practice, you'd use a more sophisticated encoding
        
        # Define alphabet (DNA, RNA, or protein)
        if all(all(c.upper() in "ACGT" for c in seq) for seq in sequences):
            alphabet = "ACGT"  # DNA
        elif all(all(c.upper() in "ACGU" for c in seq) for seq in sequences):
            alphabet = "ACGU"  # RNA
        else:
            alphabet = "ACDEFGHIKLMNPQRSTVWY"  # Protein
            
        max_length = max(len(seq) for seq in sequences)
        
        # Create one-hot encodings
        one_hot = np.zeros((batch_size, max_length, len(alphabet)))
        
        for i, seq in enumerate(sequences):
            for j, char in enumerate(seq):
                if char.upper() in alphabet:
                    idx = alphabet.index(char.upper())
                    one_hot[i, j, idx] = 1
        
        # Create attention mask
        attention_mask = np.zeros((batch_size, max_length))
        for i, seq in enumerate(sequences):
            attention_mask[i, :len(seq)] = 1
        
        # Convert to tensors
        one_hot_tensor = torch.tensor(one_hot, dtype=torch.float32).to(self.device)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.float32).to(self.device)
        
        return {
            "input_ids": one_hot_tensor,
            "attention_mask": attention_mask_tensor
        }
    
    def _process_structure_data(self, structure_data: Dict[str, np.ndarray], batch_size: int) -> Dict[str, torch.Tensor]:
        """Process structural data for the model."""
        processed = {}
        
        for key, data in structure_data.items():
            processed[key] = torch.tensor(data, dtype=torch.float32).to(self.device)
            
        # Add combined tensor for convenience
        combined = []
        for key in processed:
            combined.append(processed[key].view(batch_size, -1))
        
        processed["combined"] = torch.cat(combined, dim=1)
        
        return processed
    
    def _process_network_data(self, network_data: Dict[str, np.ndarray], batch_size: int) -> Dict[str, torch.Tensor]:
        """Process network data for the model."""
        processed = {}
        
        # Convert node features
        if "node_features" in network_data:
            processed["node_features"] = torch.tensor(
                network_data["node_features"], dtype=torch.float32
            ).to(self.device)
            
        # Convert edge index
        if "edge_index" in network_data:
            processed["edge_index"] = torch.tensor(
                network_data["edge_index"], dtype=torch.long
            ).to(self.device)
            
        # Convert edge features
        if "edge_features" in network_data:
            processed["edge_features"] = torch.tensor(
                network_data["edge_features"], dtype=torch.float32
            ).to(self.device)
            
        # Convert batch assignment
        if "batch" in network_data:
            processed["batch"] = torch.tensor(
                network_data["batch"], dtype=torch.long
            ).to(self.device)
        
        return processed
    
    def interpret_results(self, model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret model outputs in the context of bioinformatics.
        
        Args:
            model_outputs: Outputs from the model
            
        Returns:
            Dictionary with interpreted results
        """
        results = {}
        
        # Modality importance
        if "modality_importance" in model_outputs:
            modality_names = ["Expression Data", "Sequence", "Structure", "Interaction Network"]
            modality_importance = model_outputs["modality_importance"].cpu().numpy()
            
            results["modality_importance"] = {
                name: float(importance)
                for name, importance in zip(modality_names, modality_importance[0])
            }
        
        # Task predictions
        if "predictions" in model_outputs:
            predictions = model_outputs["predictions"]
            
            # Function prediction
            if "function_prediction" in predictions:
                function_pred = predictions["function_prediction"]
                if isinstance(function_pred, dict) and "probabilities" in function_pred:
                    results["function_prediction"] = function_pred["probabilities"].cpu().numpy()
                else:
                    results["function_prediction"] = function_pred.cpu().numpy()
            
            # Structure prediction
            if "structure_prediction" in predictions:
                results["structure_prediction"] = predictions["structure_prediction"].cpu().numpy()
            
            # Interaction prediction
            if "interaction_prediction" in predictions:
                results["interaction_prediction"] = predictions["interaction_prediction"].cpu().numpy()
        
        # Expression forecasts
        if "forecasts" in model_outputs:
            forecasts = model_outputs["forecasts"]
            results["expression_forecast"] = {
                "median": forecasts["median_forecast"].cpu().numpy(),
                "uncertainty": forecasts["prediction_interval_90"].cpu().numpy()
            }
        
        logger.info("Interpreted model outputs for bioinformatics")
        
        return results


class MaterialsScienceAdapter:
    """
    Adapter for applying the multi-modal architecture to materials science data.
    
    Maps various materials science data types (compositions, structures, properties,
    synthesis methods) to the architecture's modalities.
    """
    
    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Initialize the materials science adapter.
        
        Args:
            device: Device to run computations on
        """
        self.device = device
        logger.info(f"Initialized MaterialsScienceAdapter (device: {device})")
        
    def create_model(
        self,
        composition_features: int = 118,  # Number of elements in periodic table
        structure_features: int = 50,
        property_features: int = 30,
        synthesis_max_length: int = 512,
        network_node_features: int = 64,
        network_edge_features: int = 32,
        fusion_dim: int = 512,
        forecast_steps: int = 10,
        dropout: float = 0.1
    ) -> nn.Module:
        """
        Create a domain-adapted multi-modal model for materials science.
        
        Args:
            composition_features: Number of features for material composition
            structure_features: Number of features for crystal structure
            property_features: Number of material property features
            synthesis_max_length: Maximum sequence length for synthesis methods
            network_node_features: Number of node features in materials networks
            network_edge_features: Number of edge features in materials networks
            fusion_dim: Dimension for fused representations
            forecast_steps: Number of steps for property prediction
            dropout: Dropout rate
            
        Returns:
            Domain-adapted multi-modal model
        """
        # Configure time series module (for composition-property relationships)
        time_series_config = {
            "input_dim": composition_features,
            "hidden_dim": 256,
            "num_heads": 8,
            "max_seq_length": 100,  # Composition variations
            "time_scales": [1, 5, 10]  # Different composition scales
        }
        
        # Configure text module (for synthesis methods)
        text_config = {
            "base_model_name": "distilbert-base-uncased",
            "hidden_dim": 768,
            "max_length": synthesis_max_length,
            "numerical_feature_dim": 64
        }
        
        # Configure fundamental analysis module (for material properties)
        fundamental_config = {
            "input_dims": {
                "mechanical": 10,  # Mechanical properties
                "electrical": 8,  # Electrical properties
                "thermal": 5,  # Thermal properties
                "optical": 7  # Optical properties
            },
            "hidden_dim": 256
        }
        
        # Configure network analysis module (for materials networks)
        network_config = {
            "node_feature_dim": network_node_features,
            "edge_feature_dim": network_edge_features,
            "hidden_dim": 256,
            "num_layers": 3
        }
        
        # Configure task outputs for materials science
        task_config = {
            "property_prediction": {"type": "regression", "output_dim": property_features},
            "stability_prediction": {"type": "binary", "output_dim": 1},
            "synthesis_difficulty": {"type": "ordinal", "output_dim": 5},
            "bandgap_prediction": {"type": "regression", "output_dim": 1}
        }
        
        # Create the model
        model = FinancialMultiModalModel(
            time_series_config=time_series_config,
            text_config=text_config,
            fundamental_config=fundamental_config,
            network_config=network_config,
            fusion_dim=fusion_dim,
            forecast_steps=forecast_steps,
            task_config=task_config,
            dropout=dropout
        )
        
        # Domain-specific modifications
        
        # 1. Replace time series module with composition encoder
        model.time_series_module = CompositionEncoder(
            input_dim=composition_features,
            hidden_dim=time_series_config["hidden_dim"],
            output_dim=fusion_dim,
            dropout=dropout
        )
        
        # 2. Modify text encoder for synthesis methods (keep original)
        
        # 3. Replace fundamental module with materials property module
        model.fundamental_module = MaterialsPropertyModule(
            input_dims=fundamental_config["input_dims"],
            hidden_dim=fundamental_config["hidden_dim"],
            output_dim=fusion_dim,
            dropout=dropout
        )
        
        # 4. Modify network module for materials networks
        # (The original module works well for networks, so minimal changes needed)
        
        # 5. Move model to device
        model = model.to(self.device)
        
        logger.info(f"Created domain-adapted model for materials science with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model
    
    def preprocess_data(
        self,
        compositions: Optional[np.ndarray] = None,
        synthesis_texts: Optional[List[str]] = None,
        property_data: Optional[Dict[str, np.ndarray]] = None,
        structure_data: Optional[Dict[str, np.ndarray]] = None,
        network_data: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Preprocess materials science data for the model.
        
        Args:
            compositions: Material composition data (samples × elements)
            synthesis_texts: Synthesis method descriptions
            property_data: Dictionary of material property data
            structure_data: Dictionary of crystal structure data
            network_data: Dictionary of materials network data
            
        Returns:
            Dictionary of preprocessed data ready for the model
        """
        batch_size = self._get_batch_size(compositions, synthesis_texts, property_data, structure_data, network_data)
        inputs = {}
        
        # Process compositions
        if compositions is not None:
            inputs["time_series"] = torch.tensor(compositions, dtype=torch.float32).to(self.device)
            
        # Process synthesis texts
        if synthesis_texts is not None:
            inputs["text"] = self._process_synthesis_texts(synthesis_texts)
            
        # Process property data
        if property_data is not None:
            inputs["fundamental"] = self._process_property_data(property_data, batch_size)
            
        # Process network data (with structure info)
        if network_data is not None:
            if structure_data is not None:
                # Combine structure data with network data
                network_data = self._combine_structure_network(structure_data, network_data)
            
            inputs["network"] = self._process_network_data(network_data, batch_size)
        
        logger.info(f"Preprocessed materials science data with {batch_size} samples")
        
        return inputs
    
    def _get_batch_size(
        self,
        compositions: Optional[np.ndarray],
        synthesis_texts: Optional[List[str]],
        property_data: Optional[Dict[str, np.ndarray]],
        structure_data: Optional[Dict[str, np.ndarray]],
        network_data: Optional[Dict[str, np.ndarray]]
    ) -> int:
        """Determine the batch size from available data."""
        if compositions is not None:
            return compositions.shape[0]
        elif synthesis_texts is not None:
            return len(synthesis_texts)
        elif property_data is not None:
            return next(iter(property_data.values())).shape[0]
        elif structure_data is not None:
            return next(iter(structure_data.values())).shape[0]
        elif network_data is not None:
            return network_data["node_features"].shape[0] if "node_features" in network_data else 1
        else:
            return 1
    
    def _process_synthesis_texts(self, synthesis_texts: List[str]) -> Dict[str, torch.Tensor]:
        """Process synthesis method descriptions using a tokenizer."""
        try:
            from transformers import AutoTokenizer
            
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            
            # Tokenize texts
            encodings = tokenizer(
                synthesis_texts,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            encodings = {key: tensor.to(self.device) for key, tensor in encodings.items()}
            
            return encodings
            
        except ImportError:
            logger.warning("Transformers library not installed. Using basic word-based tokenization.")
            
            # Create a basic integer encoding
            # First, create a vocabulary
            vocab = set()
            for text in synthesis_texts:
                for word in text.split():
                    vocab.add(word.lower())
            
            word_to_idx = {word: i + 1 for i, word in enumerate(sorted(vocab))}
            
            # Encode texts
            max_length = 512
            input_ids = np.zeros((len(synthesis_texts), max_length), dtype=np.int64)
            attention_mask = np.zeros((len(synthesis_texts), max_length), dtype=np.int64)
            
            for i, text in enumerate(synthesis_texts):
                words = text.lower().split()[:max_length]
                for j, word in enumerate(words):
                    input_ids[i, j] = word_to_idx.get(word, 0)
                    attention_mask[i, j] = 1
            
            # Convert to tensors
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).to(self.device)
            attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long).to(self.device)
            
            return {
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor
            }
    
    def _process_property_data(self, property_data: Dict[str, np.ndarray], batch_size: int) -> Dict[str, torch.Tensor]:
        """Process material property data for the model."""
        processed = {}
        
        for key, data in property_data.items():
            processed[key] = torch.tensor(data, dtype=torch.float32).to(self.device)
            
        # Add combined tensor for convenience
        combined = []
        for key in processed:
            combined.append(processed[key].view(batch_size, -1))
        
        processed["combined"] = torch.cat(combined, dim=1)
        
        return processed
    
    def _combine_structure_network(
        self,
        structure_data: Dict[str, np.ndarray],
        network_data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Combine crystal structure data with materials network data."""
        combined = network_data.copy()
        
        # If node features exist in both, concatenate them
        if "node_features" in network_data and "atom_features" in structure_data:
            combined["node_features"] = np.concatenate(
                [network_data["node_features"], structure_data["atom_features"]],
                axis=1
            )
            
        # If edge features exist in both, concatenate them
        if "edge_features" in network_data and "bond_features" in structure_data:
            combined["edge_features"] = np.concatenate(
                [network_data["edge_features"], structure_data["bond_features"]],
                axis=1
            )
            
        # Add any missing features
        if "node_features" not in combined and "atom_features" in structure_data:
            combined["node_features"] = structure_data["atom_features"]
            
        if "edge_features" not in combined and "bond_features" in structure_data:
            combined["edge_features"] = structure_data["bond_features"]
            
        if "edge_index" not in combined and "bond_index" in structure_data:
            combined["edge_index"] = structure_data["bond_index"]
        
        return combined
    
    def _process_network_data(self, network_data: Dict[str, np.ndarray], batch_size: int) -> Dict[str, torch.Tensor]:
        """Process materials network data for the model."""
        processed = {}
        
        # Convert node features
        if "node_features" in network_data:
            processed["node_features"] = torch.tensor(
                network_data["node_features"], dtype=torch.float32
            ).to(self.device)
            
        # Convert edge index
        if "edge_index" in network_data:
            processed["edge_index"] = torch.tensor(
                network_data["edge_index"], dtype=torch.long
            ).to(self.device)
            
        # Convert edge features
        if "edge_features" in network_data:
            processed["edge_features"] = torch.tensor(
                network_data["edge_features"], dtype=torch.float32
            ).to(self.device)
            
        # Convert batch assignment
        if "batch" in network_data:
            processed["batch"] = torch.tensor(
                network_data["batch"], dtype=torch.long
            ).to(self.device)
        
        return processed
    
    def interpret_results(self, model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret model outputs in the context of materials science.
        
        Args:
            model_outputs: Outputs from the model
            
        Returns:
            Dictionary with interpreted results
        """
        results = {}
        
        # Modality importance
        if "modality_importance" in model_outputs:
            modality_names = ["Composition", "Synthesis", "Properties", "Structure"]
            modality_importance = model_outputs["modality_importance"].cpu().numpy()
            
            results["modality_importance"] = {
                name: float(importance)
                for name, importance in zip(modality_names, modality_importance[0])
            }
        
        # Task predictions
        if "predictions" in model_outputs:
            predictions = model_outputs["predictions"]
            
            # Property prediction
            if "property_prediction" in predictions:
                results["property_prediction"] = predictions["property_prediction"].cpu().numpy()
            
            # Stability prediction
            if "stability_prediction" in predictions:
                results["stability_prediction"] = predictions["stability_prediction"].cpu().numpy()
            
            # Synthesis difficulty
            if "synthesis_difficulty" in predictions:
                synth_diff = predictions["synthesis_difficulty"]
                if isinstance(synth_diff, dict) and "predicted_class" in synth_diff:
                    results["synthesis_difficulty"] = synth_diff["predicted_class"].cpu().numpy()
                else:
                    results["synthesis_difficulty"] = synth_diff.cpu().numpy()
            
            # Bandgap prediction
            if "bandgap_prediction" in predictions:
                results["bandgap_prediction"] = predictions["bandgap_prediction"].cpu().numpy()
        
        # Property forecasts
        if "forecasts" in model_outputs:
            forecasts = model_outputs["forecasts"]
            results["property_forecast"] = {
                "median": forecasts["median_forecast"].cpu().numpy(),
                "uncertainty": forecasts["prediction_interval_90"].cpu().numpy()
            }
        
        logger.info("Interpreted model outputs for materials science")
        
        return results


# Domain-specific modules

class SequenceEncoder(nn.Module):
    """
    Encoder for biological sequences (DNA, RNA, protein).
    """
    def __init__(
        self,
        max_length: int = 1000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize the sequence encoder.
        
        Args:
            max_length: Maximum sequence length
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Input projection (from one-hot to embedding)
        self.input_projection = nn.Linear(20, embedding_dim)  # 20 for protein, can handle DNA/RNA too
        
        # Positional encoding
        self.position_encoding = nn.Parameter(torch.zeros(1, max_length, embedding_dim))
        nn.init.trunc_normal_(self.position_encoding, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode biological sequences.
        
        Args:
            input_ids: One-hot encoded sequences [batch, seq_len, alphabet_size]
            attention_mask: Attention mask for padding
            
        Returns:
            Sequence embeddings
        """
        # Project inputs to embedding dimension
        x = self.input_projection(input_ids)
        
        # Add positional encoding
        seq_length = x.size(1)
        x = x + self.position_encoding[:, :seq_length, :]
        
        # Create attention mask for transformer
        # Convert from [batch, seq_len] to [batch, seq_len, seq_len]
        extended_mask = attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)
        extended_mask = (1.0 - extended_mask) * -10000.0
        
        # Apply transformer
        encoded = self.transformer(x, src_mask=extended_mask)
        
        # Global pooling (mean of non-padding tokens)
        mask_expanded = attention_mask.unsqueeze(-1).expand(encoded.size())
        sum_embeddings = torch.sum(encoded * mask_expanded, dim=1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # Project to output dimension
        output = self.output_projection(pooled)
        
        return output


class StructureModule(nn.Module):
    """
    Module for processing biological structure data.
    """
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize the structure module.
        
        Args:
            input_dims: Dictionary mapping structure data types to dimensions
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dims = input_dims
        
        # Feature extractors for different structure data types
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
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Process biological structure data.
        
        Args:
            inputs: Dictionary mapping data types to input tensors
            
        Returns:
            Structure embeddings
        """
        # Extract features from each data type
        features = {}
        for data_type, tensor in inputs.items():
            if data_type in self.feature_extractors:
                features[data_type] = self.feature_extractors[data_type](tensor)
        
        # Handle the case where a combined tensor is provided
        if "combined" in inputs and "combined" not in self.feature_extractors:
            # Create a feature extractor for the combined input
            combined_dim = inputs["combined"].size(-1)
            self.feature_extractors["combined"] = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ).to(inputs["combined"].device)
            
            features["combined"] = self.feature_extractors["combined"](inputs["combined"])
        
        # Combine features
        if features:
            # Stack features for cross-attention
            stacked_features = torch.stack(list(features.values()), dim=1)
            
            # Apply cross-attention
            attended_features, _ = self.cross_attention(
                query=stacked_features,
                key=stacked_features,
                value=stacked_features
            )
            
            # Mean pooling across feature types
            pooled_features = attended_features.mean(dim=1)
            
            # Project to output dimension
            output = self.projection(pooled_features)
            
            return output
        else:
            # No valid inputs, return zeros
            batch_size = next(iter(inputs.values())).size(0)
            return torch.zeros(batch_size, output_dim, device=next(iter(inputs.values())).device)


class CompositionEncoder(nn.Module):
    """
    Encoder for material compositions.
    """
    def __init__(
        self,
        input_dim: int = 118,  # Default to periodic table size
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize the composition encoder.
        
        Args:
            input_dim: Number of elements in composition vector
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Element-wise attention
        self.element_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Pooling attention
        self.pooling_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode material compositions.
        
        Args:
            x: Composition vectors [batch, num_elements] or [batch, seq_len, num_elements]
            
        Returns:
            Composition embeddings
        """
        # Handle different input shapes
        if x.dim() == 2:
            # If input is [batch, elements], expand to [batch, 1, elements]
            x = x.unsqueeze(1)
        
        batch_size, seq_len, num_elements = x.shape
        
        # Project elements to hidden dimension
        projected = self.input_projection(x)  # [batch, seq, hidden]
        
        # Apply element-wise attention
        attended, _ = self.element_attention(
            query=projected,
            key=projected,
            value=projected
        )
        
        # Apply weighted pooling over sequence dimension
        weights = self.pooling_attention(attended)  # [batch, seq, 1]
        pooled = torch.sum(attended * weights, dim=1)  # [batch, hidden]
        
        # Project to output dimension
        output = self.output_projection(pooled)  # [batch, output]
        
        return output


class MaterialsPropertyModule(nn.Module):
    """
    Module for processing materials property data.
    """
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize the materials property module.
        
        Args:
            input_dims: Dictionary mapping property types to dimensions
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dims = input_dims
        
        # Feature extractors for different property types
        self.feature_extractors = nn.ModuleDict({
            prop_type: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for prop_type, dim in input_dims.items()
        })
        
        # Cross-property attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Property importance weighting
        self.property_importance = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Process materials property data.
        
        Args:
            inputs: Dictionary mapping property types to input tensors
            
        Returns:
            Property embeddings
        """
        # Extract features from each property type
        features = {}
        for prop_type, tensor in inputs.items():
            if prop_type in self.feature_extractors:
                features[prop_type] = self.feature_extractors[prop_type](tensor)
        
        # Handle the case where a combined tensor is provided
        if "combined" in inputs and "combined" not in self.feature_extractors:
            # Create a feature extractor for the combined input
            combined_dim = inputs["combined"].size(-1)
            self.feature_extractors["combined"] = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ).to(inputs["combined"].device)
            
            features["combined"] = self.feature_extractors["combined"](inputs["combined"])
        
        # Combine features
        if features:
            # Stack features for cross-attention
            stacked_features = torch.stack(list(features.values()), dim=1)
            
            # Apply cross-attention
            attended_features, _ = self.cross_attention(
                query=stacked_features,
                key=stacked_features,
                value=stacked_features
            )
            
            # Apply weighted pooling
            weights = self.property_importance(attended_features)
            pooled_features = torch.sum(attended_features * weights, dim=1)
            
            # Project to output dimension
            output = self.projection(pooled_features)
            
            return output
        else:
            # No valid inputs, return zeros
            batch_size = next(iter(inputs.values())).size(0)
            return torch.zeros(batch_size, output_dim, device=next(iter(inputs.values())).device)