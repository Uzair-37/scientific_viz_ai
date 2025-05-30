"""
Scientific data loaders and processing utilities.

This module provides functionality for loading and preprocessing data 
for materials science and bioinformatics applications.
"""
import os
import logging
import json
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class MaterialsScienceDataset(Dataset):
    """
    Dataset for materials science applications.
    
    Handles crystal structures, spectroscopy data, and property labels.
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        modalities: List[str] = ["crystal", "spectral"],
        transform: Optional[Dict[str, callable]] = None,
        max_atoms: int = 100,
        max_samples: Optional[int] = None,
        target_property: Optional[str] = None
    ):
        """
        Initialize the materials science dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Data split ("train", "val", "test")
            modalities: List of modalities to load
            transform: Dict of transforms to apply to each modality
            max_atoms: Maximum number of atoms in crystal structures
            max_samples: Maximum number of samples to load (for debugging)
            target_property: Target property to predict (if None, load all)
        """
        self.data_dir = data_dir
        self.split = split
        self.modalities = modalities
        self.transform = transform or {}
        self.max_atoms = max_atoms
        self.max_samples = max_samples
        self.target_property = target_property
        
        # Load metadata and samples
        self.metadata = self._load_metadata()
        self.samples = self._load_samples()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load dataset metadata.
        
        Returns:
            Metadata dictionary
        """
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")
            return metadata
        else:
            logger.warning(f"No metadata file found at {metadata_path}")
            return {}
        
    def _load_samples(self) -> List[Dict[str, Any]]:
        """
        Load data samples based on the split.
        
        Returns:
            List of sample dictionaries
        """
        # In a real implementation, this would load actual data files
        # For now, we generate synthetic data
        
        # Path for the split file
        split_path = os.path.join(self.data_dir, f"{self.split}_indices.json")
        
        if os.path.exists(split_path):
            # Load real indices
            with open(split_path, "r") as f:
                indices = json.load(f)
            logger.info(f"Loaded {len(indices)} {self.split} sample indices")
        else:
            # Generate synthetic indices
            logger.warning(f"No split file found at {split_path}, generating synthetic data")
            if self.split == "train":
                indices = list(range(1000))
            elif self.split == "val":
                indices = list(range(1000, 1200))
            else:  # test
                indices = list(range(1200, 1500))
        
        # Limit the number of samples if needed
        if self.max_samples and len(indices) > self.max_samples:
            indices = indices[:self.max_samples]
        
        # Load or generate samples
        samples = []
        
        for idx in indices:
            sample = self._load_or_generate_sample(idx)
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} {self.split} samples")
        return samples
    
    def _load_or_generate_sample(self, idx: int) -> Dict[str, Any]:
        """
        Load a sample from files or generate synthetic data.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample dictionary
        """
        # In a real implementation, this would load from files
        # For now, we generate synthetic data
        sample = {"idx": idx}
        
        if "crystal" in self.modalities:
            # Generate synthetic crystal structure
            num_atoms = np.random.randint(5, self.max_atoms)
            sample["crystal"] = {
                "atom_types": np.random.randint(1, 100, size=num_atoms),  # Atomic numbers
                "positions": np.random.rand(num_atoms, 3) * 10,  # Coordinates in Angstroms
                "lattice": np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])  # Lattice parameters
            }
        
        if "spectral" in self.modalities:
            # Generate synthetic spectroscopy data
            sample["spectral"] = {
                "xrd": np.random.rand(1024),  # X-ray diffraction
                "xps": np.random.rand(512),   # X-ray photoelectron spectroscopy
                "raman": np.random.rand(256)  # Raman spectroscopy
            }
        
        if self.target_property:
            # Generate synthetic target property
            sample["target"] = np.random.rand()
        else:
            # Generate multiple properties
            sample["properties"] = {
                "band_gap": np.random.rand() * 5,
                "formation_energy": np.random.rand() * -10,
                "elastic_modulus": np.random.rand() * 200,
                "thermal_conductivity": np.random.rand() * 100,
                "dielectric_constant": np.random.rand() * 30
            }
        
        return sample
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        processed_sample = {}
        
        # Process crystal structure data
        if "crystal" in self.modalities and "crystal" in sample:
            crystal_data = sample["crystal"]
            
            # Convert to tensors and pad to max_atoms
            atom_types = np.zeros(self.max_atoms, dtype=np.int64)
            positions = np.zeros((self.max_atoms, 3), dtype=np.float32)
            mask = np.zeros(self.max_atoms, dtype=np.float32)
            
            # Fill with actual data
            num_atoms = len(crystal_data["atom_types"])
            atom_types[:num_atoms] = crystal_data["atom_types"]
            positions[:num_atoms] = crystal_data["positions"]
            mask[:num_atoms] = 1.0
            
            # Create tensor dictionary
            processed_sample["atom_types"] = torch.tensor(atom_types)
            processed_sample["positions"] = torch.tensor(positions)
            processed_sample["lattice"] = torch.tensor(crystal_data["lattice"], dtype=torch.float32)
            processed_sample["mask"] = torch.tensor(mask)
            
            # Apply transform if provided
            if "crystal" in self.transform:
                processed_sample = self.transform["crystal"](processed_sample)
        
        # Process spectral data
        if "spectral" in self.modalities and "spectral" in sample:
            spectral_data = sample["spectral"]
            
            # Combine spectral data
            combined_spectrum = np.concatenate([
                spectral_data["xrd"],
                spectral_data["xps"],
                spectral_data["raman"]
            ])
            
            processed_sample["spectrum"] = torch.tensor(combined_spectrum, dtype=torch.float32)
            
            # Apply transform if provided
            if "spectral" in self.transform:
                processed_sample["spectrum"] = self.transform["spectral"](processed_sample["spectrum"])
        
        # Process target data
        if self.target_property and "properties" in sample:
            processed_sample["target"] = torch.tensor([sample["properties"][self.target_property]], dtype=torch.float32)
        elif "target" in sample:
            processed_sample["target"] = torch.tensor([sample["target"]], dtype=torch.float32)
        elif "properties" in sample:
            # Convert all properties to a tensor
            properties = np.array(list(sample["properties"].values()), dtype=np.float32)
            processed_sample["target"] = torch.tensor(properties)
        
        return processed_sample


class BioinformaticsDataset(Dataset):
    """
    Dataset for bioinformatics applications.
    
    Handles protein sequences, structures, and genomic data.
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        modalities: List[str] = ["protein_seq", "protein_struct", "genomic"],
        transform: Optional[Dict[str, callable]] = None,
        max_seq_length: int = 1000,
        max_residues: int = 500,
        max_genomic_length: int = 5000,
        max_samples: Optional[int] = None
    ):
        """
        Initialize the bioinformatics dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Data split ("train", "val", "test")
            modalities: List of modalities to load
            transform: Dict of transforms to apply to each modality
            max_seq_length: Maximum protein sequence length
            max_residues: Maximum number of residues in protein structures
            max_genomic_length: Maximum genomic sequence length
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.data_dir = data_dir
        self.split = split
        self.modalities = modalities
        self.transform = transform or {}
        self.max_seq_length = max_seq_length
        self.max_residues = max_residues
        self.max_genomic_length = max_genomic_length
        self.max_samples = max_samples
        
        # Define amino acid and nucleotide mappings
        self.aa_to_idx = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.aa_to_idx['<pad>'] = 0
        self.aa_to_idx['<unk>'] = 21
        
        self.nuc_to_idx = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}
        
        # Load metadata and samples
        self.metadata = self._load_metadata()
        self.samples = self._load_samples()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load dataset metadata.
        
        Returns:
            Metadata dictionary
        """
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")
            return metadata
        else:
            logger.warning(f"No metadata file found at {metadata_path}")
            return {}
        
    def _load_samples(self) -> List[Dict[str, Any]]:
        """
        Load data samples based on the split.
        
        Returns:
            List of sample dictionaries
        """
        # Path for the split file
        split_path = os.path.join(self.data_dir, f"{self.split}_indices.json")
        
        if os.path.exists(split_path):
            # Load real indices
            with open(split_path, "r") as f:
                indices = json.load(f)
            logger.info(f"Loaded {len(indices)} {self.split} sample indices")
        else:
            # Generate synthetic indices
            logger.warning(f"No split file found at {split_path}, generating synthetic data")
            if self.split == "train":
                indices = list(range(1000))
            elif self.split == "val":
                indices = list(range(1000, 1200))
            else:  # test
                indices = list(range(1200, 1500))
        
        # Limit the number of samples if needed
        if self.max_samples and len(indices) > self.max_samples:
            indices = indices[:self.max_samples]
        
        # Load or generate samples
        samples = []
        
        for idx in indices:
            sample = self._load_or_generate_sample(idx)
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} {self.split} samples")
        return samples
    
    def _load_or_generate_sample(self, idx: int) -> Dict[str, Any]:
        """
        Load a sample from files or generate synthetic data.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample dictionary
        """
        # In a real implementation, this would load from files
        # For now, we generate synthetic data
        sample = {"idx": idx}
        
        if "protein_seq" in self.modalities:
            # Generate synthetic protein sequence
            seq_length = np.random.randint(50, self.max_seq_length)
            sequence = np.random.choice(list(self.aa_to_idx.keys())[:20], size=seq_length)
            sample["protein_seq"] = "".join(sequence)
        
        if "protein_struct" in self.modalities:
            # Generate synthetic protein structure
            if "protein_seq" in sample:
                # Use same length as sequence if available
                num_residues = len(sample["protein_seq"])
            else:
                num_residues = np.random.randint(50, self.max_residues)
                
            # Generate residue types
            residue_types = np.random.choice(list(self.aa_to_idx.keys())[:20], size=num_residues)
            
            # Generate 3D coordinates with some spatial coherence
            positions = np.zeros((num_residues, 3))
            positions[0] = np.random.randn(3)
            
            for i in range(1, num_residues):
                # Add a random displacement to the previous residue's position
                positions[i] = positions[i-1] + np.random.randn(3) * 0.5 + np.array([1.0, 0.0, 0.0])
            
            sample["protein_struct"] = {
                "residues": "".join(residue_types),
                "positions": positions
            }
        
        if "genomic" in self.modalities:
            # Generate synthetic genomic data
            genomic_length = np.random.randint(500, self.max_genomic_length)
            sequence = np.random.choice(['A', 'C', 'G', 'T'], size=genomic_length)
            sample["genomic"] = "".join(sequence)
            
            # Generate gene expression values
            num_genes = 100
            sample["expression"] = np.random.rand(num_genes) * 10
        
        # Generate target values (e.g., protein function, binding affinity)
        sample["target"] = np.random.rand()
        
        return sample
    
    def _encode_sequence(self, sequence: str, max_length: int, mapping: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a biological sequence as indices.
        
        Args:
            sequence: Amino acid or nucleotide sequence
            max_length: Maximum sequence length
            mapping: Dictionary mapping characters to indices
            
        Returns:
            Tuple of (encoded_sequence, attention_mask)
        """
        # Truncate if necessary
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        
        # Convert to indices
        encoded = np.zeros(max_length, dtype=np.int64)
        mask = np.zeros(max_length, dtype=np.float32)
        
        for i, char in enumerate(sequence):
            encoded[i] = mapping.get(char, mapping.get('<unk>', 0))
            mask[i] = 1.0
        
        return encoded, mask
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        processed_sample = {}
        
        # Process protein sequence data
        if "protein_seq" in self.modalities and "protein_seq" in sample:
            sequence = sample["protein_seq"]
            encoded_seq, seq_mask = self._encode_sequence(sequence, self.max_seq_length, self.aa_to_idx)
            
            processed_sample["sequences"] = torch.tensor(encoded_seq)
            processed_sample["attention_mask"] = torch.tensor(seq_mask)
            
            # Apply transform if provided
            if "protein_seq" in self.transform:
                processed_sample = self.transform["protein_seq"](processed_sample)
        
        # Process protein structure data
        if "protein_struct" in self.modalities and "protein_struct" in sample:
            struct_data = sample["protein_struct"]
            
            # Encode residues
            encoded_residues, res_mask = self._encode_sequence(
                struct_data["residues"], 
                self.max_residues, 
                self.aa_to_idx
            )
            
            # Pad positions
            positions = np.zeros((self.max_residues, 3), dtype=np.float32)
            num_residues = len(struct_data["residues"])
            positions[:num_residues] = struct_data["positions"]
            
            # Create tensor dictionary
            processed_sample["residues"] = torch.tensor(encoded_residues)
            processed_sample["coordinates"] = torch.tensor(positions)
            processed_sample["mask"] = torch.tensor(res_mask)
            
            # Apply transform if provided
            if "protein_struct" in self.transform:
                processed_sample = self.transform["protein_struct"](processed_sample)
        
        # Process genomic data
        if "genomic" in self.modalities and "genomic" in sample:
            # Encode DNA sequence
            if "genomic" in sample:
                dna_seq = sample["genomic"]
                encoded_dna, dna_mask = self._encode_sequence(dna_seq, self.max_genomic_length, self.nuc_to_idx)
                
                # Convert to one-hot encoding
                one_hot = np.zeros((self.max_genomic_length, 5), dtype=np.float32)
                for i, idx in enumerate(encoded_dna):
                    if idx > 0:
                        one_hot[i, idx] = 1.0
                
                processed_sample["genomic_seq"] = torch.tensor(one_hot)
                processed_sample["genomic_mask"] = torch.tensor(dna_mask)
            
            # Process gene expression data
            if "expression" in sample:
                processed_sample["expression"] = torch.tensor(sample["expression"], dtype=torch.float32)
            
            # Apply transform if provided
            if "genomic" in self.transform:
                processed_sample = self.transform["genomic"](processed_sample)
        
        # Process target data
        if "target" in sample:
            processed_sample["target"] = torch.tensor([sample["target"]], dtype=torch.float32)
        
        return processed_sample


def create_scientific_dataloaders(
    data_type: str,
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    modalities: List[str] = None,
    max_samples: Optional[int] = None,
    train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for scientific data.
    
    Args:
        data_type: Type of data ("materials" or "bio")
        data_dir: Directory containing the dataset
        batch_size: Batch size for the data loaders
        num_workers: Number of workers for data loading
        modalities: List of modalities to load (domain-specific)
        max_samples: Maximum number of samples to load
        train_val_test_split: Ratios for train/val/test split
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Default modalities based on data type
    if modalities is None:
        if data_type == "materials":
            modalities = ["crystal", "spectral"]
        elif data_type == "bio":
            modalities = ["protein_seq", "protein_struct", "genomic"]
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    # Create datasets
    if data_type == "materials":
        train_dataset = MaterialsScienceDataset(
            data_dir=data_dir,
            split="train",
            modalities=modalities,
            max_samples=max_samples
        )
        
        val_dataset = MaterialsScienceDataset(
            data_dir=data_dir,
            split="val",
            modalities=modalities,
            max_samples=max_samples
        )
        
        test_dataset = MaterialsScienceDataset(
            data_dir=data_dir,
            split="test",
            modalities=modalities,
            max_samples=max_samples
        )
        
    elif data_type == "bio":
        train_dataset = BioinformaticsDataset(
            data_dir=data_dir,
            split="train",
            modalities=modalities,
            max_samples=max_samples
        )
        
        val_dataset = BioinformaticsDataset(
            data_dir=data_dir,
            split="val",
            modalities=modalities,
            max_samples=max_samples
        )
        
        test_dataset = BioinformaticsDataset(
            data_dir=data_dir,
            split="test",
            modalities=modalities,
            max_samples=max_samples
        )
        
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders for {data_type} data:")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    logger.info(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader