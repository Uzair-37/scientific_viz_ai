"""
Data loading and processing utilities for scientific multi-modal data.
"""
import os
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

class ScientificDataset(Dataset):
    """
    Multi-modal scientific dataset for training and evaluation.
    
    Supports loading of text, images, graphs, and numerical data.
    """
    
    def __init__(self, 
                 data_dir: str, 
                 modalities: List[str] = ["text", "image", "numerical"],
                 transform=None, 
                 max_samples: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the dataset
            modalities: List of modalities to load
            transform: Transforms to apply to the data
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.data_dir = data_dir
        self.modalities = modalities
        self.transform = transform
        self.max_samples = max_samples
        
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load all data samples from the data directory.
        
        Returns:
            List of data samples, each as a dictionary with modality keys
        """
        logger.info(f"Loading data from {self.data_dir}")
        
        # TODO: Implement actual data loading logic
        # This is a placeholder that creates dummy data
        samples = []
        for i in range(100):
            sample = {}
            
            if "text" in self.modalities:
                sample["text"] = f"Sample scientific text {i}"
                
            if "image" in self.modalities:
                # Dummy image as numpy array
                sample["image"] = np.random.rand(224, 224, 3)
                
            if "numerical" in self.modalities:
                # Dummy numerical features
                sample["numerical"] = np.random.rand(50)
                
            if "graph" in self.modalities:
                # Dummy graph structure
                sample["graph"] = {
                    "nodes": np.random.rand(10, 16),
                    "edges": np.random.randint(0, 10, (15, 2)),
                    "edge_attr": np.random.rand(15, 4)
                }
                
            samples.append(sample)
            
            if self.max_samples and len(samples) >= self.max_samples:
                break
                
        logger.info(f"Loaded {len(samples)} samples")
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.data[idx]
        
        # Apply transforms if provided
        if self.transform:
            sample = self.transform(sample)
            
        # Convert numpy arrays to tensors
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                sample[key] = torch.from_numpy(value).float()
                
        return sample


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    modalities: List[str] = ["text", "image", "numerical"],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for the data loaders
        modalities: List of modalities to load
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create the full dataset
    full_dataset = ScientificDataset(data_dir=data_dir, modalities=modalities)
    
    # Calculate sizes
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
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
    
    logger.info(f"Created data loaders with {train_size} training, {val_size} validation, and {test_size} test samples")
    
    return train_loader, val_loader, test_loader