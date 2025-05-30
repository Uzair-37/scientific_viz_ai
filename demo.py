#!/usr/bin/env python3
"""
Multi-Modal Generative AI for Scientific Visualization and Discovery Demo

This script demonstrates the capabilities of the scientific visualization and discovery system.
"""
import os
import argparse
import logging
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.multimodal_model import MultiModalTransformer, DiffusionModel
from models.scientific_adapters.materials_science import MaterialsScienceAdapter
from models.scientific_adapters.bioinformatics import BioinformaticsAdapter
from models.discovery import PatternDiscovery
from models.scientific_trainer import ScientificTrainer, UncertaintyLoss, ScientificMetrics, create_scientific_trainer
from visualization.viz_generator import ScientificVisualizer
from visualization.scientific_viz import ScientificDomainVisualizer, ScientificMaterialsVisualizer, ScientificBioinformaticsVisualizer
from data.scientific_data import MaterialsScienceDataset, BioinformaticsDataset, create_scientific_dataloaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def demo_materials_science(output_dir: str = "./output/materials", use_gpu: bool = True):
    """
    Demonstrate materials science capabilities.
    
    Args:
        output_dir: Directory to save outputs
        use_gpu: Whether to use GPU if available
    """
    logger.info("Running materials science demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data directory for synthetic data
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create materials science dataset
    train_dataset = MaterialsScienceDataset(
        data_dir=data_dir,
        split="train",
        modalities=["crystal", "spectral"],
        max_samples=100
    )
    
    # Sample a batch
    sample = train_dataset[0]
    
    # Visualize crystal structure
    logger.info("Visualizing crystal structure")
    visualizer = ScientificDomainVisualizer(output_dir=output_dir)
    materials_viz = visualizer.get_visualizer("materials")
    
    fig = materials_viz.visualize_crystal_structure(
        atom_types=sample["atom_types"],
        positions=sample["positions"],
        lattice=sample["lattice"],
        title="Example Crystal Structure",
        save_path=os.path.join(output_dir, "crystal_structure.png"),
        interactive=False,
        show_unit_cell=True
    )
    
    # Generate synthetic spectroscopy data
    xrd_data = np.random.rand(1024)
    # Add some peaks to make it look more realistic
    peak_positions = np.random.choice(range(100, 900), size=5, replace=False)
    for pos in peak_positions:
        xrd_data[pos-10:pos+10] = 0.2
        xrd_data[pos-5:pos+5] = 0.5
        xrd_data[pos-2:pos+2] = 0.8
        xrd_data[pos] = 1.0
    
    # Visualize spectroscopy
    logger.info("Visualizing spectroscopy data")
    fig = materials_viz.visualize_spectroscopy(
        spectrum=xrd_data,
        x_values=np.linspace(10, 80, len(xrd_data)),
        spectrum_type="XRD",
        title="Example X-ray Diffraction Pattern",
        save_path=os.path.join(output_dir, "xrd_spectrum.png"),
        interactive=False,
        show_peaks=True
    )
    
    # Generate synthetic property data
    feature_data = np.random.rand(100, 20)
    property_values = np.random.rand(100) * 5  # e.g., band gaps from 0-5 eV
    
    # Visualize property map
    logger.info("Visualizing property map")
    fig = materials_viz.visualize_property_map(
        property_values=property_values,
        feature_data=feature_data,
        property_name="Band Gap (eV)",
        title="Materials Band Gap Map",
        save_path=os.path.join(output_dir, "property_map.png"),
        interactive=False,
        reduction_method="pca",
        cluster_data=True,
        num_clusters=5
    )
    
    # Initialize model
    logger.info("Initializing materials science model")
    model = MaterialsScienceAdapter(
        embed_dim=256,
        crystal_hidden_dim=128,
        spectral_hidden_dim=128,
        property_hidden_dim=64,
        num_properties=1,
        uncertainty=True
    )
    model.to(device)
    
    # Create dataloaders for demo
    train_loader, val_loader, _ = create_scientific_dataloaders(
        data_type="materials",
        data_dir=data_dir,
        batch_size=16,
        num_workers=2,
        max_samples=100
    )
    
    # Initialize trainer
    logger.info("Setting up trainer")
    trainer = ScientificTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        uncertainty_aware=True
    )
    
    # Create loss function
    loss_fn = UncertaintyLoss(base_loss_fn="mse")
    
    # Create metrics
    metrics = {
        "mae": ScientificMetrics.mean_absolute_error,
        "rmse": ScientificMetrics.root_mean_squared_error,
        "r2": ScientificMetrics.r2_score
    }
    
    # Run a few epochs for demo
    logger.info("Running mini-training for demonstration")
    trainer.train(
        epochs=3,
        loss_fn=loss_fn,
        metrics=metrics,
        early_stopping_patience=5
    )
    
    # Plot training history
    trainer.plot_history(save_path=os.path.join(output_dir, "training_history.png"))
    
    logger.info("Materials science demo completed successfully")


def demo_bioinformatics(output_dir: str = "./output/bioinformatics", use_gpu: bool = True):
    """
    Demonstrate bioinformatics capabilities.
    
    Args:
        output_dir: Directory to save outputs
        use_gpu: Whether to use GPU if available
    """
    logger.info("Running bioinformatics demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data directory for synthetic data
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create bioinformatics dataset
    train_dataset = BioinformaticsDataset(
        data_dir=data_dir,
        split="train",
        modalities=["protein_seq", "protein_struct", "genomic"],
        max_samples=100
    )
    
    # Sample a batch
    sample = train_dataset[0]
    
    # Create visualizer
    visualizer = ScientificDomainVisualizer(output_dir=output_dir)
    bio_viz = visualizer.get_visualizer("bio")
    
    # Generate a demo protein sequence
    protein_seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    motifs = ["HKLRVDP", "FLASVST"]
    
    # Visualize protein sequence
    logger.info("Visualizing protein sequence")
    fig = bio_viz.visualize_protein_sequence(
        sequence=protein_seq,
        title="Hemoglobin Alpha Chain",
        save_path=os.path.join(output_dir, "protein_sequence.png"),
        show_properties=True,
        highlight_motifs=motifs,
        interactive=False
    )
    
    # Generate synthetic protein structure data
    num_residues = len(protein_seq)
    # Create a basic helix shape
    positions = np.zeros((num_residues, 3))
    for i in range(num_residues):
        angle = i * 100 * np.pi / 180
        positions[i, 0] = 5 * np.cos(angle)
        positions[i, 1] = 5 * np.sin(angle)
        positions[i, 2] = i * 1.5
    
    # Visualize protein structure
    logger.info("Visualizing protein structure")
    fig = bio_viz.visualize_protein_structure(
        residues=protein_seq,
        coordinates=positions,
        title="Protein Structure Example",
        save_path=os.path.join(output_dir, "protein_structure.png"),
        interactive=False,
        color_mode="residue_type",
        highlight_residues=[50, 51, 52, 53, 54]
    )
    
    # Generate synthetic genomic data
    genomic_seq = ''.join(np.random.choice(['A', 'C', 'G', 'T'], size=1000))
    
    # Visualize genomic data
    logger.info("Visualizing genomic data")
    fig = bio_viz.visualize_genomic_data(
        genomic_data={
            'sequence': genomic_seq,
            'expression': np.random.rand(50) * 10
        },
        title="Genomic Data Example",
        save_path=os.path.join(output_dir, "genomic_data.png"),
        interactive=False,
        genomic_region="Chromosome 1:1000-2000"
    )
    
    # Initialize model
    logger.info("Initializing bioinformatics model")
    model = BioinformaticsAdapter(
        embed_dim=256,
        protein_seq_hidden_dim=128,
        protein_struct_hidden_dim=128,
        genomic_hidden_dim=128,
        num_outputs=1,
        uncertainty=True
    )
    model.to(device)
    
    # Create dataloaders for demo
    train_loader, val_loader, _ = create_scientific_dataloaders(
        data_type="bio",
        data_dir=data_dir,
        batch_size=16,
        num_workers=2,
        max_samples=100
    )
    
    # Initialize trainer
    logger.info("Setting up trainer")
    trainer = create_scientific_trainer(
        model_type="bio",
        model_config={
            "embed_dim": 256,
            "protein_seq_hidden_dim": 128,
            "protein_struct_hidden_dim": 128,
            "genomic_hidden_dim": 128,
            "num_outputs": 1,
            "uncertainty": True
        },
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        uncertainty_aware=True
    )
    
    # Create loss function
    loss_fn = UncertaintyLoss(base_loss_fn="mse")
    
    # Create metrics
    metrics = {
        "mae": ScientificMetrics.mean_absolute_error,
        "rmse": ScientificMetrics.root_mean_squared_error,
        "r2": ScientificMetrics.r2_score
    }
    
    # Run a few epochs for demo
    logger.info("Running mini-training for demonstration")
    trainer.train(
        epochs=3,
        loss_fn=loss_fn,
        metrics=metrics,
        early_stopping_patience=5
    )
    
    # Plot training history
    trainer.plot_history(save_path=os.path.join(output_dir, "training_history.png"))
    
    logger.info("Bioinformatics demo completed successfully")


def demo_multimodal_fusion(output_dir: str = "./output/multimodal", use_gpu: bool = True):
    """
    Demonstrate multimodal fusion capabilities.
    
    Args:
        output_dir: Directory to save outputs
        use_gpu: Whether to use GPU if available
    """
    logger.info("Running multimodal fusion demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize multimodal transformer
    logger.info("Initializing multimodal transformer")
    model = MultiModalTransformer(
        modalities=["text", "image", "numerical", "graph"],
        modality_dims={
            "text": 768,  # BERT/DistilBERT dimension
            "image": 224*224*3,  # Example image size
            "numerical": 50,  # Example numerical features
            "graph": 16  # Example graph node features
        },
        embed_dim=512,
        num_encoder_layers=4,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.1
    )
    model.to(device)
    
    # Create random inputs for demonstration
    batch_size = 2
    inputs = {
        "text": torch.ones((batch_size, 10), dtype=torch.long, device=device),  # Tokenized text
        "image": torch.rand((batch_size, 3, 224, 224), device=device),  # RGB images
        "numerical": torch.rand((batch_size, 50), device=device),  # Numerical features
        "graph": torch.rand((batch_size, 10, 16), device=device)  # Graph node features
    }
    
    # Run forward pass to demonstrate multimodal fusion
    logger.info("Running multimodal fusion")
    with torch.no_grad():
        outputs = model(inputs)
    
    # Print output shapes
    for key, tensor in outputs.items():
        if isinstance(tensor, dict):
            logger.info(f"{key}: {len(tensor)} modalities")
        else:
            logger.info(f"{key}: {tensor.shape}")
    
    # Initialize diffusion model
    logger.info("Initializing diffusion model")
    diffusion = DiffusionModel(
        input_dim=512,  # Match the multimodal model's output dimension
        output_channels=3,
        output_size=256,
        time_embedding_dim=256,
        base_channels=64,
        channel_multipliers=[1, 2, 4, 8],
        num_res_blocks=2
    )
    diffusion.to(device)
    
    # Generate a visualization using the diffusion model
    logger.info("Generating visualization with diffusion model")
    visualizer = ScientificVisualizer(diffusion_model=diffusion, device=device)
    
    # Use the fused representation as conditioning
    condition = outputs["fused_repr"]
    
    # Create a simplified time and condition tensors for demonstration
    # (in a real implementation, proper diffusion sampling would be used)
    t = torch.ones((batch_size,), device=device) * 0.5
    x = torch.randn((batch_size, 3, 256, 256), device=device)
    
    # Run a single diffusion step for demonstration
    with torch.no_grad():
        generated = diffusion(x, t, condition)
    
    logger.info(f"Generated output shape: {generated.shape}")
    
    # Convert to numpy and save an example visualization
    img = generated[0].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Generated Scientific Visualization")
    plt.savefig(os.path.join(output_dir, "generated_visualization.png"), dpi=300, bbox_inches="tight")
    
    logger.info("Multimodal fusion demo completed successfully")


def demo_pattern_discovery(output_dir: str = "./output/discovery", use_gpu: bool = True):
    """
    Demonstrate pattern discovery capabilities.
    
    Args:
        output_dir: Directory to save outputs
        use_gpu: Whether to use GPU if available
    """
    logger.info("Running pattern discovery demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize pattern discovery module
    discovery = PatternDiscovery(device=device)
    
    # Generate synthetic high-dimensional data with clusters
    logger.info("Generating synthetic data with clusters")
    n_samples = 1000
    n_features = 50
    n_clusters = 5
    
    # Create cluster centers
    centers = np.random.rand(n_clusters, n_features) * 2 - 1
    
    # Generate samples around centers
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        cluster = i % n_clusters
        X[i] = centers[cluster] + np.random.randn(n_features) * 0.1
        y[i] = cluster
    
    # Add some noise features
    X[:, -10:] = np.random.randn(n_samples, 10)
    
    # Perform dimension reduction
    logger.info("Performing dimension reduction")
    methods = ["pca", "tsne", "umap"]
    reduced_data = {}
    
    for method in methods:
        try:
            reduced = discovery.dimension_reduction(X, method=method, n_components=2)
            reduced_data[method] = reduced
            logger.info(f"Dimension reduction with {method} complete")
        except Exception as e:
            logger.warning(f"Dimension reduction with {method} failed: {str(e)}")
    
    # Create a visualization comparing the methods
    plt.figure(figsize=(15, 5))
    
    for i, (method, data) in enumerate(reduced_data.items()):
        plt.subplot(1, len(reduced_data), i+1)
        scatter = plt.scatter(data[:, 0], data[:, 1], c=y, cmap='viridis', alpha=0.8)
        plt.title(f"{method.upper()} Projection")
        plt.colorbar(scatter, label='Cluster')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dimension_reduction_comparison.png"), dpi=300, bbox_inches="tight")
    
    # Perform clustering
    logger.info("Performing clustering")
    methods = ["kmeans", "dbscan", "agglomerative"]
    clustering_results = {}
    
    for method in methods:
        try:
            labels, score = discovery.cluster_data(X, method=method, n_clusters=n_clusters)
            clustering_results[method] = {
                "labels": labels,
                "score": score
            }
            logger.info(f"Clustering with {method} complete, score: {score:.4f}")
        except Exception as e:
            logger.warning(f"Clustering with {method} failed: {str(e)}")
    
    # Visualize clustering results
    plt.figure(figsize=(15, 5))
    
    # Use PCA for visualization
    pca_data = reduced_data.get("pca", discovery.dimension_reduction(X, method="pca", n_components=2))
    
    for i, (method, result) in enumerate(clustering_results.items()):
        plt.subplot(1, len(clustering_results), i+1)
        scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=result["labels"], cmap='tab10', alpha=0.8)
        plt.title(f"{method.capitalize()} Clustering")
        plt.colorbar(scatter, label='Cluster')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clustering_comparison.png"), dpi=300, bbox_inches="tight")
    
    # Find feature importance
    logger.info("Finding feature importance")
    importances = discovery.find_feature_importance(X, y)
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    top_k = 20
    sorted_idx = np.argsort(importances)[::-1][:top_k]
    
    plt.bar(range(top_k), importances[sorted_idx])
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.xticks(range(top_k), sorted_idx)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300, bbox_inches="tight")
    
    # Find anomalies
    logger.info("Finding anomalies")
    anomalies = discovery.find_anomalies(X, method="isolation_forest", contamination=0.05)
    
    # Visualize anomalies
    plt.figure(figsize=(8, 8))
    plt.scatter(pca_data[~anomalies, 0], pca_data[~anomalies, 1], c='blue', alpha=0.5, label='Normal')
    plt.scatter(pca_data[anomalies, 0], pca_data[anomalies, 1], c='red', alpha=0.8, label='Anomaly')
    plt.title('Anomaly Detection')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "anomaly_detection.png"), dpi=300, bbox_inches="tight")
    
    logger.info("Pattern discovery demo completed successfully")


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description="Multi-Modal Generative AI for Scientific Visualization and Discovery Demo"
    )
    
    parser.add_argument("--demo", type=str, choices=["all", "materials", "bio", "multimodal", "discovery"],
                      default="all", help="Demo to run")
    parser.add_argument("--output", type=str, default="./demo_output",
                      help="Output directory for demo results")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Running demo(s): {args.demo}")
    logger.info(f"Output directory: {output_dir}")
    
    # Run selected demos
    if args.demo in ["all", "materials"]:
        demo_materials_science(
            output_dir=os.path.join(output_dir, "materials"),
            use_gpu=not args.no_gpu
        )
    
    if args.demo in ["all", "bio"]:
        demo_bioinformatics(
            output_dir=os.path.join(output_dir, "bioinformatics"),
            use_gpu=not args.no_gpu
        )
    
    if args.demo in ["all", "multimodal"]:
        demo_multimodal_fusion(
            output_dir=os.path.join(output_dir, "multimodal"),
            use_gpu=not args.no_gpu
        )
    
    if args.demo in ["all", "discovery"]:
        demo_pattern_discovery(
            output_dir=os.path.join(output_dir, "discovery"),
            use_gpu=not args.no_gpu
        )
    
    logger.info("Demo completed successfully")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()