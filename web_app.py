#!/usr/bin/env python3
"""
Web application for the Scientific Visualization AI project.

This script provides a web interface for users to upload scientific data,
generate visualizations, and discover patterns using the trained models.
"""
import os
import io
import argparse
import logging
import tempfile
from datetime import datetime
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import streamlit as st

try:
    from models.multimodal_model import MultiModalTransformer, DiffusionModel
    from models.scientific_adapters.materials_science import MaterialsScienceAdapter
    from models.scientific_adapters.bioinformatics import BioinformaticsAdapter
    from models.discovery import PatternDiscovery
except ImportError:
    # Fallback when models are not available
    MultiModalTransformer = None
    DiffusionModel = None
    MaterialsScienceAdapter = None
    BioinformaticsAdapter = None
    PatternDiscovery = None
from visualization.scientific_viz import ScientificDomainVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Global variables to store models and visualizers
models = {}
visualizers = {}
discovery = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(model_dir, use_gpu=True):
    """Load pretrained models."""
    global models, visualizers, discovery, device
    
    if not use_gpu:
        device = torch.device("cpu")
    
    logger.info(f"Loading models from {model_dir} to {device}")
    
    # Initialize pattern discovery
    discovery = PatternDiscovery(device=device)
    
    # Initialize domain visualizers
    visualizers["materials"] = ScientificDomainVisualizer(
        discovery=discovery,
        output_dir="./temp_viz"
    ).get_visualizer("materials")
    
    visualizers["bio"] = ScientificDomainVisualizer(
        discovery=discovery,
        output_dir="./temp_viz"
    ).get_visualizer("bio")
    
    # Check for available model checkpoints
    materials_path = os.path.join(model_dir, "materials", "best_model.pth")
    bio_path = os.path.join(model_dir, "bioinformatics", "best_model.pth")
    diffusion_path = os.path.join(model_dir, "diffusion", "best_model.pth")
    
    # Initialize materials science model
    models["materials"] = MaterialsScienceAdapter(
        embed_dim=256,
        crystal_hidden_dim=128,
        spectral_hidden_dim=128,
        property_hidden_dim=64,
        num_properties=1,
        uncertainty=True
    )
    
    # Initialize bioinformatics model
    models["bio"] = BioinformaticsAdapter(
        embed_dim=256,
        protein_seq_hidden_dim=128,
        protein_struct_hidden_dim=128,
        genomic_hidden_dim=128,
        num_outputs=1,
        uncertainty=True
    )
    
    # Initialize diffusion model
    models["diffusion"] = DiffusionModel(
        input_dim=256,
        output_channels=3,
        output_size=256,
        time_embedding_dim=128
    )
    
    # Load pretrained weights if available
    if os.path.exists(materials_path):
        logger.info(f"Loading materials model from {materials_path}")
        models["materials"].load_state_dict(
            torch.load(materials_path, map_location=device)["model_state_dict"]
        )
    else:
        logger.warning(f"No pretrained materials model found at {materials_path}")
    
    if os.path.exists(bio_path):
        logger.info(f"Loading bioinformatics model from {bio_path}")
        models["bio"].load_state_dict(
            torch.load(bio_path, map_location=device)["model_state_dict"]
        )
    else:
        logger.warning(f"No pretrained bioinformatics model found at {bio_path}")
    
    if os.path.exists(diffusion_path):
        logger.info(f"Loading diffusion model from {diffusion_path}")
        models["diffusion"].load_state_dict(
            torch.load(diffusion_path, map_location=device)["model_state_dict"]
        )
    else:
        logger.warning(f"No pretrained diffusion model found at {diffusion_path}")
    
    # Move models to device and set to evaluation mode
    for name, model in models.items():
        model.to(device)
        model.eval()
    
    logger.info("Models loaded successfully")


def _parse_crystal_structure(uploaded_file):
    """Parse crystal structure data from various file formats."""
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    # For demo purposes, we'll create synthetic data
    # In a real implementation, use libraries like ASE or pymatgen to parse these files
    if file_ext in ['.cif', '.xyz', '.poscar', '.vasp']:
        # Create synthetic crystal structure
        num_atoms = 50
        atom_types = np.random.randint(1, 100, size=num_atoms)
        positions = np.random.rand(num_atoms, 3) * 10
        lattice = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        
        return {
            "atom_types": atom_types,
            "positions": positions,
            "lattice": lattice,
            "format": file_ext[1:]
        }
    else:
        raise ValueError(f"Unsupported crystal structure format: {file_ext}")


def _parse_spectroscopy_data(uploaded_file):
    """Parse spectroscopy data from various file formats."""
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_ext in ['.csv', '.txt', '.xy', '.xye']:
        # Read data assuming x, y format
        try:
            df = pd.read_csv(uploaded_file, header=None, delim_whitespace=True)
            x_values = df.iloc[:, 0].values
            intensity = df.iloc[:, 1].values
            
            return {
                "x_values": x_values,
                "spectrum": intensity,
                "format": file_ext[1:]
            }
        except Exception as e:
            # If failed, create synthetic data
            logger.warning(f"Failed to parse spectroscopy data: {str(e)}")
            x_values = np.linspace(0, 90, 1000)
            intensity = np.zeros_like(x_values)
            
            # Add some peaks
            for pos in [20, 30, 45, 60, 75]:
                peak_width = np.random.uniform(0.5, 2.0)
                intensity += np.exp(-(x_values - pos)**2 / (2 * peak_width**2))
            
            # Add noise
            intensity += np.random.randn(len(x_values)) * 0.05
            
            return {
                "x_values": x_values,
                "spectrum": intensity,
                "format": file_ext[1:]
            }
    else:
        raise ValueError(f"Unsupported spectroscopy data format: {file_ext}")


def _parse_protein_data(uploaded_file):
    """Parse protein sequence or structure data."""
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_ext in ['.pdb', '.cif', '.mmcif']:
        # For demo purposes, create synthetic protein structure
        # In a real implementation, use libraries like Biopython to parse these files
        sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
        num_residues = len(sequence)
        
        # Create a basic helix shape
        positions = np.zeros((num_residues, 3))
        for i in range(num_residues):
            angle = i * 100 * np.pi / 180
            positions[i, 0] = 5 * np.cos(angle)
            positions[i, 1] = 5 * np.sin(angle)
            positions[i, 2] = i * 1.5
            
        return {
            "sequence": sequence,
            "positions": positions,
            "format": file_ext[1:]
        }
    elif file_ext in ['.fasta', '.fa', '.seq', '.txt']:
        # Read sequence from file
        try:
            content = uploaded_file.read().decode('utf-8')
            lines = content.strip().split('\n')
            
            sequence = ""
            for line in lines:
                if not line.startswith('>'):  # Skip FASTA header lines
                    sequence += line.strip()
            
            return {
                "sequence": sequence,
                "format": file_ext[1:]
            }
        except Exception as e:
            logger.warning(f"Failed to parse protein sequence data: {str(e)}")
            # Return a default sequence
            sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
            return {
                "sequence": sequence,
                "format": "txt"
            }
    else:
        raise ValueError(f"Unsupported protein data format: {file_ext}")


def _parse_genomic_data(uploaded_file):
    """Parse genomic data from various file formats."""
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_ext in ['.fasta', '.fa', '.fna', '.ffn', '.txt']:
        # Read genomic sequence from file
        try:
            content = uploaded_file.read().decode('utf-8')
            lines = content.strip().split('\n')
            
            sequence = ""
            for line in lines:
                if not line.startswith('>'):  # Skip FASTA header lines
                    sequence += line.strip()
            
            return {
                "sequence": sequence,
                "format": file_ext[1:]
            }
        except Exception as e:
            logger.warning(f"Failed to parse genomic sequence data: {str(e)}")
            # Return a default sequence
            nucleotides = ['A', 'C', 'G', 'T']
            sequence = ''.join(np.random.choice(nucleotides, size=500))
            return {
                "sequence": sequence,
                "format": "txt"
            }
    elif file_ext in ['.csv', '.tsv', '.txt', '.expr']:
        # Read gene expression data
        try:
            df = pd.read_csv(uploaded_file)
            gene_names = df.iloc[:, 0].values if df.shape[1] > 1 else [f"Gene_{i}" for i in range(df.shape[0])]
            expression = df.iloc[:, 1].values if df.shape[1] > 1 else df.iloc[:, 0].values
            
            return {
                "gene_names": gene_names,
                "expression": expression,
                "format": file_ext[1:]
            }
        except Exception as e:
            logger.warning(f"Failed to parse gene expression data: {str(e)}")
            # Return default expression data
            num_genes = 50
            gene_names = [f"Gene_{i}" for i in range(num_genes)]
            expression = np.random.rand(num_genes) * 10
            
            return {
                "gene_names": gene_names,
                "expression": expression,
                "format": "csv"
            }
    else:
        raise ValueError(f"Unsupported genomic data format: {file_ext}")


def _parse_numerical_data(uploaded_file):
    """Parse numerical data from various file formats."""
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_ext in ['.csv', '.tsv', '.txt', '.data']:
        # Read numerical data
        try:
            if file_ext == '.tsv':
                df = pd.read_csv(uploaded_file, sep='\t')
            else:
                df = pd.read_csv(uploaded_file)
            
            # Get column names
            feature_names = df.columns.tolist()
            
            # Convert to numpy array
            data = df.values
            
            return {
                "data": data,
                "feature_names": feature_names,
                "format": file_ext[1:]
            }
        except Exception as e:
            logger.warning(f"Failed to parse numerical data: {str(e)}")
            # Return default numerical data
            num_samples = 100
            num_features = this 
            data = np.random.rand(num_samples, 10)
            feature_names = [f"Feature_{i}" for i in range(10)]
            
            return {
                "data": data,
                "feature_names": feature_names,
                "format": "csv"
            }
    else:
        raise ValueError(f"Unsupported numerical data format: {file_ext}")


def materials_science_app():
    """Streamlit app for materials science domain."""
    st.title("Materials Science Visualization")
    
    st.write("""
    This application allows you to upload materials science data (crystal structures, spectroscopy, etc.) 
    and generate insightful visualizations. You can also discover patterns and predict properties.
    """)
    
    # Sidebar for upload and options
    st.sidebar.header("Data Upload")
    
    upload_type = st.sidebar.selectbox(
        "Select data type",
        ["Crystal Structure", "Spectroscopy", "Both"]
    )
    
    uploaded_data = {}
    
    if upload_type in ["Crystal Structure", "Both"]:
        crystal_file = st.sidebar.file_uploader(
            "Upload crystal structure (CIF, XYZ, POSCAR)",
            type=["cif", "xyz", "poscar", "vasp"],
            key="crystal_upload"
        )
        
        if crystal_file:
            try:
                uploaded_data["crystal"] = _parse_crystal_structure(crystal_file)
                st.sidebar.success(f"Crystal structure loaded: {crystal_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error parsing crystal structure: {str(e)}")
    
    if upload_type in ["Spectroscopy", "Both"]:
        spectroscopy_file = st.sidebar.file_uploader(
            "Upload spectroscopy data (CSV, TXT, XY)",
            type=["csv", "txt", "xy", "xye"],
            key="spectroscopy_upload"
        )
        
        if spectroscopy_file:
            try:
                uploaded_data["spectroscopy"] = _parse_spectroscopy_data(spectroscopy_file)
                st.sidebar.success(f"Spectroscopy data loaded: {spectroscopy_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error parsing spectroscopy data: {str(e)}")
    
    # Generate demo data if nothing is uploaded
    if not uploaded_data and st.sidebar.button("Use Demo Data"):
        # Generate demo crystal structure
        num_atoms = 50
        atom_types = np.random.randint(1, 100, size=num_atoms)
        positions = np.random.rand(num_atoms, 3) * 10
        lattice = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        
        uploaded_data["crystal"] = {
            "atom_types": atom_types,
            "positions": positions,
            "lattice": lattice,
            "format": "demo"
        }
        
        # Generate demo spectroscopy data
        x_values = np.linspace(0, 90, 1000)
        intensity = np.zeros_like(x_values)
        
        # Add some peaks
        for pos in [20, 30, 45, 60, 75]:
            peak_width = np.random.uniform(0.5, 2.0)
            intensity += np.exp(-(x_values - pos)**2 / (2 * peak_width**2))
        
        # Add noise
        intensity += np.random.randn(len(x_values)) * 0.05
        
        uploaded_data["spectroscopy"] = {
            "x_values": x_values,
            "spectrum": intensity,
            "format": "demo"
        }
        
        st.sidebar.success("Demo data loaded!")
    
    # Visualization options
    st.sidebar.header("Visualization Options")
    
    # Show visualizations if data is uploaded
    if uploaded_data:
        if "crystal" in uploaded_data:
            st.header("Crystal Structure Visualization")
            
            with st.expander("Crystal Structure Options", expanded=True):
                show_unit_cell = st.checkbox("Show Unit Cell", value=True)
                atom_size = st.slider("Atom Size", min_value=10, max_value=100, value=50, step=5)
                
                col1, col2 = st.columns(2)
                with col1:
                    visualization_type = st.selectbox(
                        "Visualization Type",
                        ["Static", "Interactive"],
                        key="crystal_viz_type"
                    )
                with col2:
                    color_mode = st.selectbox(
                        "Coloring Scheme",
                        ["Element", "Custom"],
                        key="crystal_color_mode"
                    )
            
            # Create visualization
            try:
                crystal_data = uploaded_data["crystal"]
                
                # Convert to torch tensors if needed
                if not isinstance(crystal_data["atom_types"], torch.Tensor):
                    crystal_data["atom_types"] = torch.tensor(crystal_data["atom_types"])
                    crystal_data["positions"] = torch.tensor(crystal_data["positions"])
                    crystal_data["lattice"] = torch.tensor(crystal_data["lattice"])
                
                fig = visualizers["materials"].visualize_crystal_structure(
                    atom_types=crystal_data["atom_types"],
                    positions=crystal_data["positions"],
                    lattice=crystal_data["lattice"],
                    title="Crystal Structure",
                    interactive=(visualization_type == "Interactive"),
                    show_unit_cell=show_unit_cell,
                    atom_size_scale=atom_size
                )
                
                if visualization_type == "Interactive":
                    st_plotly(fig)
                else:
                    st.pyplot(fig)
                
                # Save visualization
                save_viz = st.button("Save Visualization", key="save_crystal_viz")
                if save_viz:
                    if visualization_type == "Interactive":
                        save_path = os.path.join(tempfile.gettempdir(), "crystal_structure.html")
                        fig.write_html(save_path)
                    else:
                        save_path = os.path.join(tempfile.gettempdir(), "crystal_structure.png")
                        fig.savefig(save_path, dpi=300, bbox_inches="tight")
                        
                    st.success(f"Visualization saved to {save_path}")
            
            except Exception as e:
                st.error(f"Error creating crystal structure visualization: {str(e)}")
        
        if "spectroscopy" in uploaded_data:
            st.header("Spectroscopy Visualization")
            
            with st.expander("Spectroscopy Options", expanded=True):
                spectrum_type = st.selectbox(
                    "Spectrum Type",
                    ["XRD", "XPS", "Raman", "IR", "UV-Vis"],
                    key="spectrum_type"
                )
                
                show_peaks = st.checkbox("Show Peaks", value=True)
                peak_threshold = st.slider("Peak Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
                
                visualization_type = st.selectbox(
                    "Visualization Type",
                    ["Static", "Interactive"],
                    key="spectroscopy_viz_type"
                )
            
            # Create visualization
            try:
                spectroscopy_data = uploaded_data["spectroscopy"]
                
                # Convert to torch tensors if needed
                if not isinstance(spectroscopy_data["spectrum"], torch.Tensor):
                    spectroscopy_data["spectrum"] = torch.tensor(spectroscopy_data["spectrum"])
                    spectroscopy_data["x_values"] = torch.tensor(spectroscopy_data["x_values"])
                
                fig = visualizers["materials"].visualize_spectroscopy(
                    spectrum=spectroscopy_data["spectrum"],
                    x_values=spectroscopy_data["x_values"],
                    spectrum_type=spectrum_type,
                    title=f"{spectrum_type} Spectrum",
                    interactive=(visualization_type == "Interactive"),
                    show_peaks=show_peaks,
                    peak_threshold=peak_threshold
                )
                
                if visualization_type == "Interactive":
                    st_plotly(fig)
                else:
                    st.pyplot(fig)
                
                # Save visualization
                save_viz = st.button("Save Visualization", key="save_spectroscopy_viz")
                if save_viz:
                    if visualization_type == "Interactive":
                        save_path = os.path.join(tempfile.gettempdir(), f"{spectrum_type.lower()}_spectrum.html")
                        fig.write_html(save_path)
                    else:
                        save_path = os.path.join(tempfile.gettempdir(), f"{spectrum_type.lower()}_spectrum.png")
                        fig.savefig(save_path, dpi=300, bbox_inches="tight")
                        
                    st.success(f"Visualization saved to {save_path}")
            
            except Exception as e:
                st.error(f"Error creating spectroscopy visualization: {str(e)}")
        
        # Property prediction if both crystal and spectroscopy data available
        if "crystal" in uploaded_data and "spectroscopy" in uploaded_data and "materials" in models:
            st.header("Property Prediction")
            
            with st.expander("Prediction Options", expanded=True):
                property_type = st.selectbox(
                    "Property to Predict",
                    ["Band Gap", "Formation Energy", "Elastic Modulus", "Thermal Conductivity"],
                    key="property_type"
                )
                
                run_prediction = st.button("Run Prediction")
            
            if run_prediction:
                try:
                    # Prepare input data
                    crystal_data = uploaded_data["crystal"]
                    spectroscopy_data = uploaded_data["spectroscopy"]
                    
                    # Convert to torch tensors and add batch dimension
                    atom_types = torch.tensor(crystal_data["atom_types"]).unsqueeze(0).to(device)
                    positions = torch.tensor(crystal_data["positions"]).unsqueeze(0).to(device)
                    lattice = torch.tensor(crystal_data["lattice"]).unsqueeze(0).to(device)
                    mask = torch.ones_like(atom_types, dtype=torch.float32).to(device)
                    
                    spectrum = torch.tensor(spectroscopy_data["spectrum"], dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Run model inference
                    model = models["materials"]
                    model.eval()
                    
                    with torch.no_grad():
                        # Prepare inputs for the model
                        crystal_inputs = {
                            "atom_types": atom_types,
                            "positions": positions,
                            "lattice": lattice,
                            "mask": mask
                        }
                        
                        # Run inference
                        outputs = model(
                            crystal_data=crystal_inputs,
                            spectral_data=spectrum
                        )
                    
                    # Extract predictions
                    if "property_mean" in outputs and "property_var" in outputs:
                        # Model with uncertainty
                        mean = outputs["property_mean"].cpu().numpy().flatten()[0]
                        var = outputs["property_var"].cpu().numpy().flatten()[0]
                        std = np.sqrt(var)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                label=f"Predicted {property_type}",
                                value=f"{mean:.3f}"
                            )
                        with col2:
                            st.metric(
                                label="Uncertainty (Std)",
                                value=f"{std:.3f}"
                            )
                        
                        # Create visualization of the prediction with uncertainty
                        fig, ax = plt.subplots(figsize=(8, 4))
                        
                        # Plot confidence interval
                        x = np.array([mean - 2*std, mean - std, mean, mean + std, mean + 2*std])
                        y = np.array([0.05, 0.32, 1.0, 0.32, 0.05])
                        ax.fill_between(x, y, alpha=0.3)
                        ax.axvline(x=mean, color='red', linestyle='-', linewidth=2)
                        
                        ax.set_xlabel(property_type)
                        ax.set_ylabel("Probability Density")
                        ax.set_title(f"Predicted {property_type} with Uncertainty")
                        
                        st.pyplot(fig)
                        
                    else:
                        # Model without uncertainty
                        prediction = outputs["properties"].cpu().numpy().flatten()[0]
                        
                        # Display result
                        st.metric(
                            label=f"Predicted {property_type}",
                            value=f"{prediction:.3f}"
                        )
                
                except Exception as e:
                    st.error(f"Error during property prediction: {str(e)}")
                    st.exception(e)
    
    else:
        st.info("Please upload materials science data or use the demo data to get started.")


def bioinformatics_app():
    """Streamlit app for bioinformatics domain."""
    st.title("Bioinformatics Visualization")
    
    st.write("""
    This application allows you to upload bioinformatics data (protein sequences, structures, genomic data) 
    and generate insightful visualizations. You can also discover patterns and predict properties.
    """)
    
    # Sidebar for upload and options
    st.sidebar.header("Data Upload")
    
    upload_type = st.sidebar.selectbox(
        "Select data type",
        ["Protein Sequence", "Protein Structure", "Genomic Data", "Multiple Types"]
    )
    
    uploaded_data = {}
    
    if upload_type in ["Protein Sequence", "Multiple Types"]:
        protein_seq_file = st.sidebar.file_uploader(
            "Upload protein sequence (FASTA, TXT)",
            type=["fasta", "fa", "seq", "txt"],
            key="protein_seq_upload"
        )
        
        if protein_seq_file:
            try:
                uploaded_data["protein_seq"] = _parse_protein_data(protein_seq_file)
                st.sidebar.success(f"Protein sequence loaded: {protein_seq_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error parsing protein sequence: {str(e)}")
    
    if upload_type in ["Protein Structure", "Multiple Types"]:
        protein_struct_file = st.sidebar.file_uploader(
            "Upload protein structure (PDB, CIF)",
            type=["pdb", "cif", "mmcif"],
            key="protein_struct_upload"
        )
        
        if protein_struct_file:
            try:
                uploaded_data["protein_struct"] = _parse_protein_data(protein_struct_file)
                st.sidebar.success(f"Protein structure loaded: {protein_struct_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error parsing protein structure: {str(e)}")
    
    if upload_type in ["Genomic Data", "Multiple Types"]:
        genomic_file = st.sidebar.file_uploader(
            "Upload genomic data (FASTA, CSV, TXT)",
            type=["fasta", "fa", "fna", "ffn", "csv", "tsv", "txt", "expr"],
            key="genomic_upload"
        )
        
        if genomic_file:
            try:
                uploaded_data["genomic"] = _parse_genomic_data(genomic_file)
                st.sidebar.success(f"Genomic data loaded: {genomic_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error parsing genomic data: {str(e)}")
    
    # Generate demo data if nothing is uploaded
    if not uploaded_data and st.sidebar.button("Use Demo Data"):
        # Demo protein sequence
        uploaded_data["protein_seq"] = {
            "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
            "format": "demo"
        }
        
        # Demo protein structure
        num_residues = len(uploaded_data["protein_seq"]["sequence"])
        positions = np.zeros((num_residues, 3))
        for i in range(num_residues):
            angle = i * 100 * np.pi / 180
            positions[i, 0] = 5 * np.cos(angle)
            positions[i, 1] = 5 * np.sin(angle)
            positions[i, 2] = i * 1.5
            
        uploaded_data["protein_struct"] = {
            "sequence": uploaded_data["protein_seq"]["sequence"],
            "positions": positions,
            "format": "demo"
        }
        
        # Demo genomic data
        nucleotides = ['A', 'C', 'G', 'T']
        sequence = ''.join(np.random.choice(nucleotides, size=500))
        
        uploaded_data["genomic"] = {
            "sequence": sequence,
            "gene_names": [f"Gene_{i}" for i in range(50)],
            "expression": np.random.rand(50) * 10,
            "format": "demo"
        }
        
        st.sidebar.success("Demo data loaded!")
    
    # Visualization options
    st.sidebar.header("Visualization Options")
    
    # Show visualizations if data is uploaded
    if uploaded_data:
        if "protein_seq" in uploaded_data:
            st.header("Protein Sequence Visualization")
            
            with st.expander("Sequence Options", expanded=True):
                show_properties = st.checkbox("Show Properties", value=True)
                
                motifs_input = st.text_input(
                    "Highlight Motifs (comma separated)",
                    value="",
                    help="Enter motifs to highlight, separated by commas"
                )
                
                highlight_motifs = [m.strip() for m in motifs_input.split(',')] if motifs_input else None
                
                visualization_type = st.selectbox(
                    "Visualization Type",
                    ["Static", "Interactive"],
                    key="protein_seq_viz_type"
                )
            
            # Create visualization
            try:
                protein_seq = uploaded_data["protein_seq"]["sequence"]
                
                fig = visualizers["bio"].visualize_protein_sequence(
                    sequence=protein_seq,
                    title="Protein Sequence",
                    show_properties=show_properties,
                    highlight_motifs=highlight_motifs,
                    interactive=(visualization_type == "Interactive")
                )
                
                if visualization_type == "Interactive":
                    st_plotly(fig)
                else:
                    st.pyplot(fig)
                
                # Save visualization
                save_viz = st.button("Save Visualization", key="save_protein_seq_viz")
                if save_viz:
                    if visualization_type == "Interactive":
                        save_path = os.path.join(tempfile.gettempdir(), "protein_sequence.html")
                        fig.write_html(save_path)
                    else:
                        save_path = os.path.join(tempfile.gettempdir(), "protein_sequence.png")
                        fig.savefig(save_path, dpi=300, bbox_inches="tight")
                        
                    st.success(f"Visualization saved to {save_path}")
            
            except Exception as e:
                st.error(f"Error creating protein sequence visualization: {str(e)}")
        
        if "protein_struct" in uploaded_data:
            st.header("Protein Structure Visualization")
            
            with st.expander("Structure Options", expanded=True):
                color_mode = st.selectbox(
                    "Coloring Scheme",
                    ["Residue Type", "Secondary Structure", "Chain"],
                    key="protein_struct_color_mode"
                )
                
                highlight_input = st.text_input(
                    "Highlight Residues (comma separated indices)",
                    value="",
                    help="Enter residue indices to highlight, separated by commas"
                )
                
                highlight_residues = [int(idx.strip()) for idx in highlight_input.split(',')] if highlight_input else None
                
                visualization_type = st.selectbox(
                    "Visualization Type",
                    ["Static", "Interactive"],
                    key="protein_struct_viz_type"
                )
            
            # Create visualization
            try:
                protein_struct = uploaded_data["protein_struct"]
                
                # Map color mode from UI to function parameter
                color_mode_param = "residue_type"
                if color_mode == "Secondary Structure":
                    color_mode_param = "secondary"
                elif color_mode == "Chain":
                    color_mode_param = "chain"
                
                # Get sequence if present, otherwise get it from protein_seq if available
                if "sequence" in protein_struct:
                    sequence = protein_struct["sequence"]
                elif "protein_seq" in uploaded_data:
                    sequence = uploaded_data["protein_seq"]["sequence"]
                else:
                    # Default to "A" for all residues
                    sequence = "A" * protein_struct["positions"].shape[0]
                
                fig = visualizers["bio"].visualize_protein_structure(
                    residues=sequence,
                    coordinates=protein_struct["positions"],
                    title="Protein Structure",
                    highlight_residues=highlight_residues,
                    color_mode=color_mode_param,
                    interactive=(visualization_type == "Interactive")
                )
                
                if visualization_type == "Interactive":
                    st_plotly(fig)
                else:
                    st.pyplot(fig)
                
                # Save visualization
                save_viz = st.button("Save Visualization", key="save_protein_struct_viz")
                if save_viz:
                    if visualization_type == "Interactive":
                        save_path = os.path.join(tempfile.gettempdir(), "protein_structure.html")
                        fig.write_html(save_path)
                    else:
                        save_path = os.path.join(tempfile.gettempdir(), "protein_structure.png")
                        fig.savefig(save_path, dpi=300, bbox_inches="tight")
                        
                    st.success(f"Visualization saved to {save_path}")
            
            except Exception as e:
                st.error(f"Error creating protein structure visualization: {str(e)}")
                st.exception(e)
        
        if "genomic" in uploaded_data:
            st.header("Genomic Data Visualization")
            
            with st.expander("Genomic Options", expanded=True):
                genomic_viz_type = st.selectbox(
                    "Visualization Type",
                    ["Sequence", "Expression", "Combined"],
                    key="genomic_viz_type"
                )
                
                genomic_region = st.text_input(
                    "Genomic Region",
                    value="",
                    help="Enter a description of the genomic region (optional)"
                )
                
                interactive_genomic = st.checkbox("Interactive Visualization", value=False, key="interactive_genomic")
            
            # Create visualization
            try:
                genomic_data = uploaded_data["genomic"]
                
                # Prepare visualization data
                viz_data = {}
                
                if "sequence" in genomic_data and genomic_viz_type in ["Sequence", "Combined"]:
                    viz_data["sequence"] = genomic_data["sequence"]
                
                if "expression" in genomic_data and "gene_names" in genomic_data and genomic_viz_type in ["Expression", "Combined"]:
                    viz_data["expression"] = genomic_data["expression"]
                    viz_data["gene_names"] = genomic_data["gene_names"]
                
                if viz_data:
                    fig = visualizers["bio"].visualize_genomic_data(
                        genomic_data=viz_data,
                        title="Genomic Data Visualization",
                        genomic_region=genomic_region if genomic_region else None,
                        interactive=interactive_genomic
                    )
                    
                    if interactive_genomic:
                        st_plotly(fig)
                    else:
                        st.pyplot(fig)
                    
                    # Save visualization
                    save_viz = st.button("Save Visualization", key="save_genomic_viz")
                    if save_viz:
                        if interactive_genomic:
                            save_path = os.path.join(tempfile.gettempdir(), "genomic_data.html")
                            fig.write_html(save_path)
                        else:
                            save_path = os.path.join(tempfile.gettempdir(), "genomic_data.png")
                            fig.savefig(save_path, dpi=300, bbox_inches="tight")
                            
                        st.success(f"Visualization saved to {save_path}")
                else:
                    st.warning("No appropriate genomic data for the selected visualization type.")
            
            except Exception as e:
                st.error(f"Error creating genomic visualization: {str(e)}")
                st.exception(e)
        
        # Property prediction if protein structure and sequence available
        if "protein_struct" in uploaded_data and "protein_seq" in uploaded_data and "bio" in models:
            st.header("Property Prediction")
            
            with st.expander("Prediction Options", expanded=True):
                property_type = st.selectbox(
                    "Property to Predict",
                    ["Binding Affinity", "Solubility", "Thermal Stability", "Function"],
                    key="bio_property_type"
                )
                
                run_prediction = st.button("Run Prediction", key="bio_prediction")
            
            if run_prediction:
                try:
                    # Prepare input data
                    protein_seq = uploaded_data["protein_seq"]["sequence"]
                    protein_struct = uploaded_data["protein_struct"]
                    
                    # Convert sequence to indices
                    aa_to_idx = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
                    aa_to_idx['<pad>'] = 0
                    aa_to_idx['<unk>'] = 21
                    
                    seq_indices = []
                    for aa in protein_seq:
                        seq_indices.append(aa_to_idx.get(aa, aa_to_idx['<unk>']))
                    
                    # Add padding to fixed length
                    max_length = 1000
                    if len(seq_indices) > max_length:
                        seq_indices = seq_indices[:max_length]
                    else:
                        seq_indices.extend([0] * (max_length - len(seq_indices)))
                    
                    # Create attention mask
                    attention_mask = [1 if idx > 0 else 0 for idx in seq_indices]
                    
                    # Prepare structure data
                    positions = protein_struct["positions"]
                    
                    # If positions is larger than sequence, truncate
                    if positions.shape[0] > len(protein_seq):
                        positions = positions[:len(protein_seq)]
                    
                    # Convert to residue indices
                    residue_indices = []
                    for aa in protein_seq[:positions.shape[0]]:
                        residue_indices.append(aa_to_idx.get(aa, aa_to_idx['<unk>']))
                        
                    # Add padding to fixed length
                    max_residues = 500
                    res_padding = [0] * (max_residues - len(residue_indices)) if len(residue_indices) < max_residues else []
                    residue_indices = residue_indices[:max_residues] + res_padding
                    
                    # Pad positions
                    pos_padding = np.zeros((max_residues - positions.shape[0], 3)) if positions.shape[0] < max_residues else np.zeros((0, 3))
                    positions = np.vstack([positions[:max_residues], pos_padding])
                    
                    # Convert to torch tensors and add batch dimension
                    sequences = torch.tensor(seq_indices).unsqueeze(0).to(device)
                    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
                    residues = torch.tensor(residue_indices).unsqueeze(0).to(device)
                    coordinates = torch.tensor(positions, dtype=torch.float32).unsqueeze(0).to(device)
                    residue_mask = torch.tensor([1 if idx > 0 else 0 for idx in residue_indices]).unsqueeze(0).to(device)
                    
                    # Run model inference
                    model = models["bio"]
                    model.eval()
                    
                    with torch.no_grad():
                        # Prepare inputs for the model
                        protein_seq_input = {
                            "sequences": sequences,
                            "attention_mask": attention_mask
                        }
                        
                        protein_struct_input = {
                            "residues": residues,
                            "coordinates": coordinates,
                            "mask": residue_mask
                        }
                        
                        # Run inference
                        outputs = model(
                            protein_seq=protein_seq_input,
                            protein_struct=protein_struct_input
                        )
                    
                    # Extract predictions
                    if "prediction_mean" in outputs and "prediction_var" in outputs:
                        # Model with uncertainty
                        mean = outputs["prediction_mean"].cpu().numpy().flatten()[0]
                        var = outputs["prediction_var"].cpu().numpy().flatten()[0]
                        std = np.sqrt(var)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                label=f"Predicted {property_type}",
                                value=f"{mean:.3f}"
                            )
                        with col2:
                            st.metric(
                                label="Uncertainty (Std)",
                                value=f"{std:.3f}"
                            )
                        
                        # Create visualization of the prediction with uncertainty
                        fig, ax = plt.subplots(figsize=(8, 4))
                        
                        # Plot confidence interval
                        x = np.array([mean - 2*std, mean - std, mean, mean + std, mean + 2*std])
                        y = np.array([0.05, 0.32, 1.0, 0.32, 0.05])
                        ax.fill_between(x, y, alpha=0.3)
                        ax.axvline(x=mean, color='red', linestyle='-', linewidth=2)
                        
                        ax.set_xlabel(property_type)
                        ax.set_ylabel("Probability Density")
                        ax.set_title(f"Predicted {property_type} with Uncertainty")
                        
                        st.pyplot(fig)
                        
                    else:
                        # Model without uncertainty
                        prediction = outputs["predictions"].cpu().numpy().flatten()[0]
                        
                        # Display result
                        st.metric(
                            label=f"Predicted {property_type}",
                            value=f"{prediction:.3f}"
                        )
                
                except Exception as e:
                    st.error(f"Error during property prediction: {str(e)}")
                    st.exception(e)
    
    else:
        st.info("Please upload bioinformatics data or use the demo data to get started.")


def discovery_app():
    """Streamlit app for pattern discovery."""
    st.title("Scientific Pattern Discovery")
    
    st.write("""
    This application allows you to upload numerical data and discover patterns, 
    clusters, and correlations using various machine learning techniques.
    """)
    
    # Sidebar for upload and options
    st.sidebar.header("Data Upload")
    
    data_file = st.sidebar.file_uploader(
        "Upload numerical data (CSV, TSV, TXT)",
        type=["csv", "tsv", "txt", "data"]
    )
    
    uploaded_data = {}
    
    if data_file:
        try:
            uploaded_data["numerical"] = _parse_numerical_data(data_file)
            st.sidebar.success(f"Data loaded: {data_file.name}")
        except Exception as e:
            st.sidebar.error(f"Error parsing data: {str(e)}")
    
    # Generate demo data if nothing is uploaded
    if not uploaded_data and st.sidebar.button("Use Demo Data"):
        # Generate synthetic clustered data
        n_samples = 1000
        n_features = 20
        n_clusters = 5
        
        # Create cluster centers
        centers = np.random.rand(n_clusters, n_features) * 2 - 1
        
        # Generate samples around centers
        data = np.zeros((n_samples, n_features))
        labels = np.zeros(n_samples)
        
        for i in range(n_samples):
            cluster = i % n_clusters
            data[i] = centers[cluster] + np.random.randn(n_features) * 0.1
            labels[i] = cluster
        
        # Create feature names
        feature_names = [f"Feature_{i+1}" for i in range(n_features)]
        
        uploaded_data["numerical"] = {
            "data": data,
            "feature_names": feature_names,
            "labels": labels,
            "format": "demo"
        }
        
        st.sidebar.success("Demo data loaded!")
    
    # Discovery tools
    if uploaded_data and "numerical" in uploaded_data:
        st.header("Pattern Discovery Tools")
        
        data = uploaded_data["numerical"]["data"]
        feature_names = uploaded_data["numerical"].get("feature_names", [f"Feature_{i+1}" for i in range(data.shape[1])])
        
        # Dimension reduction
        st.subheader("Dimension Reduction")
        
        with st.expander("Dimension Reduction Options", expanded=True):
            dimension_methods = {
                "PCA": "Principal Component Analysis",
                "t-SNE": "t-Distributed Stochastic Neighbor Embedding",
                "UMAP": "Uniform Manifold Approximation and Projection"
            }
            
            reduction_method = st.selectbox(
                "Reduction Method",
                list(dimension_methods.keys()),
                format_func=lambda x: f"{x}: {dimension_methods[x]}"
            )
            
            n_components = st.slider(
                "Number of Components",
                min_value=2,
                max_value=min(10, data.shape[1]),
                value=2
            )
            
            run_reduction = st.button("Run Dimension Reduction")
        
        if run_reduction:
            try:
                method_map = {"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap"}
                method = method_map[reduction_method]
                
                with st.spinner(f"Running {reduction_method}..."):
                    reduced_data = discovery.dimension_reduction(
                        data, 
                        method=method, 
                        n_components=n_components
                    )
                
                # Store for later use
                uploaded_data["numerical"]["reduced_data"] = reduced_data
                uploaded_data["numerical"]["reduction_method"] = reduction_method
                
                # Visualization
                if n_components == 2:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7)
                    ax.set_title(f"{reduction_method} Projection")
                    ax.set_xlabel("Component 1")
                    ax.set_ylabel("Component 2")
                    
                    st.pyplot(fig)
                    
                elif n_components == 3:
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    scatter = ax.scatter(
                        reduced_data[:, 0],
                        reduced_data[:, 1],
                        reduced_data[:, 2],
                        alpha=0.7
                    )
                    ax.set_title(f"{reduction_method} Projection")
                    ax.set_xlabel("Component 1")
                    ax.set_ylabel("Component 2")
                    ax.set_zlabel("Component 3")
                    
                    st.pyplot(fig)
                    
                else:
                    st.write(f"{reduction_method} results ({n_components} components):")
                    df = pd.DataFrame(
                        reduced_data,
                        columns=[f"Component {i+1}" for i in range(n_components)]
                    )
                    st.dataframe(df)
            
            except Exception as e:
                st.error(f"Error during dimension reduction: {str(e)}")
                st.exception(e)
        
        # Clustering
        st.subheader("Clustering")
        
        with st.expander("Clustering Options", expanded=True):
            clustering_methods = {
                "K-means": "K-means Clustering",
                "DBSCAN": "Density-Based Spatial Clustering",
                "Hierarchical": "Hierarchical Clustering",
                "Gaussian Mixture": "Gaussian Mixture Models"
            }
            
            cluster_method = st.selectbox(
                "Clustering Method",
                list(clustering_methods.keys()),
                format_func=lambda x: f"{x}: {clustering_methods[x]}"
            )
            
            if cluster_method in ["K-means", "Hierarchical", "Gaussian Mixture"]:
                n_clusters = st.slider(
                    "Number of Clusters",
                    min_value=2,
                    max_value=10,
                    value=5
                )
            else:
                eps = st.slider(
                    "DBSCAN Epsilon",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1
                )
                min_samples = st.slider(
                    "DBSCAN Min Samples",
                    min_value=2,
                    max_value=20,
                    value=5
                )
            
            use_reduced = st.checkbox(
                "Use Reduced Data",
                value="reduced_data" in uploaded_data["numerical"],
                disabled="reduced_data" not in uploaded_data["numerical"]
            )
            
            run_clustering = st.button("Run Clustering")
        
        if run_clustering:
            try:
                method_map = {
                    "K-means": "kmeans",
                    "DBSCAN": "dbscan",
                    "Hierarchical": "agglomerative",
                    "Gaussian Mixture": "gaussian_mixture"
                }
                method = method_map[cluster_method]
                
                # Prepare data
                input_data = uploaded_data["numerical"]["reduced_data"] if use_reduced else data
                
                with st.spinner(f"Running {cluster_method}..."):
                    # Prepare clustering parameters
                    kwargs = {}
                    if method == "kmeans" or method == "agglomerative" or method == "gaussian_mixture":
                        kwargs["n_clusters"] = n_clusters
                    elif method == "dbscan":
                        kwargs["eps"] = eps
                        kwargs["min_samples"] = min_samples
                    
                    # Run clustering
                    labels, score = discovery.cluster_data(
                        input_data,
                        method=method,
                        **kwargs
                    )
                
                # Store for later use
                uploaded_data["numerical"]["cluster_labels"] = labels
                uploaded_data["numerical"]["cluster_method"] = cluster_method
                
                # Display clustering score if available
                if score is not None:
                    st.metric(
                        label="Clustering Score (Silhouette)",
                        value=f"{score:.3f}"
                    )
                
                # Visualization with reduced data
                if "reduced_data" in uploaded_data["numerical"] and uploaded_data["numerical"]["reduced_data"].shape[1] >= 2:
                    reduced = uploaded_data["numerical"]["reduced_data"]
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(
                        reduced[:, 0],
                        reduced[:, 1],
                        c=labels,
                        cmap='tab10',
                        alpha=0.7
                    )
                    
                    # Add legend for clusters
                    unique_labels = np.unique(labels)
                    if -1 in unique_labels:  # DBSCAN can have -1 for noise
                        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor='k', markersize=10,
                                                    label='Noise')]
                        for label in unique_labels:
                            if label != -1:
                                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                                markerfacecolor=scatter.cmap(scatter.norm(label)), 
                                                                markersize=10, label=f'Cluster {label+1}'))
                    else:
                        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=scatter.cmap(scatter.norm(label)), 
                                                    markersize=10, label=f'Cluster {label+1}')
                                        for label in unique_labels]
                    
                    ax.legend(handles=legend_elements)
                    ax.set_title(f"{cluster_method} Clustering")
                    ax.set_xlabel(f"{uploaded_data['numerical']['reduction_method']} Component 1")
                    ax.set_ylabel(f"{uploaded_data['numerical']['reduction_method']} Component 2")
                    
                    st.pyplot(fig)
                    
                    # Create cluster statistics
                    st.subheader("Cluster Statistics")
                    
                    cluster_stats = []
                    for label in np.unique(labels):
                        if label != -1:  # Skip noise
                            mask = labels == label
                            cluster_data = data[mask]
                            size = mask.sum()
                            means = np.mean(cluster_data, axis=0)
                            stds = np.std(cluster_data, axis=0)
                            
                            top_features = np.argsort(means)[-5:]  # Top 5 features by mean
                            
                            stats = {
                                "Cluster": f"Cluster {label+1}",
                                "Size": size,
                                "Proportion": f"{size/len(labels)*100:.1f}%",
                                "Top Features": ", ".join([feature_names[i] for i in top_features])
                            }
                            
                            cluster_stats.append(stats)
                    
                    if cluster_stats:
                        st.table(pd.DataFrame(cluster_stats))
                
                # Show distribution of clusters
                unique_labels = np.unique(labels)
                counts = [np.sum(labels == label) for label in unique_labels]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(
                    [f"Cluster {label+1}" if label != -1 else "Noise" for label in unique_labels],
                    counts
                )
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height}',
                           ha='center', va='bottom')
                
                ax.set_title("Cluster Distribution")
                ax.set_xlabel("Cluster")
                ax.set_ylabel("Number of Samples")
                
                st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error during clustering: {str(e)}")
                st.exception(e)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        
        with st.expander("Correlation Options", expanded=True):
            corr_methods = {
                "Pearson": "Pearson Correlation (linear)",
                "Spearman": "Spearman Correlation (monotonic)",
                "Kendall": "Kendall Tau Correlation (ordinal)"
            }
            
            corr_method = st.selectbox(
                "Correlation Method",
                list(corr_methods.keys()),
                format_func=lambda x: f"{x}: {corr_methods[x]}"
            )
            
            threshold = st.slider(
                "Correlation Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05
            )
            
            run_correlation = st.button("Run Correlation Analysis")
        
        if run_correlation:
            try:
                method_map = {
                    "Pearson": "pearson",
                    "Spearman": "spearman",
                    "Kendall": "kendall"
                }
                method = method_map[corr_method]
                
                with st.spinner(f"Running {corr_method} correlation analysis..."):
                    # Create dictionary of features
                    feature_dict = {}
                    for i, name in enumerate(feature_names):
                        feature_dict[name] = data[:, i]
                    
                    # Run correlation analysis
                    corr_matrix, significant_pairs = discovery.find_correlations(
                        feature_dict,
                        method=method,
                        threshold=threshold
                    )
                
                # Store for later use
                uploaded_data["numerical"]["corr_matrix"] = corr_matrix
                uploaded_data["numerical"]["corr_method"] = corr_method
                
                # Visualization of correlation matrix
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Show correlation matrix as heatmap
                im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                
                # Add ticks and labels
                if len(feature_names) <= 20:  # Show all labels if not too many
                    ax.set_xticks(np.arange(len(feature_names)))
                    ax.set_yticks(np.arange(len(feature_names)))
                    ax.set_xticklabels(feature_names, rotation=45, ha="right")
                    ax.set_yticklabels(feature_names)
                else:  # Show every n labels if too many
                    n = max(1, len(feature_names) // 20)
                    ax.set_xticks(np.arange(0, len(feature_names), n))
                    ax.set_yticks(np.arange(0, len(feature_names), n))
                    ax.set_xticklabels([feature_names[i] for i in range(0, len(feature_names), n)], rotation=45, ha="right")
                    ax.set_yticklabels([feature_names[i] for i in range(0, len(feature_names), n)])
                    
                # Add colorbar
                cbar = fig.colorbar(im)
                cbar.set_label(f"{corr_method} Correlation")
                
                ax.set_title(f"{corr_method} Correlation Matrix")
                
                # Resize plot area for readability
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Show significant pairs
                if len(significant_pairs) > 0:
                    st.subheader(f"Significant Correlations (|r| > {threshold})")
                    
                    sig_corrs = []
                    for i, j in significant_pairs:
                        feat1 = feature_names[i]
                        feat2 = feature_names[j]
                        corr = corr_matrix[i, j]
                        
                        sig_corrs.append({
                            "Feature 1": feat1,
                            "Feature 2": feat2,
                            "Correlation": f"{corr:.3f}"
                        })
                    
                    # Sort by absolute correlation
                    sig_corrs = sorted(sig_corrs, key=lambda x: abs(float(x["Correlation"])), reverse=True)
                    
                    st.table(pd.DataFrame(sig_corrs))
                else:
                    st.info(f"No significant correlations found with threshold {threshold}")
            
            except Exception as e:
                st.error(f"Error during correlation analysis: {str(e)}")
                st.exception(e)
        
        # Anomaly detection
        st.subheader("Anomaly Detection")
        
        with st.expander("Anomaly Detection Options", expanded=True):
            anomaly_methods = {
                "Isolation Forest": "Efficiently detects anomalies in high-dimensional spaces",
                "Local Outlier Factor": "Identifies anomalies based on local density",
                "One-Class SVM": "Finds the boundary of normal data"
            }
            
            anomaly_method = st.selectbox(
                "Anomaly Detection Method",
                list(anomaly_methods.keys()),
                format_func=lambda x: f"{x}: {anomaly_methods[x]}"
            )
            
            contamination = st.slider(
                "Expected Anomaly Proportion",
                min_value=0.01,
                max_value=0.2,
                value=0.05,
                step=0.01
            )
            
            use_reduced_anomaly = st.checkbox(
                "Use Reduced Data for Anomaly Detection",
                value="reduced_data" in uploaded_data["numerical"],
                disabled="reduced_data" not in uploaded_data["numerical"],
                key="use_reduced_anomaly"
            )
            
            run_anomaly = st.button("Run Anomaly Detection")
        
        if run_anomaly:
            try:
                method_map = {
                    "Isolation Forest": "isolation_forest",
                    "Local Outlier Factor": "local_outlier_factor",
                    "One-Class SVM": "one_class_svm"
                }
                method = method_map[anomaly_method]
                
                # Prepare data
                input_data = uploaded_data["numerical"]["reduced_data"] if use_reduced_anomaly else data
                
                with st.spinner(f"Running {anomaly_method}..."):
                    # Run anomaly detection
                    anomalies = discovery.find_anomalies(
                        input_data,
                        method=method,
                        contamination=contamination
                    )
                
                # Store for later use
                uploaded_data["numerical"]["anomalies"] = anomalies
                uploaded_data["numerical"]["anomaly_method"] = anomaly_method
                
                # Display number of anomalies
                n_anomalies = np.sum(anomalies)
                st.metric(
                    label="Detected Anomalies",
                    value=f"{n_anomalies} ({n_anomalies/len(anomalies)*100:.1f}%)"
                )
                
                # Visualization with reduced data
                if "reduced_data" in uploaded_data["numerical"] and uploaded_data["numerical"]["reduced_data"].shape[1] >= 2:
                    reduced = uploaded_data["numerical"]["reduced_data"]
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Plot normal samples
                    ax.scatter(
                        reduced[~anomalies, 0],
                        reduced[~anomalies, 1],
                        c='blue',
                        label='Normal',
                        alpha=0.5
                    )
                    
                    # Plot anomalies
                    ax.scatter(
                        reduced[anomalies, 0],
                        reduced[anomalies, 1],
                        c='red',
                        label='Anomaly',
                        alpha=0.8
                    )
                    
                    ax.set_title(f"Anomaly Detection using {anomaly_method}")
                    ax.set_xlabel(f"{uploaded_data['numerical']['reduction_method']} Component 1")
                    ax.set_ylabel(f"{uploaded_data['numerical']['reduction_method']} Component 2")
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                # Show feature distributions for anomalies vs normal
                st.subheader("Feature Distributions")
                
                # Select top features with the most different distributions
                feature_diffs = []
                for i, feature in enumerate(feature_names):
                    normal_mean = np.mean(data[~anomalies, i])
                    anomaly_mean = np.mean(data[anomalies, i])
                    diff = abs(normal_mean - anomaly_mean)
                    feature_diffs.append((feature, diff))
                
                # Sort by difference
                feature_diffs = sorted(feature_diffs, key=lambda x: x[1], reverse=True)
                
                # Show top 3 features
                top_features = [f[0] for f in feature_diffs[:3]]
                top_indices = [feature_names.index(f) for f in top_features]
                
                fig, axes = plt.subplots(1, len(top_features), figsize=(15, 5))
                if len(top_features) == 1:
                    axes = [axes]
                
                for i, (feature, idx) in enumerate(zip(top_features, top_indices)):
                    # Plot histograms for normal vs anomaly
                    axes[i].hist(data[~anomalies, idx], bins=20, alpha=0.5, label='Normal', color='blue')
                    axes[i].hist(data[anomalies, idx], bins=20, alpha=0.5, label='Anomaly', color='red')
                    
                    axes[i].set_title(feature)
                    if i == 0:
                        axes[i].legend()
                
                fig.suptitle("Feature Distributions: Normal vs Anomalies")
                plt.tight_layout()
                
                st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error during anomaly detection: {str(e)}")
                st.exception(e)
    
    else:
        st.info("Please upload numerical data or use the demo data to get started.")


def main():
    """Main function for the web application."""
    st.set_page_config(
        page_title="Scientific Visualization AI",
        page_icon="🧬",
        layout="wide"
    )
    
    st.sidebar.title("Scientific Visualization AI")
    st.sidebar.image("https://placehold.co/600x200/white/blue?text=Scientific+Viz+AI", width=300)
    
    # App selection
    app_mode = st.sidebar.selectbox(
        "Choose Application",
        ["Home", "Materials Science", "Bioinformatics", "Pattern Discovery"]
    )
    
    # Initialize models if not already loaded
    if not models:
        model_dir = os.environ.get("MODEL_DIR", "./models")
        use_gpu = not os.environ.get("NO_GPU", False)
        load_models(model_dir, use_gpu)
    
    # Run the selected app
    if app_mode == "Home":
        home_app()
    elif app_mode == "Materials Science":
        materials_science_app()
    elif app_mode == "Bioinformatics":
        bioinformatics_app()
    elif app_mode == "Pattern Discovery":
        discovery_app()
    

def home_app():
    """Home page of the web application."""
    st.title("Multi-Modal Generative AI for Scientific Visualization and Discovery")
    
    st.write("""
    Welcome to the Scientific Visualization AI platform! This application uses advanced machine learning 
    techniques to analyze scientific data, discover patterns, and generate insightful visualizations.
    """)
    
    # Features
    st.header("Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Materials Science")
        st.write("""
        - Visualize crystal structures in 3D
        - Analyze spectroscopy data (XRD, XPS, Raman, etc.)
        - Predict material properties with uncertainty quantification
        - Discover relationships between structure and properties
        """)
        
        st.subheader("🧬 Bioinformatics")
        st.write("""
        - Visualize protein sequences and structures
        - Analyze genomic data and gene expression
        - Predict protein properties and interactions
        - Identify functional motifs and domains
        """)
    
    with col2:
        st.subheader("📊 Pattern Discovery")
        st.write("""
        - Reduce dimensionality of complex datasets
        - Cluster data to find natural groupings
        - Detect correlations between features
        - Identify anomalies and outliers
        """)
        
        st.subheader("🧠 Multi-Modal Learning")
        st.write("""
        - Combine information from multiple data types
        - Generate novel visualizations with diffusion models
        - Provide uncertainty-aware predictions
        - Create interactive visualizations for exploration
        """)
    
    # How to use
    st.header("How to Use")
    
    st.write("""
    1. Choose an application from the sidebar on the left
    2. Upload your scientific data or use the provided demo data
    3. Configure visualization and analysis options
    4. Explore the results and download visualizations
    """)
    
    # Example visualizations
    st.header("Example Visualizations")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            # Example crystal structure
            st.subheader("Crystal Structure")
            st.image("https://placehold.co/600x400/lightblue/white?text=Crystal+Structure", 
                    caption="Example crystal structure visualization")
            
            # Example protein sequence
            st.subheader("Protein Sequence")
            st.image("https://placehold.co/600x400/lightgreen/white?text=Protein+Sequence", 
                    caption="Example protein sequence visualization")
        
        with col2:
            # Example spectroscopy
            st.subheader("Spectroscopy Data")
            st.image("https://placehold.co/600x400/lightpink/white?text=Spectroscopy+Data", 
                    caption="Example spectroscopy visualization")
            
            # Example pattern discovery
            st.subheader("Pattern Discovery")
            st.image("https://placehold.co/600x400/lightyellow/white?text=Pattern+Discovery", 
                    caption="Example pattern discovery visualization")
    
    # About
    st.header("About")
    
    st.write("""
    This application is part of the Multi-Modal Generative AI for Scientific Visualization and Discovery project, 
    which aims to develop advanced machine learning techniques for scientific data analysis and visualization.
    
    The project combines transformer-based architectures, diffusion models, and pattern discovery algorithms 
    to provide a comprehensive platform for scientific visualization and discovery.
    """)
    
    # Disclaimer
    st.sidebar.markdown("---")
    st.sidebar.caption("Disclaimer: This application is for demonstration purposes only. The predictions and visualizations are based on synthetic data and models for this demo version.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scientific Visualization AI Web Application"
    )
    
    parser.add_argument("--model-dir", type=str, default="./models",
                      help="Directory containing model checkpoints")
    parser.add_argument("--port", type=int, default=8501,
                      help="Port to run the web application on")
    parser.add_argument("--no-gpu", action="store_true",
                      help="Disable GPU usage")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["MODEL_DIR"] = args.model_dir
    os.environ["NO_GPU"] = str(args.no_gpu)
    
    # Run the application using Streamlit CLI
    import streamlit.web.cli as stcli
    import sys
    
    sys.argv = ["streamlit", "run", __file__, "--server.port", str(args.port)]
    sys.exit(stcli.main())