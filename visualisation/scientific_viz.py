"""
Scientific visualization generator for multi-modal scientific data.

This module provides specialized visualizations for scientific domains, including
materials science and bioinformatics.
"""
import os
import logging
import json
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd

from .viz_generator import ScientificVisualizer
from ..models.discovery import PatternDiscovery

logger = logging.getLogger(__name__)


class ScientificMaterialsVisualizer:
    """
    Specialized visualizations for materials science data.
    """
    def __init__(
        self,
        output_dir: str = "./visualizations/materials",
        discovery: Optional[PatternDiscovery] = None
    ):
        """
        Initialize the materials science visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            discovery: Pattern discovery module for finding relationships
        """
        self.output_dir = output_dir
        self.discovery = discovery or PatternDiscovery()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def visualize_crystal_structure(
        self,
        atom_types: Union[np.ndarray, torch.Tensor],
        positions: Union[np.ndarray, torch.Tensor],
        lattice: Optional[Union[np.ndarray, torch.Tensor]] = None,
        title: str = "Crystal Structure",
        save_path: Optional[str] = None,
        interactive: bool = False,
        show_unit_cell: bool = True,
        atom_size_scale: float = 50.0,
        element_colors: Optional[Dict[int, str]] = None
    ) -> Union[Figure, go.Figure]:
        """
        Visualize a crystal structure in 3D.
        
        Args:
            atom_types: Atomic numbers
            positions: Cartesian coordinates of atoms (Å)
            lattice: Unit cell parameters [a, b, c, alpha, beta, gamma]
            title: Plot title
            save_path: Path to save the visualization
            interactive: Whether to return an interactive plotly visualization
            show_unit_cell: Whether to show the unit cell
            atom_size_scale: Scale factor for atom sizes
            element_colors: Custom colors for elements
            
        Returns:
            Figure object
        """
        # Convert to numpy arrays
        if isinstance(atom_types, torch.Tensor):
            atom_types = atom_types.detach().cpu().numpy()
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
        if lattice is not None and isinstance(lattice, torch.Tensor):
            lattice = lattice.detach().cpu().numpy()
            
        # Filter out padding
        mask = atom_types > 0
        atom_types = atom_types[mask]
        positions = positions[mask]
        
        # Element colors and sizes
        if element_colors is None:
            # Default colors for common elements
            element_colors = {
                1: "#FFFFFF",   # H - white
                6: "#999999",   # C - gray
                7: "#3050F8",   # N - blue
                8: "#FF0D0D",   # O - red
                9: "#90E050",   # F - light green
                15: "#FF8000",  # P - orange
                16: "#FFFF30",  # S - yellow
                17: "#1FF01F",  # Cl - green
                26: "#E06633",  # Fe - rust
                29: "#C88033",  # Cu - copper
                30: "#7D80B0",  # Zn - light purple
                47: "#C0C0C0",  # Ag - silver
                79: "#FFD123"   # Au - gold
            }
        
        # Element radii (covalent radii in Å)
        element_radii = {
            1: 0.31,   # H
            6: 0.76,   # C
            7: 0.71,   # N
            8: 0.66,   # O
            9: 0.57,   # F
            15: 1.07,  # P
            16: 1.05,  # S
            17: 1.02,  # Cl
            26: 1.26,  # Fe
            29: 1.38,  # Cu
            30: 1.31,  # Zn
            47: 1.44,  # Ag
            79: 1.36   # Au
        }
        
        # Default radius for elements not in the dictionary
        default_radius = 1.0
        
        # Get colors and sizes for each atom
        colors_list = [element_colors.get(at, "#CCCCCC") for at in atom_types]
        sizes = [element_radii.get(at, default_radius) * atom_size_scale for at in atom_types]
        
        # Unit cell vectors if lattice is provided
        if lattice is not None and show_unit_cell:
            a, b, c, alpha, beta, gamma = lattice
            
            # Convert angles from degrees to radians
            alpha_rad = np.radians(alpha)
            beta_rad = np.radians(beta)
            gamma_rad = np.radians(gamma)
            
            # Compute unit cell vectors
            # This is a simplified calculation for orthogonal cells
            # For general triclinic cells, use a proper crystal toolbox
            v1 = np.array([a, 0, 0])
            v2 = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0])
            v3 = np.array([c * np.cos(beta_rad), 
                          c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad),
                          c * np.sqrt(1 - np.cos(beta_rad)**2 - 
                                    ((np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad))**2)])
            
            # Unit cell corners
            corners = np.array([
                [0, 0, 0],
                v1,
                v2,
                v3,
                v1 + v2,
                v1 + v3,
                v2 + v3,
                v1 + v2 + v3
            ])
        
        # Create the visualization
        if interactive:
            # Plotly interactive 3D visualization
            fig = go.Figure()
            
            # Add atoms
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors_list,
                    symbol='circle',
                    line=dict(color='black', width=1)
                ),
                text=[f"Element: {at}" for at in atom_types],
                name='Atoms'
            ))
            
            # Add unit cell if requested
            if lattice is not None and show_unit_cell:
                # Lines for the unit cell
                lines = [
                    (0, 1), (0, 2), (0, 3),
                    (1, 4), (1, 5),
                    (2, 4), (2, 6),
                    (3, 5), (3, 6),
                    (4, 7), (5, 7), (6, 7)
                ]
                
                for start, end in lines:
                    fig.add_trace(go.Scatter3d(
                        x=[corners[start, 0], corners[end, 0]],
                        y=[corners[start, 1], corners[end, 1]],
                        z=[corners[start, 2], corners[end, 2]],
                        mode='lines',
                        line=dict(color='black', width=3),
                        showlegend=False
                    ))
            
            # Set layout
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='X (Å)',
                    yaxis_title='Y (Å)',
                    zaxis_title='Z (Å)',
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            # Save if requested
            if save_path:
                if not save_path.endswith('.html'):
                    save_path += '.html'
                fig.write_html(save_path)
                logger.info(f"Saved interactive crystal structure visualization to {save_path}")
                
            return fig
            
        else:
            # Matplotlib static visualization
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot atoms
            for i, (pos, at, size) in enumerate(zip(positions, atom_types, sizes)):
                ax.scatter(
                    pos[0], pos[1], pos[2],
                    color=element_colors.get(at, "#CCCCCC"),
                    s=size,
                    edgecolors='black',
                    alpha=0.8
                )
            
            # Plot unit cell if requested
            if lattice is not None and show_unit_cell:
                # Lines for the unit cell
                lines = [
                    (0, 1), (0, 2), (0, 3),
                    (1, 4), (1, 5),
                    (2, 4), (2, 6),
                    (3, 5), (3, 6),
                    (4, 7), (5, 7), (6, 7)
                ]
                
                for start, end in lines:
                    ax.plot(
                        [corners[start, 0], corners[end, 0]],
                        [corners[start, 1], corners[end, 1]],
                        [corners[start, 2], corners[end, 2]],
                        color='black',
                        linewidth=1
                    )
            
            # Set labels and title
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.set_title(title)
            
            # Equal aspect ratio
            # Get max range for all axes
            max_range = np.array([
                positions[:, 0].max() - positions[:, 0].min(),
                positions[:, 1].max() - positions[:, 1].min(),
                positions[:, 2].max() - positions[:, 2].min()
            ]).max() / 2.0
            
            mid_x = (positions[:, 0].max() + positions[:, 0].min()) / 2
            mid_y = (positions[:, 1].max() + positions[:, 1].min()) / 2
            mid_z = (positions[:, 2].max() + positions[:, 2].min()) / 2
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Add a legend for element types
            unique_elements = np.unique(atom_types)
            handles = []
            labels = []
            
            element_names = {
                1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl",
                26: "Fe", 29: "Cu", 30: "Zn", 47: "Ag", 79: "Au"
            }
            
            for elem in unique_elements:
                color = element_colors.get(elem, "#CCCCCC")
                handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10))
                labels.append(element_names.get(elem, f"Element {elem}"))
                
            ax.legend(handles, labels, loc='upper right')
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved crystal structure visualization to {save_path}")
            
            return fig
    
    def visualize_spectroscopy(
        self,
        spectrum: Union[np.ndarray, torch.Tensor],
        x_values: Optional[Union[np.ndarray, torch.Tensor]] = None,
        spectrum_type: str = "XRD",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        interactive: bool = False,
        show_peaks: bool = True,
        peak_threshold: float = 0.5
    ) -> Union[Figure, go.Figure]:
        """
        Visualize spectroscopy data.
        
        Args:
            spectrum: Intensity values of the spectrum
            x_values: X-axis values (e.g., 2theta for XRD, energy for XPS)
            spectrum_type: Type of spectrum ("XRD", "XPS", "Raman", etc.)
            title: Plot title
            save_path: Path to save the visualization
            interactive: Whether to return an interactive plotly visualization
            show_peaks: Whether to highlight peaks in the spectrum
            peak_threshold: Threshold for peak detection (relative to max)
            
        Returns:
            Figure object
        """
        # Convert to numpy arrays
        if isinstance(spectrum, torch.Tensor):
            spectrum = spectrum.detach().cpu().numpy()
        if x_values is not None and isinstance(x_values, torch.Tensor):
            x_values = x_values.detach().cpu().numpy()
            
        # Generate x values if not provided
        if x_values is None:
            x_values = np.arange(len(spectrum))
        
        # Default title if not provided
        if title is None:
            title = f"{spectrum_type} Spectrum"
            
        # Peak detection
        peaks = []
        if show_peaks:
            from scipy.signal import find_peaks
            
            # Normalize spectrum for peak detection
            normalized = spectrum / np.max(spectrum)
            peak_indices, _ = find_peaks(normalized, height=peak_threshold)
            peaks = [(x_values[i], spectrum[i]) for i in peak_indices]
        
        # Create visualization
        if interactive:
            # Plotly interactive visualization
            fig = go.Figure()
            
            # Add spectrum line
            fig.add_trace(go.Scatter(
                x=x_values,
                y=spectrum,
                mode='lines',
                name=spectrum_type,
                line=dict(width=2, color='blue')
            ))
            
            # Add peaks if requested
            if show_peaks and peaks:
                peak_x = [p[0] for p in peaks]
                peak_y = [p[1] for p in peaks]
                
                fig.add_trace(go.Scatter(
                    x=peak_x,
                    y=peak_y,
                    mode='markers',
                    name='Peaks',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='circle'
                    )
                ))
            
            # X-axis label based on spectrum type
            x_label = "2θ (degrees)" if spectrum_type == "XRD" else \
                     "Binding Energy (eV)" if spectrum_type == "XPS" else \
                     "Raman Shift (cm⁻¹)" if spectrum_type == "Raman" else \
                     "Wavelength (nm)" if spectrum_type == "UV-Vis" else \
                     "Wavenumber (cm⁻¹)" if spectrum_type == "IR" else \
                     "X Value"
            
            # Set layout
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title='Intensity (a.u.)',
                legend_title=spectrum_type,
                hovermode='closest'
            )
            
            # Save if requested
            if save_path:
                if not save_path.endswith('.html'):
                    save_path += '.html'
                fig.write_html(save_path)
                logger.info(f"Saved interactive spectrum visualization to {save_path}")
                
            return fig
            
        else:
            # Matplotlib static visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot spectrum
            ax.plot(x_values, spectrum, 'b-', linewidth=1.5, label=spectrum_type)
            
            # Add peaks if requested
            if show_peaks and peaks:
                peak_x = [p[0] for p in peaks]
                peak_y = [p[1] for p in peaks]
                ax.plot(peak_x, peak_y, 'ro', markersize=5, label='Peaks')
                
                # Add peak annotations
                for i, (x, y) in enumerate(zip(peak_x, peak_y)):
                    ax.annotate(
                        f"{x:.1f}",
                        xy=(x, y),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5)
                    )
            
            # X-axis label based on spectrum type
            x_label = "2θ (degrees)" if spectrum_type == "XRD" else \
                     "Binding Energy (eV)" if spectrum_type == "XPS" else \
                     "Raman Shift (cm⁻¹)" if spectrum_type == "Raman" else \
                     "Wavelength (nm)" if spectrum_type == "UV-Vis" else \
                     "Wavenumber (cm⁻¹)" if spectrum_type == "IR" else \
                     "X Value"
            
            # Set labels and title
            ax.set_xlabel(x_label)
            ax.set_ylabel('Intensity (a.u.)')
            ax.set_title(title)
            ax.legend()
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved spectrum visualization to {save_path}")
            
            return fig
    
    def visualize_property_map(
        self,
        property_values: Union[np.ndarray, torch.Tensor],
        feature_data: Union[np.ndarray, torch.Tensor],
        property_name: str = "Band Gap",
        feature_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        interactive: bool = False,
        reduction_method: str = "pca",
        cluster_data: bool = True,
        num_clusters: int = 5
    ) -> Union[Figure, go.Figure]:
        """
        Visualize material properties in a feature space.
        
        Args:
            property_values: Property values for each material
            feature_data: Feature vectors for each material
            property_name: Name of the property
            feature_names: Names of features
            title: Plot title
            save_path: Path to save the visualization
            interactive: Whether to return an interactive plotly visualization
            reduction_method: Dimension reduction method for visualization
            cluster_data: Whether to cluster the data
            num_clusters: Number of clusters if clustering
            
        Returns:
            Figure object
        """
        # Convert to numpy arrays
        if isinstance(property_values, torch.Tensor):
            property_values = property_values.detach().cpu().numpy()
        if isinstance(feature_data, torch.Tensor):
            feature_data = feature_data.detach().cpu().numpy()
            
        # Ensure property_values is 1D
        property_values = property_values.reshape(-1)
        
        # Default title if not provided
        if title is None:
            title = f"{property_name} in Material Feature Space"
            
        # Reduce dimensionality for visualization
        reduced_data = self.discovery.dimension_reduction(
            feature_data, 
            method=reduction_method, 
            n_components=2
        )
        
        # Cluster data if requested
        if cluster_data:
            cluster_labels, _ = self.discovery.cluster_data(
                feature_data,
                method="kmeans",
                n_clusters=num_clusters
            )
        else:
            cluster_labels = None
        
        # Create visualization
        if interactive:
            # Plotly interactive visualization
            fig = go.Figure()
            
            # Color by property value
            if cluster_data:
                # Color by cluster with property in hover data
                for i in range(num_clusters):
                    mask = cluster_labels == i
                    fig.add_trace(go.Scatter(
                        x=reduced_data[mask, 0],
                        y=reduced_data[mask, 1],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=f'rgba({i*50}, {255-i*30}, {i*20}, 0.8)'
                        ),
                        text=[f"{property_name}: {val:.3f}" for val in property_values[mask]],
                        name=f"Cluster {i+1}"
                    ))
            else:
                # Direct color mapping to property value
                fig.add_trace(go.Scatter(
                    x=reduced_data[:, 0],
                    y=reduced_data[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=property_values,
                        colorscale='Viridis',
                        colorbar=dict(title=property_name),
                        showscale=True
                    ),
                    text=[f"{property_name}: {val:.3f}" for val in property_values]
                ))
            
            # Set layout
            fig.update_layout(
                title=title,
                xaxis_title=f"{reduction_method.upper()} Component 1",
                yaxis_title=f"{reduction_method.upper()} Component 2",
                hovermode='closest'
            )
            
            # Save if requested
            if save_path:
                if not save_path.endswith('.html'):
                    save_path += '.html'
                fig.write_html(save_path)
                logger.info(f"Saved interactive property map to {save_path}")
                
            return fig
            
        else:
            # Matplotlib static visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot data
            if cluster_data:
                # Color by cluster
                scatter = ax.scatter(
                    reduced_data[:, 0],
                    reduced_data[:, 1],
                    c=cluster_labels,
                    cmap='tab10',
                    s=50,
                    alpha=0.8,
                    edgecolors='k'
                )
                
                # Add legend for clusters
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                            markersize=10, label=f'Cluster {i+1}')
                                 for i in range(num_clusters)]
                ax.legend(handles=legend_elements, loc='upper right')
                
                # Add colorbar for property values as a separate scatter
                sc = ax.scatter(
                    reduced_data[:, 0],
                    reduced_data[:, 1],
                    c=property_values,
                    cmap='viridis',
                    s=0  # Make points invisible
                )
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label(property_name)
                
            else:
                # Direct color mapping to property value
                scatter = ax.scatter(
                    reduced_data[:, 0],
                    reduced_data[:, 1],
                    c=property_values,
                    cmap='viridis',
                    s=50,
                    alpha=0.8,
                    edgecolors='k'
                )
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(property_name)
            
            # Set labels and title
            ax.set_xlabel(f"{reduction_method.upper()} Component 1")
            ax.set_ylabel(f"{reduction_method.upper()} Component 2")
            ax.set_title(title)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved property map to {save_path}")
            
            return fig


class ScientificBioinformaticsVisualizer:
    """
    Specialized visualizations for bioinformatics data.
    """
    def __init__(
        self,
        output_dir: str = "./visualizations/bioinformatics",
        discovery: Optional[PatternDiscovery] = None
    ):
        """
        Initialize the bioinformatics visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            discovery: Pattern discovery module for finding relationships
        """
        self.output_dir = output_dir
        self.discovery = discovery or PatternDiscovery()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Amino acid properties lookup
        self.aa_properties = {
            'A': {'hydropathy': 1.8, 'volume': 88.6, 'charge': 0, 'polarity': 0},
            'R': {'hydropathy': -4.5, 'volume': 173.4, 'charge': 1, 'polarity': 1},
            'N': {'hydropathy': -3.5, 'volume': 114.1, 'charge': 0, 'polarity': 1},
            'D': {'hydropathy': -3.5, 'volume': 111.1, 'charge': -1, 'polarity': 1},
            'C': {'hydropathy': 2.5, 'volume': 108.5, 'charge': 0, 'polarity': 0},
            'Q': {'hydropathy': -3.5, 'volume': 143.8, 'charge': 0, 'polarity': 1},
            'E': {'hydropathy': -3.5, 'volume': 138.4, 'charge': -1, 'polarity': 1},
            'G': {'hydropathy': -0.4, 'volume': 60.1, 'charge': 0, 'polarity': 0},
            'H': {'hydropathy': -3.2, 'volume': 153.2, 'charge': 0, 'polarity': 1},
            'I': {'hydropathy': 4.5, 'volume': 166.7, 'charge': 0, 'polarity': 0},
            'L': {'hydropathy': 3.8, 'volume': 166.7, 'charge': 0, 'polarity': 0},
            'K': {'hydropathy': -3.9, 'volume': 168.6, 'charge': 1, 'polarity': 1},
            'M': {'hydropathy': 1.9, 'volume': 162.9, 'charge': 0, 'polarity': 0},
            'F': {'hydropathy': 2.8, 'volume': 189.9, 'charge': 0, 'polarity': 0},
            'P': {'hydropathy': -1.6, 'volume': 112.7, 'charge': 0, 'polarity': 0},
            'S': {'hydropathy': -0.8, 'volume': 89.0, 'charge': 0, 'polarity': 1},
            'T': {'hydropathy': -0.7, 'volume': 116.1, 'charge': 0, 'polarity': 1},
            'W': {'hydropathy': -0.9, 'volume': 227.8, 'charge': 0, 'polarity': 0},
            'Y': {'hydropathy': -1.3, 'volume': 193.6, 'charge': 0, 'polarity': 1},
            'V': {'hydropathy': 4.2, 'volume': 140.0, 'charge': 0, 'polarity': 0},
        }
        
        # Amino acid color scheme (based on properties)
        self.aa_colors = {
            'A': '#8CFF8C',  # Alanine (hydrophobic) - light green
            'R': '#00007C',  # Arginine (basic) - dark blue
            'N': '#FF7C70',  # Asparagine (polar) - pink
            'D': '#A00042',  # Aspartic acid (acidic) - dark red
            'C': '#FFFF70',  # Cysteine (special) - yellow
            'Q': '#FF4C4C',  # Glutamine (polar) - red
            'E': '#660000',  # Glutamic acid (acidic) - dark red
            'G': '#FFFFFF',  # Glycine (special) - white
            'H': '#7070FF',  # Histidine (basic) - light blue
            'I': '#004C00',  # Isoleucine (hydrophobic) - dark green
            'L': '#455E45',  # Leucine (hydrophobic) - olive green
            'K': '#4747B8',  # Lysine (basic) - medium blue
            'M': '#B8A042',  # Methionine (hydrophobic) - tan
            'F': '#534C42',  # Phenylalanine (aromatic) - brown
            'P': '#525252',  # Proline (special) - gray
            'S': '#FF7042',  # Serine (polar) - orange
            'T': '#B84C00',  # Threonine (polar) - orange-brown
            'W': '#4F4600',  # Tryptophan (aromatic) - dark yellow
            'Y': '#8C704C',  # Tyrosine (aromatic) - light brown
            'V': '#FF8CFF',  # Valine (hydrophobic) - light purple
        }
    
    def visualize_protein_sequence(
        self,
        sequence: str,
        title: str = "Protein Sequence",
        save_path: Optional[str] = None,
        show_properties: bool = True,
        highlight_motifs: Optional[List[str]] = None,
        interactive: bool = False
    ) -> Union[Figure, go.Figure]:
        """
        Visualize a protein sequence with highlighting for properties.
        
        Args:
            sequence: Amino acid sequence
            title: Plot title
            save_path: Path to save the visualization
            show_properties: Whether to show property plots
            highlight_motifs: List of sequence motifs to highlight
            interactive: Whether to return an interactive plotly visualization
            
        Returns:
            Figure object
        """
        # Convert sequence to uppercase
        sequence = sequence.upper()
        
        # Calculate sequence properties along the chain
        if show_properties:
            hydropathy = []
            volume = []
            charge = []
            polarity = []
            
            for aa in sequence:
                if aa in self.aa_properties:
                    hydropathy.append(self.aa_properties[aa]['hydropathy'])
                    volume.append(self.aa_properties[aa]['volume'] / 200.0)  # Normalize by max value
                    charge.append(self.aa_properties[aa]['charge'])
                    polarity.append(self.aa_properties[aa]['polarity'])
                else:
                    # Default values for unknown amino acids
                    hydropathy.append(0)
                    volume.append(0.5)
                    charge.append(0)
                    polarity.append(0)
            
            # Apply a sliding window smoothing for hydropathy (e.g., Kyte-Doolittle)
            window_size = 7
            smooth_hydropathy = np.convolve(hydropathy, np.ones(window_size)/window_size, mode='valid')
            
            # Add padding to match sequence length
            padding = window_size // 2
            smooth_hydropathy = np.pad(smooth_hydropathy, (padding, window_size - padding - 1), mode='edge')
        
        # Find motif positions if requested
        motif_positions = []
        if highlight_motifs:
            for motif in highlight_motifs:
                motif = motif.upper()
                start_pos = 0
                while True:
                    pos = sequence.find(motif, start_pos)
                    if pos == -1:
                        break
                    motif_positions.append((pos, pos + len(motif), motif))
                    start_pos = pos + 1
        
        # Create visualization
        if interactive:
            # Plotly interactive visualization
            if show_properties:
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    subplot_titles=["Amino Acid Sequence", "Hydropathy Profile", "Residue Properties"],
                    vertical_spacing=0.1,
                    row_heights=[0.4, 0.3, 0.3]
                )
            else:
                fig = make_subplots(
                    rows=1, cols=1,
                    subplot_titles=["Amino Acid Sequence"]
                )
            
            # Amino acid sequence visualization
            # Create a bar for each amino acid
            for i, aa in enumerate(sequence):
                color = self.aa_colors.get(aa, '#CCCCCC')
                
                # Check if this position is part of a motif
                in_motif = False
                motif_name = ""
                for start, end, motif in motif_positions:
                    if start <= i < end:
                        in_motif = True
                        motif_name = motif
                        break
                
                # Add border if it's part of a motif
                if in_motif:
                    marker_line = dict(width=2, color='black')
                    opacity = 1.0
                else:
                    marker_line = dict(width=1, color='gray')
                    opacity = 0.8
                
                # Add the bar
                fig.add_trace(
                    go.Bar(
                        x=[i],
                        y=[1],
                        marker_color=color,
                        marker_line=marker_line,
                        opacity=opacity,
                        width=0.8,
                        showlegend=False,
                        hovertext=f"Position {i+1}: {aa}" + (f" (Motif: {motif_name})" if in_motif else "")
                    ),
                    row=1, col=1
                )
            
            # Add property plots if requested
            if show_properties:
                # Hydropathy plot
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(sequence))),
                        y=smooth_hydropathy,
                        mode='lines',
                        line=dict(width=2, color='blue'),
                        name='Hydropathy'
                    ),
                    row=2, col=1
                )
                
                # Add zero line for hydropathy
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=len(sequence),
                    y1=0,
                    line=dict(color="black", width=1, dash="dash"),
                    row=2, col=1
                )
                
                # Bar plots for volume, charge, and polarity
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(sequence))),
                        y=volume,
                        name='Volume',
                        marker_color='green',
                        opacity=0.5,
                        width=0.8
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(sequence))),
                        y=charge,
                        name='Charge',
                        marker_color='red',
                        opacity=0.5,
                        width=0.8
                    ),
                    row=3, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=title,
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=800 if show_properties else 400
            )
            
            fig.update_xaxes(title_text="Position", row=3 if show_properties else 1, col=1)
            fig.update_yaxes(title_text="", showticklabels=False, row=1, col=1)
            
            if show_properties:
                fig.update_yaxes(title_text="Hydropathy", row=2, col=1)
                fig.update_yaxes(title_text="Property Value", row=3, col=1)
            
            # Save if requested
            if save_path:
                if not save_path.endswith('.html'):
                    save_path += '.html'
                fig.write_html(save_path)
                logger.info(f"Saved interactive protein sequence visualization to {save_path}")
                
            return fig
            
        else:
            # Matplotlib static visualization
            if show_properties:
                fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
                ax_seq, ax_hydro, ax_props = axs
            else:
                fig, ax_seq = plt.subplots(figsize=(12, 4))
            
            # Amino acid sequence visualization
            for i, aa in enumerate(sequence):
                color = self.aa_colors.get(aa, '#CCCCCC')
                
                # Check if this position is part of a motif
                in_motif = False
                motif_name = ""
                for start, end, motif in motif_positions:
                    if start <= i < end:
                        in_motif = True
                        motif_name = motif
                        break
                
                # Add a colored rectangle for each amino acid
                rect = plt.Rectangle((i, 0), 1, 1, color=color, alpha=0.8, 
                                   edgecolor='black' if in_motif else 'gray', 
                                   linewidth=2 if in_motif else 0.5)
                ax_seq.add_patch(rect)
                
                # Add amino acid label
                ax_seq.text(i+0.5, 0.5, aa, ha='center', va='center', 
                          fontsize=8, fontweight='bold' if in_motif else 'normal')
            
            # Set sequence axis properties
            ax_seq.set_xlim(0, len(sequence))
            ax_seq.set_ylim(0, 1)
            ax_seq.set_title(title)
            ax_seq.set_ylabel('Sequence')
            ax_seq.set_yticks([])
            
            # Add property plots if requested
            if show_properties:
                # Hydropathy plot
                ax_hydro.plot(smooth_hydropathy, 'b-', linewidth=1.5)
                ax_hydro.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax_hydro.set_ylabel('Hydropathy')
                
                # Highlight hydrophobic and hydrophilic regions
                for i in range(len(smooth_hydropathy)):
                    if smooth_hydropathy[i] > 0:
                        ax_hydro.axvspan(i, i+1, alpha=0.2, color='green')
                    else:
                        ax_hydro.axvspan(i, i+1, alpha=0.1, color='blue')
                
                # Property plots
                ax_props.bar(range(len(sequence)), volume, alpha=0.4, color='green', label='Volume')
                ax_props.bar(range(len(sequence)), charge, alpha=0.4, color='red', label='Charge')
                ax_props.bar(range(len(sequence)), polarity, alpha=0.3, color='purple', label='Polarity')
                
                ax_props.set_ylabel('Properties')
                ax_props.set_xlabel('Position')
                ax_props.legend(loc='upper right')
                
                # Highlight motifs across all plots
                for start, end, motif in motif_positions:
                    ax_seq.axvspan(start, end, alpha=0.3, color='yellow')
                    ax_hydro.axvspan(start, end, alpha=0.3, color='yellow')
                    ax_props.axvspan(start, end, alpha=0.3, color='yellow')
            else:
                ax_seq.set_xlabel('Position')
                
                # Highlight motifs
                for start, end, motif in motif_positions:
                    ax_seq.axvspan(start, end, alpha=0.3, color='yellow')
            
            # Add legend for amino acid properties
            handles = []
            labels = []
            property_groups = {
                'Hydrophobic': ['A', 'I', 'L', 'M', 'F', 'V', 'C'],
                'Polar': ['N', 'Q', 'S', 'T', 'Y'],
                'Acidic': ['D', 'E'],
                'Basic': ['R', 'H', 'K'],
                'Special': ['G', 'P', 'W']
            }
            
            for group, aas in property_groups.items():
                color = self.aa_colors.get(aas[0], '#CCCCCC')
                handles.append(plt.Rectangle((0,0), 1, 1, color=color))
                labels.append(group)
                
            ax_seq.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 1))
            
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved protein sequence visualization to {save_path}")
            
            return fig
    
    def visualize_protein_structure(
        self,
        residues: Union[List[str], str],
        coordinates: Union[np.ndarray, torch.Tensor],
        title: str = "Protein Structure",
        save_path: Optional[str] = None,
        interactive: bool = False,
        color_mode: str = "residue_type",
        highlight_residues: Optional[List[int]] = None,
        show_secondary_structure: bool = True
    ) -> Union[Figure, go.Figure]:
        """
        Visualize a protein structure in 3D.
        
        Args:
            residues: List of amino acid residues or string sequence
            coordinates: 3D coordinates of residues [num_residues, 3]
            title: Plot title
            save_path: Path to save the visualization
            interactive: Whether to return an interactive plotly visualization
            color_mode: How to color residues ("residue_type", "chain", "secondary")
            highlight_residues: List of residue indices to highlight
            show_secondary_structure: Whether to show secondary structure elements
            
        Returns:
            Figure object
        """
        # Convert to numpy arrays
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.detach().cpu().numpy()
            
        # Convert string sequence to list if needed
        if isinstance(residues, str):
            residues = list(residues.upper())
        
        # Make sure coordinates match residues
        num_residues = len(residues)
        if coordinates.shape[0] < num_residues:
            # Truncate residues to match coordinates
            residues = residues[:coordinates.shape[0]]
            num_residues = len(residues)
        elif coordinates.shape[0] > num_residues:
            # Truncate coordinates to match residues
            coordinates = coordinates[:num_residues]
        
        # Detect secondary structure (very simplified version)
        if show_secondary_structure:
            # Calculate distances between residues i and i+3
            secondary_structure = ['C'] * num_residues  # Coil by default
            
            for i in range(num_residues - 3):
                # Check for alpha helix pattern (i to i+3/i+4 distances)
                dist_i_i3 = np.linalg.norm(coordinates[i] - coordinates[i+3])
                if i < num_residues - 4:
                    dist_i_i4 = np.linalg.norm(coordinates[i] - coordinates[i+4])
                else:
                    dist_i_i4 = float('inf')
                
                # Typical i to i+3/i+4 distances in alpha helices
                if dist_i_i3 < 6.0 or dist_i_i4 < 6.5:
                    secondary_structure[i] = 'H'  # Helix
                    secondary_structure[i+1] = 'H'
                    secondary_structure[i+2] = 'H'
                    secondary_structure[i+3] = 'H'
                    if i < num_residues - 4:
                        secondary_structure[i+4] = 'H'
            
            # Detect beta sheets (more challenging without hydrogen bond info)
            # This is just a placeholder logic - real detection would use H-bonds
            for i in range(num_residues - 2):
                if secondary_structure[i] == 'C':
                    # Look for extended stretches
                    extended = True
                    for j in range(1, min(3, num_residues - i)):
                        if i+j < num_residues and np.linalg.norm(coordinates[i] - coordinates[i+j]) < 3.5:
                            extended = False
                            break
                    
                    if extended:
                        secondary_structure[i] = 'E'  # Sheet/Extended
        else:
            secondary_structure = ['C'] * num_residues
        
        # Assign colors based on the selected mode
        colors = []
        for i, aa in enumerate(residues):
            if highlight_residues and i in highlight_residues:
                colors.append('#FF0000')  # Red for highlighted residues
                continue
                
            if color_mode == "residue_type":
                colors.append(self.aa_colors.get(aa, '#CCCCCC'))
            elif color_mode == "secondary":
                if secondary_structure[i] == 'H':
                    colors.append('#FF0000')  # Red for helix
                elif secondary_structure[i] == 'E':
                    colors.append('#FFFF00')  # Yellow for sheet
                else:
                    colors.append('#FFFFFF')  # White for coil
            else:  # "chain" or default
                colors.append('#1E90FF')  # Blue for a single chain
        
        # Create visualization
        if interactive:
            # Plotly interactive visualization
            fig = go.Figure()
            
            # Add residues as spheres
            fig.add_trace(go.Scatter3d(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                z=coordinates[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=colors,
                    symbol='circle',
                    line=dict(color='black', width=0.5)
                ),
                text=[f"{i+1}: {aa}" for i, aa in enumerate(residues)],
                name='Residues'
            ))
            
            # Add backbone connections
            x_line = []
            y_line = []
            z_line = []
            
            for i in range(num_residues - 1):
                # Add line between consecutive residues
                x_line.extend([coordinates[i, 0], coordinates[i+1, 0], None])
                y_line.extend([coordinates[i, 1], coordinates[i+1, 1], None])
                z_line.extend([coordinates[i, 2], coordinates[i+1, 2], None])
            
            fig.add_trace(go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))
            
            # Set layout
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='X (Å)',
                    yaxis_title='Y (Å)',
                    zaxis_title='Z (Å)',
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            # Save if requested
            if save_path:
                if not save_path.endswith('.html'):
                    save_path += '.html'
                fig.write_html(save_path)
                logger.info(f"Saved interactive protein structure visualization to {save_path}")
                
            return fig
            
        else:
            # Matplotlib static visualization
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot residues
            for i, (aa, pos) in enumerate(zip(residues, coordinates)):
                ax.scatter(
                    pos[0], pos[1], pos[2],
                    color=colors[i],
                    s=100,
                    edgecolors='black',
                    alpha=0.8
                )
                
                # Add residue labels
                if highlight_residues and i in highlight_residues:
                    ax.text(pos[0], pos[1], pos[2], f"{i+1}:{aa}", fontsize=8)
            
            # Plot backbone connections
            for i in range(num_residues - 1):
                ax.plot(
                    [coordinates[i, 0], coordinates[i+1, 0]],
                    [coordinates[i, 1], coordinates[i+1, 1]],
                    [coordinates[i, 2], coordinates[i+1, 2]],
                    color='black',
                    linewidth=1,
                    alpha=0.8
                )
            
            # Set labels and title
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.set_title(title)
            
            # Add legend
            if color_mode == "residue_type":
                handles = []
                labels = []
                property_groups = {
                    'Hydrophobic': ['A', 'I', 'L', 'M', 'F', 'V', 'C'],
                    'Polar': ['N', 'Q', 'S', 'T', 'Y'],
                    'Acidic': ['D', 'E'],
                    'Basic': ['R', 'H', 'K'],
                    'Special': ['G', 'P', 'W']
                }
                
                for group, aas in property_groups.items():
                    color = self.aa_colors.get(aas[0], '#CCCCCC')
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10))
                    labels.append(group)
                    
                ax.legend(handles, labels, loc='upper right')
                
            elif color_mode == "secondary":
                handles = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0000', markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFFF00', markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFFFFF', markersize=10)
                ]
                labels = ['Alpha Helix', 'Beta Sheet', 'Coil']
                ax.legend(handles, labels, loc='upper right')
            
            # Equal aspect ratio
            # Get max range for all axes
            max_range = np.array([
                coordinates[:, 0].max() - coordinates[:, 0].min(),
                coordinates[:, 1].max() - coordinates[:, 1].min(),
                coordinates[:, 2].max() - coordinates[:, 2].min()
            ]).max() / 2.0
            
            mid_x = (coordinates[:, 0].max() + coordinates[:, 0].min()) / 2
            mid_y = (coordinates[:, 1].max() + coordinates[:, 1].min()) / 2
            mid_z = (coordinates[:, 2].max() + coordinates[:, 2].min()) / 2
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved protein structure visualization to {save_path}")
            
            return fig
    
    def visualize_genomic_data(
        self,
        genomic_data: Dict[str, Union[np.ndarray, torch.Tensor, str]],
        title: str = "Genomic Data Visualization",
        save_path: Optional[str] = None,
        interactive: bool = False,
        genomic_region: Optional[str] = None
    ) -> Union[Figure, go.Figure]:
        """
        Visualize genomic data (sequences and/or expression values).
        
        Args:
            genomic_data: Dictionary with genomic data
                - sequence: Optional DNA/RNA sequence
                - expression: Optional gene expression values
                - positions: Optional genomic positions
                - gene_names: Optional gene names for expression data
            title: Plot title
            save_path: Path to save the visualization
            interactive: Whether to return an interactive plotly visualization
            genomic_region: Optional description of the genomic region
            
        Returns:
            Figure object
        """
        # Extract data from the dictionary
        sequence = genomic_data.get('sequence', None)
        expression = genomic_data.get('expression', None)
        positions = genomic_data.get('positions', None)
        gene_names = genomic_data.get('gene_names', None)
        
        # Convert to numpy arrays
        if expression is not None and isinstance(expression, torch.Tensor):
            expression = expression.detach().cpu().numpy()
        if positions is not None and isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()
        
        # Create an appropriate visualization based on available data
        if sequence and expression is None:
            # Sequence-only visualization
            return self._visualize_genomic_sequence(
                sequence=sequence,
                positions=positions,
                title=title,
                save_path=save_path,
                interactive=interactive,
                genomic_region=genomic_region
            )
        elif expression is not None and sequence is None:
            # Expression-only visualization
            return self._visualize_gene_expression(
                expression=expression,
                gene_names=gene_names,
                title=title,
                save_path=save_path,
                interactive=interactive
            )
        else:
            # Combined visualization with sequence and expression
            return self._visualize_combined_genomic(
                sequence=sequence,
                expression=expression,
                positions=positions,
                gene_names=gene_names,
                title=title,
                save_path=save_path,
                interactive=interactive,
                genomic_region=genomic_region
            )
    
    def _visualize_genomic_sequence(
        self,
        sequence: str,
        positions: Optional[np.ndarray] = None,
        title: str = "DNA Sequence",
        save_path: Optional[str] = None,
        interactive: bool = False,
        genomic_region: Optional[str] = None
    ) -> Union[Figure, go.Figure]:
        """Helper method for visualizing genomic sequences."""
        # Nucleotide colors
        nt_colors = {
            'A': '#00FF00',  # Green
            'C': '#0000FF',  # Blue
            'G': '#FFB300',  # Orange
            'T': '#FF0000',  # Red
            'U': '#FF0000',  # Red (same as T)
            'N': '#AAAAAA',  # Gray
            '-': '#FFFFFF'   # White
        }
        
        # Count nucleotides
        nt_counts = {nt: sequence.count(nt) for nt in "ACGTN-"}
        total = len(sequence)
        nt_fractions = {nt: count/total for nt, count in nt_counts.items() if count > 0}
        
        # Calculate GC content
        gc_content = (nt_counts.get('G', 0) + nt_counts.get('C', 0)) / total * 100 if total > 0 else 0
        
        # Generate positions if not provided
        if positions is None:
            positions = np.arange(len(sequence))
            
        # Sequence visualization
        if interactive:
            # Plotly interactive visualization
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=["Nucleotide Sequence", "Nucleotide Distribution"],
                vertical_spacing=0.2,
                row_heights=[0.7, 0.3]
            )
            
            # Add bars for nucleotides
            for i, nt in enumerate(sequence):
                color = nt_colors.get(nt, '#CCCCCC')
                
                fig.add_trace(
                    go.Bar(
                        x=[positions[i]],
                        y=[1],
                        marker_color=color,
                        marker_line=dict(width=0.5, color='gray'),
                        opacity=0.9,
                        width=0.8,
                        showlegend=False,
                        hovertext=f"Position {positions[i]}: {nt}"
                    ),
                    row=1, col=1
                )
            
            # Add pie chart for nucleotide distribution
            fig.add_trace(
                go.Pie(
                    labels=list(nt_fractions.keys()),
                    values=list(nt_fractions.values()),
                    hole=0.4,
                    marker_colors=[nt_colors.get(nt, '#CCCCCC') for nt in nt_fractions.keys()],
                    textinfo='label+percent',
                    insidetextorientation='radial'
                ),
                row=2, col=1
            )
            
            # Add GC content annotation
            fig.add_annotation(
                x=0.5, y=0.1,
                xref="paper", yref="paper",
                text=f"GC Content: {gc_content:.1f}%",
                showarrow=False,
                font=dict(size=14, color="black"),
                align="center",
                row=2, col=1
            )
            
            # Set layout
            subtitle = f" - {genomic_region}" if genomic_region else ""
            fig.update_layout(
                title=f"{title}{subtitle}",
                height=600,
                showlegend=False
            )
            
            fig.update_xaxes(title_text="Position", row=1, col=1)
            fig.update_yaxes(title_text="", showticklabels=False, row=1, col=1)
            
            # Save if requested
            if save_path:
                if not save_path.endswith('.html'):
                    save_path += '.html'
                fig.write_html(save_path)
                logger.info(f"Saved interactive genomic sequence visualization to {save_path}")
                
            return fig
            
        else:
            # Matplotlib static visualization
            fig, (ax_seq, ax_pie) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
            
            # Nucleotide sequence visualization
            for i, nt in enumerate(sequence):
                color = nt_colors.get(nt, '#CCCCCC')
                
                rect = plt.Rectangle((positions[i], 0), 1, 1, color=color, 
                                   edgecolor='gray', linewidth=0.5, alpha=0.9)
                ax_seq.add_patch(rect)
            
            # Set sequence axis properties
            ax_seq.set_xlim(positions[0], positions[-1])
            ax_seq.set_ylim(0, 1)
            subtitle = f" - {genomic_region}" if genomic_region else ""
            ax_seq.set_title(f"{title}{subtitle}")
            ax_seq.set_ylabel('Sequence')
            ax_seq.set_yticks([])
            
            # Add nucleotide distribution pie chart
            wedges, texts, autotexts = ax_pie.pie(
                nt_fractions.values(), 
                labels=nt_fractions.keys(),
                colors=[nt_colors.get(nt, '#CCCCCC') for nt in nt_fractions.keys()],
                autopct='%1.1f%%',
                startangle=90
            )
            
            # Equal aspect ratio for pie chart
            ax_pie.axis('equal')
            
            # Add GC content annotation
            ax_pie.annotate(
                f"GC Content: {gc_content:.1f}%",
                xy=(0.5, 0),
                xycoords='axes fraction',
                xytext=(0, -20),
                textcoords='offset points',
                ha='center',
                va='top',
                fontsize=12
            )
            
            # Add legend for nucleotides
            handles = []
            labels = []
            for nt, color in nt_colors.items():
                if nt in nt_counts and nt_counts[nt] > 0:
                    handles.append(plt.Rectangle((0,0), 1, 1, color=color))
                    labels.append(nt)
                    
            ax_seq.legend(handles, labels, loc='upper right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved genomic sequence visualization to {save_path}")
            
            return fig
    
    def _visualize_gene_expression(
        self,
        expression: np.ndarray,
        gene_names: Optional[List[str]] = None,
        title: str = "Gene Expression",
        save_path: Optional[str] = None,
        interactive: bool = False
    ) -> Union[Figure, go.Figure]:
        """Helper method for visualizing gene expression data."""
        # Generate gene names if not provided
        num_genes = len(expression)
        if gene_names is None:
            gene_names = [f"Gene_{i+1}" for i in range(num_genes)]
            
        # Truncate long gene lists for visibility
        max_display = 50
        if num_genes > max_display:
            # Find top expressed genes
            top_indices = np.argsort(expression)[-max_display:]
            expression = expression[top_indices]
            gene_names = [gene_names[i] for i in top_indices]
            num_genes = max_display
        
        # Sort by expression level
        sort_indices = np.argsort(expression)
        expression = expression[sort_indices]
        gene_names = [gene_names[i] for i in sort_indices]
        
        # Create visualization
        if interactive:
            # Plotly interactive visualization
            fig = go.Figure()
            
            # Add expression bars
            fig.add_trace(go.Bar(
                x=expression,
                y=gene_names,
                orientation='h',
                marker=dict(
                    color=expression,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Expression')
                ),
                hovertemplate='%{y}: %{x:.2f}<extra></extra>'
            ))
            
            # Set layout
            fig.update_layout(
                title=title,
                xaxis_title='Expression Level',
                yaxis_title='Gene',
                height=max(500, num_genes * 20),
                margin=dict(l=150)  # More space for gene names
            )
            
            # Save if requested
            if save_path:
                if not save_path.endswith('.html'):
                    save_path += '.html'
                fig.write_html(save_path)
                logger.info(f"Saved interactive gene expression visualization to {save_path}")
                
            return fig
            
        else:
            # Matplotlib static visualization
            fig, ax = plt.subplots(figsize=(10, max(8, num_genes * 0.25)))
            
            # Create horizontal bar chart
            bars = ax.barh(
                range(num_genes),
                expression,
                color=plt.cm.viridis(expression / max(expression))
            )
            
            # Add labels
            ax.set_yticks(range(num_genes))
            ax.set_yticklabels(gene_names)
            ax.set_xlabel('Expression Level')
            ax.set_title(title)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(min(expression), max(expression)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Expression')
            
            # Adjust layout for gene names
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved gene expression visualization to {save_path}")
            
            return fig
    
    def _visualize_combined_genomic(
        self,
        sequence: str,
        expression: np.ndarray,
        positions: Optional[np.ndarray] = None,
        gene_names: Optional[List[str]] = None,
        title: str = "Genomic Data",
        save_path: Optional[str] = None,
        interactive: bool = False,
        genomic_region: Optional[str] = None
    ) -> Union[Figure, go.Figure]:
        """Helper method for visualizing combined genomic data."""
        # Implementation depends on specific data relationships
        # This is a simplified example - real implementation would need more context
        
        # Nucleotide colors
        nt_colors = {
            'A': '#00FF00',  # Green
            'C': '#0000FF',  # Blue
            'G': '#FFB300',  # Orange
            'T': '#FF0000',  # Red
            'U': '#FF0000',  # Red (same as T)
            'N': '#AAAAAA',  # Gray
            '-': '#FFFFFF'   # White
        }
        
        # Generate positions if not provided
        if positions is None:
            positions = np.arange(len(sequence))
            
        # Generate gene names if not provided
        num_genes = len(expression)
        if gene_names is None:
            gene_names = [f"Gene_{i+1}" for i in range(num_genes)]
        
        # Create visualization
        if interactive:
            # Plotly interactive visualization
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=["DNA Sequence", "Gene Expression"],
                vertical_spacing=0.2,
                row_heights=[0.5, 0.5]
            )
            
            # Add sequence bars
            for i, nt in enumerate(sequence[:500]):  # Limit to first 500 bases for visualization
                color = nt_colors.get(nt, '#CCCCCC')
                
                fig.add_trace(
                    go.Bar(
                        x=[positions[i]],
                        y=[1],
                        marker_color=color,
                        marker_line=dict(width=0.5, color='gray'),
                        opacity=0.9,
                        width=0.8,
                        showlegend=False,
                        hovertext=f"Position {positions[i]}: {nt}"
                    ),
                    row=1, col=1
                )
            
            # Add expression scatter plot
            fig.add_trace(
                go.Scatter(
                    x=np.arange(num_genes),
                    y=expression,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=expression,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Expression')
                    ),
                    text=gene_names,
                    hovertemplate='%{text}: %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Set layout
            subtitle = f" - {genomic_region}" if genomic_region else ""
            fig.update_layout(
                title=f"{title}{subtitle}",
                height=700
            )
            
            fig.update_xaxes(title_text="Position", row=1, col=1)
            fig.update_yaxes(title_text="", showticklabels=False, row=1, col=1)
            fig.update_xaxes(title_text="Gene Index", row=2, col=1)
            fig.update_yaxes(title_text="Expression Level", row=2, col=1)
            
            # Save if requested
            if save_path:
                if not save_path.endswith('.html'):
                    save_path += '.html'
                fig.write_html(save_path)
                logger.info(f"Saved interactive combined genomic visualization to {save_path}")
                
            return fig
            
        else:
            # Matplotlib static visualization
            fig, (ax_seq, ax_expr) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 2])
            
            # Nucleotide sequence visualization (first 500 bases)
            for i, nt in enumerate(sequence[:500]):
                color = nt_colors.get(nt, '#CCCCCC')
                
                rect = plt.Rectangle((positions[i], 0), 1, 1, color=color, 
                                   edgecolor='gray', linewidth=0.5, alpha=0.9)
                ax_seq.add_patch(rect)
            
            # Set sequence axis properties
            ax_seq.set_xlim(positions[0], positions[0] + 500)
            ax_seq.set_ylim(0, 1)
            subtitle = f" - {genomic_region}" if genomic_region else ""
            ax_seq.set_title(f"{title}{subtitle}")
            ax_seq.set_ylabel('Sequence')
            ax_seq.set_yticks([])
            
            # Add nucleotide legend
            handles = []
            labels = []
            for nt, color in nt_colors.items():
                if nt in sequence:
                    handles.append(plt.Rectangle((0,0), 1, 1, color=color))
                    labels.append(nt)
                    
            ax_seq.legend(handles, labels, loc='upper right')
            
            # Gene expression visualization
            scatter = ax_expr.scatter(
                np.arange(num_genes),
                expression,
                c=expression,
                cmap='viridis',
                s=50,
                alpha=0.8
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax_expr)
            cbar.set_label('Expression Level')
            
            # Set expression axis properties
            ax_expr.set_xlabel('Gene Index')
            ax_expr.set_ylabel('Expression Level')
            
            # Label top expressed genes
            top_indices = np.argsort(expression)[-5:]  # Top 5 expressed genes
            for idx in top_indices:
                ax_expr.annotate(
                    gene_names[idx],
                    xy=(idx, expression[idx]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8
                )
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved combined genomic visualization to {save_path}")
            
            return fig


class ScientificDomainVisualizer:
    """
    Factory class for scientific domain-specific visualizers.
    """
    def __init__(
        self,
        discovery: Optional[PatternDiscovery] = None,
        output_dir: str = "./visualizations"
    ):
        """
        Initialize the scientific domain visualizer.
        
        Args:
            discovery: Pattern discovery module
            output_dir: Base directory for visualizations
        """
        self.discovery = discovery or PatternDiscovery()
        self.output_dir = output_dir
        
        # Create domain-specific visualizers
        self.materials_viz = ScientificMaterialsVisualizer(
            output_dir=os.path.join(output_dir, "materials"),
            discovery=self.discovery
        )
        
        self.bio_viz = ScientificBioinformaticsVisualizer(
            output_dir=os.path.join(output_dir, "bioinformatics"),
            discovery=self.discovery
        )
        
        # Use generic visualizer for other cases
        self.generic_viz = ScientificVisualizer(
            output_dir=output_dir
        )
    
    def get_visualizer(self, domain: str) -> Union[ScientificMaterialsVisualizer, ScientificBioinformaticsVisualizer, ScientificVisualizer]:
        """
        Get a domain-specific visualizer.
        
        Args:
            domain: Scientific domain ("materials", "bio", or other)
            
        Returns:
            Domain-specific visualizer
        """
        if domain.lower() == "materials":
            return self.materials_viz
        elif domain.lower() in ["bio", "bioinformatics"]:
            return self.bio_viz
        else:
            return self.generic_viz