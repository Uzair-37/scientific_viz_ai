"""
Visualization generation module for scientific data.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..models.multimodal_model import DiffusionModel

logger = logging.getLogger(__name__)

class ScientificVisualizer:
    """
    Generate visualizations from scientific data using generative models.
    """
    def __init__(
        self,
        diffusion_model: Optional[DiffusionModel] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        output_dir: str = "./visualizations"
    ):
        """
        Initialize the visualizer.
        
        Args:
            diffusion_model: Diffusion model for generating visualizations
            device: Device to run inference on
            output_dir: Directory to save visualizations
        """
        self.diffusion_model = diffusion_model
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Move model to device if provided
        if diffusion_model is not None:
            self.diffusion_model.to(device)
            self.diffusion_model.eval()
    
    def load_diffusion_model(self, model_path: str) -> None:
        """
        Load a diffusion model from a checkpoint.
        
        Args:
            model_path: Path to the diffusion model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.diffusion_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.diffusion_model.load_state_dict(checkpoint)
            
        self.diffusion_model.eval()
        logger.info(f"Loaded diffusion model from {model_path}")
    
    def generate_visualization(
        self,
        conditioning: torch.Tensor,
        num_steps: int = 1000,
        seed: Optional[int] = None,
        viz_type: str = "image",
        save_path: Optional[str] = None
    ) -> Union[np.ndarray, Figure, Dict[str, Any]]:
        """
        Generate a visualization using the diffusion model.
        
        Args:
            conditioning: Conditioning input for the diffusion model
            num_steps: Number of diffusion steps
            seed: Random seed for reproducibility
            viz_type: Type of visualization to generate (image, plot, interactive)
            save_path: Path to save the visualization (optional)
            
        Returns:
            Generated visualization (format depends on viz_type)
        """
        if self.diffusion_model is None:
            raise ValueError("Diffusion model not provided")
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Move conditioning to device
        conditioning = conditioning.to(self.device)
        
        # Generate image using diffusion process
        # This is a simplified implementation - a full implementation would include
        # the complete diffusion sampling algorithm
        with torch.no_grad():
            # Start with random noise
            batch_size = conditioning.shape[0]
            shape = (batch_size, 3, 256, 256)  # Assuming 256x256 RGB images
            x = torch.randn(shape, device=self.device)
            
            # Simplified reverse diffusion process
            for i in range(num_steps, 0, -1):
                t = torch.full((batch_size,), i / num_steps, device=self.device)
                
                # Predict noise or x0
                predicted = self.diffusion_model(x, t, conditioning)
                
                # Update x (simplified)
                # In a full implementation, this would follow the proper updating rule
                # based on the specific diffusion algorithm (DDPM, DDIM, etc.)
                alpha = 0.99 ** i
                x = alpha * predicted + (1 - alpha) * x
        
        # Process output based on visualization type
        if viz_type == "image":
            # Convert to numpy image
            images = x.cpu().numpy()
            # Convert from CxHxW to HxWxC and normalize to [0, 1]
            images = np.transpose(images, (0, 2, 3, 1))
            images = (images - images.min()) / (images.max() - images.min())
            
            # Save if path provided
            if save_path is not None:
                for i, img in enumerate(images):
                    img_path = f"{save_path}_{i}.png" if batch_size > 1 else save_path
                    plt.imsave(img_path, img)
                    logger.info(f"Saved visualization to {img_path}")
            
            return images
            
        elif viz_type == "plot":
            # Create matplotlib plot
            fig, axes = plt.subplots(1, batch_size, figsize=(5 * batch_size, 5))
            if batch_size == 1:
                axes = [axes]
                
            images = x.cpu().numpy()
            images = np.transpose(images, (0, 2, 3, 1))
            images = (images - images.min()) / (images.max() - images.min())
            
            for i, (ax, img) in enumerate(zip(axes, images)):
                ax.imshow(img)
                ax.set_title(f"Generated Visualization {i+1}")
                ax.axis("off")
            
            # Save if path provided
            if save_path is not None:
                fig.savefig(save_path, bbox_inches="tight", dpi=300)
                logger.info(f"Saved visualization to {save_path}")
            
            return fig
            
        elif viz_type == "interactive":
            # Create interactive Plotly visualization
            images = x.cpu().numpy()
            images = np.transpose(images, (0, 2, 3, 1))
            images = (images - images.min()) / (images.max() - images.min())
            
            fig = make_subplots(rows=1, cols=batch_size, 
                               subplot_titles=[f"Generated Visualization {i+1}" for i in range(batch_size)])
            
            for i, img in enumerate(images):
                fig.add_trace(
                    go.Image(z=img),
                    row=1, col=i+1
                )
            
            fig.update_layout(
                title="Generated Scientific Visualizations",
                showlegend=False
            )
            
            # Save if path provided
            if save_path is not None:
                fig.write_html(save_path)
                logger.info(f"Saved interactive visualization to {save_path}")
            
            return fig
            
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
    
    def create_scientific_plot(
        self,
        data: Dict[str, np.ndarray],
        plot_type: str = "scatter",
        title: str = "Scientific Visualization",
        labels: Dict[str, str] = {},
        colormap: str = "viridis",
        save_path: Optional[str] = None,
        interactive: bool = False
    ) -> Union[Figure, Dict[str, Any]]:
        """
        Create a scientific plot from data.
        
        Args:
            data: Dictionary of data arrays
            plot_type: Type of plot (scatter, line, heatmap, etc.)
            title: Plot title
            labels: Dictionary of axis labels
            colormap: Colormap for plots that use color
            save_path: Path to save the plot (optional)
            interactive: Whether to return an interactive Plotly plot
            
        Returns:
            Matplotlib Figure or Plotly figure
        """
        if interactive:
            # Create interactive Plotly plot
            if plot_type == "scatter":
                fig = go.Figure()
                
                for name, values in data.items():
                    if "x" in values and "y" in values:
                        scatter_kwargs = {"x": values["x"], "y": values["y"], "name": name}
                        
                        if "z" in values:
                            # 3D scatter plot
                            fig = go.Figure(data=[go.Scatter3d(
                                x=values["x"], y=values["y"], z=values["z"],
                                name=name,
                                mode="markers",
                                marker=dict(
                                    size=5,
                                    color=values.get("color", None),
                                    colorscale=colormap
                                )
                            )])
                        else:
                            # 2D scatter plot
                            fig.add_trace(go.Scatter(
                                **scatter_kwargs,
                                mode="markers",
                                marker=dict(
                                    size=8,
                                    color=values.get("color", None),
                                    colorscale=colormap
                                )
                            ))
                
                # Add labels
                fig.update_layout(
                    title=title,
                    xaxis_title=labels.get("x", "X"),
                    yaxis_title=labels.get("y", "Y"),
                    legend_title=labels.get("legend", "Legend")
                )
                
            elif plot_type == "line":
                fig = go.Figure()
                
                for name, values in data.items():
                    if "x" in values and "y" in values:
                        fig.add_trace(go.Scatter(
                            x=values["x"], 
                            y=values["y"],
                            name=name,
                            mode="lines+markers"
                        ))
                
                # Add labels
                fig.update_layout(
                    title=title,
                    xaxis_title=labels.get("x", "X"),
                    yaxis_title=labels.get("y", "Y"),
                    legend_title=labels.get("legend", "Legend")
                )
                
            elif plot_type == "heatmap":
                if "z" in data:
                    fig = go.Figure(data=go.Heatmap(
                        z=data["z"],
                        x=data.get("x", None),
                        y=data.get("y", None),
                        colorscale=colormap
                    ))
                    
                    # Add labels
                    fig.update_layout(
                        title=title,
                        xaxis_title=labels.get("x", "X"),
                        yaxis_title=labels.get("y", "Y")
                    )
                else:
                    raise ValueError("Heatmap requires 'z' data")
                    
            elif plot_type == "surface":
                if "z" in data:
                    fig = go.Figure(data=go.Surface(
                        z=data["z"],
                        x=data.get("x", None),
                        y=data.get("y", None),
                        colorscale=colormap
                    ))
                    
                    # Add labels
                    fig.update_layout(
                        title=title,
                        scene=dict(
                            xaxis_title=labels.get("x", "X"),
                            yaxis_title=labels.get("y", "Y"),
                            zaxis_title=labels.get("z", "Z")
                        )
                    )
                else:
                    raise ValueError("Surface plot requires 'z' data")
            
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            # Save if path provided
            if save_path is not None:
                fig.write_html(save_path)
                logger.info(f"Saved interactive plot to {save_path}")
            
            return fig
            
        else:
            # Create static Matplotlib plot
            if plot_type == "scatter":
                fig, ax = plt.subplots(figsize=(10, 8))
                
                for name, values in data.items():
                    if "x" in values and "y" in values:
                        scatter_kwargs = {"x": values["x"], "y": values["y"], "label": name}
                        
                        if "z" in values:
                            # 3D scatter plot
                            fig = plt.figure(figsize=(10, 8))
                            ax = fig.add_subplot(111, projection="3d")
                            sc = ax.scatter(
                                values["x"], values["y"], values["z"],
                                c=values.get("color", None),
                                cmap=colormap,
                                s=30,
                                alpha=0.7,
                                label=name
                            )
                        else:
                            # 2D scatter plot
                            sc = ax.scatter(
                                values["x"], values["y"],
                                c=values.get("color", None),
                                cmap=colormap,
                                s=50,
                                alpha=0.7,
                                label=name
                            )
                
                # Add colorbar if color data is provided
                for name, values in data.items():
                    if "color" in values:
                        plt.colorbar(sc, ax=ax, label=labels.get("color", "Value"))
                        break
                
                # Add labels
                ax.set_title(title)
                ax.set_xlabel(labels.get("x", "X"))
                ax.set_ylabel(labels.get("y", "Y"))
                if "z" in next(iter(data.values())):
                    ax.set_zlabel(labels.get("z", "Z"))
                ax.legend()
                
            elif plot_type == "line":
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for name, values in data.items():
                    if "x" in values and "y" in values:
                        ax.plot(values["x"], values["y"], label=name, marker="o")
                
                # Add labels
                ax.set_title(title)
                ax.set_xlabel(labels.get("x", "X"))
                ax.set_ylabel(labels.get("y", "Y"))
                ax.legend()
                
            elif plot_type == "heatmap":
                fig, ax = plt.subplots(figsize=(10, 8))
                
                if "z" in data:
                    im = ax.imshow(
                        data["z"],
                        cmap=colormap,
                        aspect="auto",
                        origin="lower"
                    )
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax)
                    
                    # Set x and y ticks if provided
                    if "x" in data:
                        ax.set_xticks(np.arange(len(data["x"])))
                        ax.set_xticklabels(data["x"])
                        
                    if "y" in data:
                        ax.set_yticks(np.arange(len(data["y"])))
                        ax.set_yticklabels(data["y"])
                    
                    # Add labels
                    ax.set_title(title)
                    ax.set_xlabel(labels.get("x", "X"))
                    ax.set_ylabel(labels.get("y", "Y"))
                else:
                    raise ValueError("Heatmap requires 'z' data")
                    
            elif plot_type == "surface":
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")
                
                if "z" in data:
                    # Create meshgrid if x and y are provided
                    if "x" in data and "y" in data:
                        X, Y = np.meshgrid(data["x"], data["y"])
                    else:
                        X, Y = np.meshgrid(
                            np.arange(data["z"].shape[1]),
                            np.arange(data["z"].shape[0])
                        )
                    
                    # Plot surface
                    surf = ax.plot_surface(
                        X, Y, data["z"],
                        cmap=colormap,
                        linewidth=0,
                        antialiased=True
                    )
                    
                    # Add colorbar
                    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                    
                    # Add labels
                    ax.set_title(title)
                    ax.set_xlabel(labels.get("x", "X"))
                    ax.set_ylabel(labels.get("y", "Y"))
                    ax.set_zlabel(labels.get("z", "Z"))
                else:
                    raise ValueError("Surface plot requires 'z' data")
            
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            # Save if path provided
            if save_path is not None:
                fig.savefig(save_path, bbox_inches="tight", dpi=300)
                logger.info(f"Saved plot to {save_path}")
            
            return fig