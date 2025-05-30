"""
Financial visualization module for generating insightful visualizations of financial data and model outputs.
"""
import os
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

class FinancialVisualizer:
    """
    Financial visualization tools for time series, uncertainty, and multi-modal insights.
    """
    
    def __init__(
        self,
        output_dir: str = "./visualizations",
        style: str = "seaborn-v0_8-darkgrid",
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100,
        colormap: str = "viridis",
        interactive: bool = True
    ):
        """
        Initialize the financial visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            style: Matplotlib style
            figsize: Default figure size
            dpi: Figure resolution
            colormap: Default colormap
            interactive: Whether to use interactive plots (Plotly)
        """
        self.output_dir = output_dir
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        self.colormap = colormap
        self.interactive = interactive
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use(style)
    
    def plot_time_series_with_uncertainty(
        self,
        dates: Union[List[str], pd.DatetimeIndex],
        values: np.ndarray,
        uncertainty: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Financial Time Series with Uncertainty",
        xlabel: str = "Date",
        ylabel: str = "Value",
        save_path: Optional[str] = None,
        show_fig: bool = True
    ) -> Union[Figure, go.Figure]:
        """
        Plot time series data with uncertainty intervals.
        
        Args:
            dates: Dates for x-axis
            values: Y values (can be multi-dimensional for multiple series)
            uncertainty: Uncertainty values as [lower_bound, upper_bound] or standard deviation
            labels: Labels for different series
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save the figure
            show_fig: Whether to display the figure
            
        Returns:
            Matplotlib Figure or Plotly Figure
        """
        # Convert dates to proper format if they're strings
        if isinstance(dates, list) and isinstance(dates[0], str):
            dates = pd.to_datetime(dates)
        
        # Convert values and uncertainty to numpy arrays
        values = np.asarray(values)
        uncertainty = np.asarray(uncertainty)
        
        # Reshape values if needed
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        
        # Create labels if not provided
        if labels is None:
            labels = [f"Series {i+1}" for i in range(values.shape[1])]
        
        if self.interactive:
            return self._plot_time_series_with_uncertainty_plotly(
                dates, values, uncertainty, labels, title, xlabel, ylabel, save_path, show_fig
            )
        else:
            return self._plot_time_series_with_uncertainty_mpl(
                dates, values, uncertainty, labels, title, xlabel, ylabel, save_path, show_fig
            )
    
    def _plot_time_series_with_uncertainty_mpl(
        self,
        dates: pd.DatetimeIndex,
        values: np.ndarray,
        uncertainty: np.ndarray,
        labels: List[str],
        title: str,
        xlabel: str,
        ylabel: str,
        save_path: Optional[str],
        show_fig: bool
    ) -> Figure:
        """Plot time series with uncertainty using Matplotlib."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        colors = plt.cm.get_cmap(self.colormap, values.shape[1])
        
        for i in range(values.shape[1]):
            # Plot the main line
            ax.plot(dates, values[:, i], label=labels[i], color=colors(i))
            
            # Plot uncertainty
            if uncertainty.ndim == 2:  # Single std dev value
                ax.fill_between(
                    dates,
                    values[:, i] - uncertainty[:, i],
                    values[:, i] + uncertainty[:, i],
                    alpha=0.2,
                    color=colors(i)
                )
            elif uncertainty.ndim == 3:  # [lower, upper] bounds
                ax.fill_between(
                    dates,
                    uncertainty[:, 0, i],
                    uncertainty[:, 1, i],
                    alpha=0.2,
                    color=colors(i)
                )
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        
        # Rotate date labels for better readability
        fig.autofmt_xdate()
        
        # Adjust layout
        fig.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path)
            logger.info(f"Saved figure to {save_path}")
        
        # Show if requested
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _plot_time_series_with_uncertainty_plotly(
        self,
        dates: pd.DatetimeIndex,
        values: np.ndarray,
        uncertainty: np.ndarray,
        labels: List[str],
        title: str,
        xlabel: str,
        ylabel: str,
        save_path: Optional[str],
        show_fig: bool
    ) -> go.Figure:
        """Plot time series with uncertainty using Plotly."""
        fig = go.Figure()
        
        # Create color scale
        import plotly.express as px
        colors = px.colors.sample_colorscale(
            self.colormap, 
            np.linspace(0, 1, values.shape[1])
        )
        
        for i in range(values.shape[1]):
            # Plot the main line
            fig.add_trace(go.Scatter(
                x=dates,
                y=values[:, i],
                mode='lines',
                name=labels[i],
                line=dict(color=colors[i])
            ))
            
            # Plot uncertainty
            if uncertainty.ndim == 2:  # Single std dev value
                fig.add_trace(go.Scatter(
                    x=np.concatenate([dates, dates[::-1]]),
                    y=np.concatenate([
                        values[:, i] + uncertainty[:, i],
                        (values[:, i] - uncertainty[:, i])[::-1]
                    ]),
                    fill='toself',
                    fillcolor=colors[i].replace('rgb', 'rgba').replace(')', ', 0.2)'),
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name=f"{labels[i]} Uncertainty"
                ))
            elif uncertainty.ndim == 3:  # [lower, upper] bounds
                fig.add_trace(go.Scatter(
                    x=np.concatenate([dates, dates[::-1]]),
                    y=np.concatenate([
                        uncertainty[:, 1, i],
                        uncertainty[:, 0, i][::-1]
                    ]),
                    fill='toself',
                    fillcolor=colors[i].replace('rgb', 'rgba').replace(')', ', 0.2)'),
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name=f"{labels[i]} Uncertainty"
                ))
        
        # Set layout
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            legend_title="Series",
            hovermode="x unified"
        )
        
        # Save if path provided
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Saved figure to {save_path}")
        
        # Show if requested
        if show_fig:
            fig.show()
        
        return fig
    
    def plot_return_distribution(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        title: str = "Return Distribution",
        xlabel: str = "Return",
        ylabel: str = "Frequency",
        save_path: Optional[str] = None,
        show_fig: bool = True,
        kde: bool = True
    ) -> Union[Figure, go.Figure]:
        """
        Plot distribution of returns with optional benchmark comparison.
        
        Args:
            returns: Array of returns (can be multi-dimensional for multiple series)
            benchmark_returns: Optional benchmark returns for comparison
            labels: Labels for different series
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save the figure
            show_fig: Whether to display the figure
            kde: Whether to show kernel density estimate
            
        Returns:
            Matplotlib Figure or Plotly Figure
        """
        # Convert to numpy array if needed
        returns = np.asarray(returns)
        
        # Reshape returns if needed
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        
        # Create labels if not provided
        if labels is None:
            labels = [f"Strategy {i+1}" for i in range(returns.shape[1])]
            if benchmark_returns is not None:
                labels.append("Benchmark")
        
        if self.interactive:
            return self._plot_return_distribution_plotly(
                returns, benchmark_returns, labels, title, xlabel, ylabel, save_path, show_fig, kde
            )
        else:
            return self._plot_return_distribution_mpl(
                returns, benchmark_returns, labels, title, xlabel, ylabel, save_path, show_fig, kde
            )
    
    def _plot_return_distribution_mpl(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray],
        labels: List[str],
        title: str,
        xlabel: str,
        ylabel: str,
        save_path: Optional[str],
        show_fig: bool,
        kde: bool
    ) -> Figure:
        """Plot return distribution using Matplotlib."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        colors = plt.cm.get_cmap(self.colormap, returns.shape[1] + (1 if benchmark_returns is not None else 0))
        
        for i in range(returns.shape[1]):
            sns.histplot(
                returns[:, i],
                kde=kde,
                label=labels[i],
                color=colors(i),
                alpha=0.5,
                ax=ax
            )
        
        # Add benchmark if provided
        if benchmark_returns is not None:
            benchmark_idx = returns.shape[1]
            sns.histplot(
                benchmark_returns,
                kde=kde,
                label=labels[benchmark_idx] if benchmark_idx < len(labels) else "Benchmark",
                color=colors(benchmark_idx),
                alpha=0.5,
                ax=ax
            )
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        
        # Add a vertical line at 0
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path)
            logger.info(f"Saved figure to {save_path}")
        
        # Show if requested
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _plot_return_distribution_plotly(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray],
        labels: List[str],
        title: str,
        xlabel: str,
        ylabel: str,
        save_path: Optional[str],
        show_fig: bool,
        kde: bool
    ) -> go.Figure:
        """Plot return distribution using Plotly."""
        fig = go.Figure()
        
        # Create color scale
        import plotly.express as px
        n_colors = returns.shape[1] + (1 if benchmark_returns is not None else 0)
        colors = px.colors.sample_colorscale(
            self.colormap, 
            np.linspace(0, 1, n_colors)
        )
        
        # Calculate bin settings for consistency
        all_returns = np.concatenate([returns.flatten()] + 
                                     ([benchmark_returns] if benchmark_returns is not None else []))
        min_val, max_val = np.min(all_returns), np.max(all_returns)
        bin_width = (max_val - min_val) / 30  # 30 bins
        
        for i in range(returns.shape[1]):
            fig.add_trace(go.Histogram(
                x=returns[:, i],
                name=labels[i],
                marker_color=colors[i],
                opacity=0.6,
                xbins=dict(
                    start=min_val,
                    end=max_val,
                    size=bin_width
                ),
                histnorm='probability'
            ))
            
            # Add KDE if requested
            if kde:
                kde_x, kde_y = self._compute_kde(returns[:, i])
                fig.add_trace(go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    mode='lines',
                    name=f"{labels[i]} KDE",
                    line=dict(color=colors[i])
                ))
        
        # Add benchmark if provided
        if benchmark_returns is not None:
            benchmark_idx = returns.shape[1]
            fig.add_trace(go.Histogram(
                x=benchmark_returns,
                name=labels[benchmark_idx] if benchmark_idx < len(labels) else "Benchmark",
                marker_color=colors[benchmark_idx],
                opacity=0.6,
                xbins=dict(
                    start=min_val,
                    end=max_val,
                    size=bin_width
                ),
                histnorm='probability'
            ))
            
            # Add KDE if requested
            if kde:
                kde_x, kde_y = self._compute_kde(benchmark_returns)
                fig.add_trace(go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    mode='lines',
                    name="Benchmark KDE",
                    line=dict(color=colors[benchmark_idx])
                ))
        
        # Add vertical line at 0
        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
        
        # Set layout
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            legend_title="Series",
            barmode='overlay'
        )
        
        # Save if path provided
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Saved figure to {save_path}")
        
        # Show if requested
        if show_fig:
            fig.show()
        
        return fig
    
    def _compute_kde(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute kernel density estimate for plotting."""
        from scipy.stats import gaussian_kde
        
        # Compute KDE
        kde = gaussian_kde(data)
        
        # Create x values
        x_min, x_max = np.min(data), np.max(data)
        x_range = x_max - x_min
        x = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)
        
        # Compute y values
        y = kde(x)
        
        return x, y
    
    def plot_correlation_matrix(
        self,
        correlation_matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Correlation Matrix",
        save_path: Optional[str] = None,
        show_fig: bool = True,
        cmap: str = "coolwarm",
        annot: bool = True
    ) -> Union[Figure, go.Figure]:
        """
        Plot correlation matrix heatmap.
        
        Args:
            correlation_matrix: Correlation matrix
            labels: Labels for the variables
            title: Plot title
            save_path: Path to save the figure
            show_fig: Whether to display the figure
            cmap: Colormap to use
            annot: Whether to annotate the heatmap with correlation values
            
        Returns:
            Matplotlib Figure or Plotly Figure
        """
        # Create labels if not provided
        if labels is None:
            labels = [f"Var {i+1}" for i in range(correlation_matrix.shape[0])]
        
        if self.interactive:
            return self._plot_correlation_matrix_plotly(
                correlation_matrix, labels, title, save_path, show_fig
            )
        else:
            return self._plot_correlation_matrix_mpl(
                correlation_matrix, labels, title, save_path, show_fig, cmap, annot
            )
    
    def _plot_correlation_matrix_mpl(
        self,
        correlation_matrix: np.ndarray,
        labels: List[str],
        title: str,
        save_path: Optional[str],
        show_fig: bool,
        cmap: str,
        annot: bool
    ) -> Figure:
        """Plot correlation matrix using Matplotlib."""
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            annot=annot,
            fmt=".2f" if annot else None,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels
        )
        
        # Set title
        ax.set_title(title)
        
        # Rotate x labels if needed
        if max(len(label) for label in labels) > 6:
            plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        fig.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path)
            logger.info(f"Saved figure to {save_path}")
        
        # Show if requested
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _plot_correlation_matrix_plotly(
        self,
        correlation_matrix: np.ndarray,
        labels: List[str],
        title: str,
        save_path: Optional[str],
        show_fig: bool
    ) -> go.Figure:
        """Plot correlation matrix using Plotly."""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=labels,
            y=labels,
            colorscale='RdBu_r',  # Similar to coolwarm
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[[f"{val:.2f}" for val in row] for row in correlation_matrix],
            hovertemplate='%{x} & %{y}: %{text}<extra></extra>'
        ))
        
        # Set layout
        fig.update_layout(
            title=title,
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        # Save if path provided
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Saved figure to {save_path}")
        
        # Show if requested
        if show_fig:
            fig.show()
        
        return fig
    
    def plot_financial_network(
        self,
        adjacency_matrix: np.ndarray,
        node_attributes: Optional[np.ndarray] = None,
        edge_weights: Optional[np.ndarray] = None,
        node_labels: Optional[List[str]] = None,
        title: str = "Financial Network",
        save_path: Optional[str] = None,
        show_fig: bool = True,
        layout: str = "spring",
        node_size_scale: float = 1.0,
        edge_width_scale: float = 1.0
    ) -> Union[Figure, go.Figure]:
        """
        Plot financial network visualization.
        
        Args:
            adjacency_matrix: Adjacency matrix defining connections
            node_attributes: Optional node attributes for coloring (e.g., sector, size)
            edge_weights: Optional edge weights for line thickness
            node_labels: Labels for the nodes
            title: Plot title
            save_path: Path to save the figure
            show_fig: Whether to display the figure
            layout: Network layout algorithm ('spring', 'circular', 'spectral', etc.)
            node_size_scale: Scaling factor for node sizes
            edge_width_scale: Scaling factor for edge widths
            
        Returns:
            Matplotlib Figure or Plotly Figure
        """
        # Create a graph from the adjacency matrix
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Add node labels if provided
        if node_labels:
            mapping = {i: label for i, label in enumerate(node_labels)}
            G = nx.relabel_nodes(G, mapping)
        
        # Add node attributes if provided
        if node_attributes is not None:
            for i, attrs in enumerate(node_attributes):
                node_id = i if node_labels is None else node_labels[i]
                if isinstance(attrs, np.ndarray) or isinstance(attrs, list):
                    for j, attr in enumerate(attrs):
                        G.nodes[node_id][f"attr_{j}"] = attr
                else:
                    G.nodes[node_id]["attr"] = attrs
        
        # Add edge weights if provided
        if edge_weights is not None:
            for i, j in G.edges():
                i_idx = i if node_labels is None else node_labels.index(i)
                j_idx = j if node_labels is None else node_labels.index(j)
                G[i][j]["weight"] = edge_weights[i_idx, j_idx]
        
        if self.interactive:
            return self._plot_financial_network_plotly(
                G, node_attributes, edge_weights, title, save_path, show_fig, layout,
                node_size_scale, edge_width_scale
            )
        else:
            return self._plot_financial_network_mpl(
                G, node_attributes, edge_weights, title, save_path, show_fig, layout,
                node_size_scale, edge_width_scale
            )
    
    def _plot_financial_network_mpl(
        self,
        G: nx.Graph,
        node_attributes: Optional[np.ndarray],
        edge_weights: Optional[np.ndarray],
        title: str,
        save_path: Optional[str],
        show_fig: bool,
        layout: str,
        node_size_scale: float,
        edge_width_scale: float
    ) -> Figure:
        """Plot financial network using Matplotlib."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Get layout positions
        if layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G)  # Default to spring layout
        
        # Draw nodes
        if node_attributes is not None:
            # Use first attribute for coloring
            node_colors = []
            for node in G.nodes():
                idx = node if isinstance(node, int) else list(G.nodes()).index(node)
                node_colors.append(node_attributes[idx][0] if isinstance(node_attributes[idx], (list, np.ndarray)) 
                                  else node_attributes[idx])
            
            # Use second attribute for sizing if available
            node_sizes = None
            if node_attributes.shape[1] > 1 if node_attributes.ndim > 1 else False:
                node_sizes = []
                for node in G.nodes():
                    idx = node if isinstance(node, int) else list(G.nodes()).index(node)
                    size = node_attributes[idx][1] if isinstance(node_attributes[idx], (list, np.ndarray)) else 300
                    node_sizes.append(size * node_size_scale)
            else:
                node_sizes = 300 * node_size_scale
            
            nx.draw_networkx_nodes(
                G, pos,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.8,
                cmap=plt.cm.get_cmap(self.colormap),
                ax=ax
            )
        else:
            nx.draw_networkx_nodes(
                G, pos,
                node_size=300 * node_size_scale,
                alpha=0.8,
                ax=ax
            )
        
        # Draw edges
        if edge_weights is not None:
            edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                width=[w * edge_width_scale for w in weights],
                alpha=0.5,
                edge_color='gray',
                ax=ax
            )
        else:
            nx.draw_networkx_edges(
                G, pos,
                width=1.0 * edge_width_scale,
                alpha=0.5,
                edge_color='gray',
                ax=ax
            )
        
        # Draw labels if we have a reasonable number of nodes
        if G.number_of_nodes() <= 50:
            nx.draw_networkx_labels(
                G, pos,
                font_size=10,
                font_family='sans-serif',
                ax=ax
            )
        
        # Set title and remove axis
        ax.set_title(title)
        ax.axis('off')
        
        # Adjust layout
        fig.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path)
            logger.info(f"Saved figure to {save_path}")
        
        # Show if requested
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _plot_financial_network_plotly(
        self,
        G: nx.Graph,
        node_attributes: Optional[np.ndarray],
        edge_weights: Optional[np.ndarray],
        title: str,
        save_path: Optional[str],
        show_fig: bool,
        layout: str,
        node_size_scale: float,
        edge_width_scale: float
    ) -> go.Figure:
        """Plot financial network using Plotly."""
        # Get layout positions
        if layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G)  # Default to spring layout
        
        # Extract node positions
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        
        # Set up node colors and sizes
        if node_attributes is not None:
            # Use first attribute for coloring
            node_colors = []
            for node in G.nodes():
                idx = node if isinstance(node, int) else list(G.nodes()).index(node)
                node_colors.append(node_attributes[idx][0] if isinstance(node_attributes[idx], (list, np.ndarray)) 
                                  else node_attributes[idx])
            
            # Use second attribute for sizing if available
            node_sizes = None
            if node_attributes.shape[1] > 1 if node_attributes.ndim > 1 else False:
                node_sizes = []
                for node in G.nodes():
                    idx = node if isinstance(node, int) else list(G.nodes()).index(node)
                    size = node_attributes[idx][1] if isinstance(node_attributes[idx], (list, np.ndarray)) else 20
                    node_sizes.append(size * node_size_scale)
            else:
                node_sizes = 20 * node_size_scale
        else:
            node_colors = [0.5] * len(G.nodes())
            node_sizes = 20 * node_size_scale
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                showscale=True,
                colorscale=self.colormap,
                color=node_colors,
                size=node_sizes,
                colorbar=dict(
                    thickness=15,
                    title='Node Attribute',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)
            ),
            text=[str(node) for node in G.nodes()],
            hoverinfo='text'
        )
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Get edge weight if available
            weight = 2.0
            if 'weight' in G[edge[0]][edge[1]]:
                weight = G[edge[0]][edge[1]]['weight']
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(
                    width=weight * edge_width_scale,
                    color='rgba(150, 150, 150, 0.5)'
                ),
                mode='lines',
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace],
                     layout=go.Layout(
                         title=title,
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20, l=5, r=5, t=40),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                     ))
        
        # Save if path provided
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Saved figure to {save_path}")
        
        # Show if requested
        if show_fig:
            fig.show()
        
        return fig
    
    def plot_feature_importance(
        self,
        features: List[str],
        importance: np.ndarray,
        title: str = "Feature Importance",
        xlabel: str = "Importance",
        ylabel: str = "Feature",
        save_path: Optional[str] = None,
        show_fig: bool = True,
        top_n: Optional[int] = None,
        sort: bool = True
    ) -> Union[Figure, go.Figure]:
        """
        Plot feature importance.
        
        Args:
            features: List of feature names
            importance: Importance scores for features
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save the figure
            show_fig: Whether to display the figure
            top_n: Optional limit to show only top N features
            sort: Whether to sort features by importance
            
        Returns:
            Matplotlib Figure or Plotly Figure
        """
        # Ensure importance is a numpy array
        importance = np.asarray(importance)
        
        # Sort features by importance if requested
        if sort:
            sorted_idx = np.argsort(importance)
            if top_n is not None:
                sorted_idx = sorted_idx[-top_n:]
            features = [features[i] for i in sorted_idx]
            importance = importance[sorted_idx]
        elif top_n is not None:
            # Just take top N without sorting
            top_idx = np.argpartition(importance, -top_n)[-top_n:]
            features = [features[i] for i in top_idx]
            importance = importance[top_idx]
        
        if self.interactive:
            return self._plot_feature_importance_plotly(
                features, importance, title, xlabel, ylabel, save_path, show_fig
            )
        else:
            return self._plot_feature_importance_mpl(
                features, importance, title, xlabel, ylabel, save_path, show_fig
            )
    
    def _plot_feature_importance_mpl(
        self,
        features: List[str],
        importance: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        save_path: Optional[str],
        show_fig: bool
    ) -> Figure:
        """Plot feature importance using Matplotlib."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importance, align='center', color=plt.cm.get_cmap(self.colormap)(0.5))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Add values to bars
        for i, v in enumerate(importance):
            ax.text(v, i, f"{v:.3f}", va='center')
        
        # Adjust layout
        fig.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path)
            logger.info(f"Saved figure to {save_path}")
        
        # Show if requested
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _plot_feature_importance_plotly(
        self,
        features: List[str],
        importance: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        save_path: Optional[str],
        show_fig: bool
    ) -> go.Figure:
        """Plot feature importance using Plotly."""
        # Create horizontal bar chart
        import plotly.express as px
        color = px.colors.sample_colorscale(self.colormap, [0.5])[0]
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color=color,
            text=[f"{v:.3f}" for v in importance],
            textposition='auto'
        ))
        
        # Set layout
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel
        )
        
        # Save if path provided
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Saved figure to {save_path}")
        
        # Show if requested
        if show_fig:
            fig.show()
        
        return fig
    
    def plot_portfolio_performance(
        self,
        dates: Union[List[str], pd.DatetimeIndex],
        portfolio_values: np.ndarray,
        benchmark_values: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        asset_names: Optional[List[str]] = None,
        title: str = "Portfolio Performance",
        save_path: Optional[str] = None,
        show_fig: bool = True,
        plot_drawdowns: bool = True
    ) -> Union[Dict[str, Figure], Dict[str, go.Figure]]:
        """
        Plot comprehensive portfolio performance visualization.
        
        Args:
            dates: Dates for x-axis
            portfolio_values: Portfolio values over time
            benchmark_values: Optional benchmark values for comparison
            weights: Optional portfolio weights over time
            asset_names: Names of assets in the portfolio
            title: Plot title
            save_path: Path to save the figure
            show_fig: Whether to display the figure
            plot_drawdowns: Whether to include drawdown plot
            
        Returns:
            Dictionary of Matplotlib Figures or Plotly Figures
        """
        # Convert dates to proper format if they're strings
        if isinstance(dates, list) and isinstance(dates[0], str):
            dates = pd.to_datetime(dates)
        
        # Convert to numpy arrays
        portfolio_values = np.asarray(portfolio_values)
        if benchmark_values is not None:
            benchmark_values = np.asarray(benchmark_values)
        
        # Calculate returns
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        if benchmark_values is not None:
            benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
        else:
            benchmark_returns = None
        
        # Calculate drawdowns
        if plot_drawdowns:
            portfolio_drawdowns = self._calculate_drawdowns(portfolio_values)
            if benchmark_values is not None:
                benchmark_drawdowns = self._calculate_drawdowns(benchmark_values)
            else:
                benchmark_drawdowns = None
        
        # Create figures dictionary
        figures = {}
        
        # Plot cumulative performance
        cum_perf_title = f"{title} - Cumulative Performance"
        cum_perf_save_path = None
        if save_path:
            base, ext = os.path.splitext(save_path)
            cum_perf_save_path = f"{base}_cumulative{ext}"
        
        labels = ["Portfolio"]
        if benchmark_values is not None:
            labels.append("Benchmark")
        
        values = portfolio_values.reshape(-1, 1)
        if benchmark_values is not None:
            values = np.hstack([values, benchmark_values.reshape(-1, 1)])
        
        figures["cumulative"] = self.plot_time_series_with_uncertainty(
            dates=dates,
            values=values,
            uncertainty=np.zeros_like(values),  # No uncertainty
            labels=labels,
            title=cum_perf_title,
            xlabel="Date",
            ylabel="Value",
            save_path=cum_perf_save_path,
            show_fig=show_fig
        )
        
        # Plot returns distribution
        returns_title = f"{title} - Returns Distribution"
        returns_save_path = None
        if save_path:
            base, ext = os.path.splitext(save_path)
            returns_save_path = f"{base}_returns{ext}"
        
        figures["returns"] = self.plot_return_distribution(
            returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            labels=labels,
            title=returns_title,
            save_path=returns_save_path,
            show_fig=show_fig
        )
        
        # Plot drawdowns if requested
        if plot_drawdowns:
            drawdown_title = f"{title} - Drawdowns"
            drawdown_save_path = None
            if save_path:
                base, ext = os.path.splitext(save_path)
                drawdown_save_path = f"{base}_drawdowns{ext}"
            
            drawdown_values = portfolio_drawdowns.reshape(-1, 1)
            if benchmark_drawdowns is not None:
                drawdown_values = np.hstack([drawdown_values, benchmark_drawdowns.reshape(-1, 1)])
            
            figures["drawdowns"] = self.plot_time_series_with_uncertainty(
                dates=dates,
                values=drawdown_values,
                uncertainty=np.zeros_like(drawdown_values),  # No uncertainty
                labels=labels,
                title=drawdown_title,
                xlabel="Date",
                ylabel="Drawdown",
                save_path=drawdown_save_path,
                show_fig=show_fig
            )
        
        # Plot weights if provided
        if weights is not None and asset_names is not None:
            weights_title = f"{title} - Portfolio Weights"
            weights_save_path = None
            if save_path:
                base, ext = os.path.splitext(save_path)
                weights_save_path = f"{base}_weights{ext}"
            
            if self.interactive:
                figures["weights"] = self._plot_portfolio_weights_plotly(
                    dates, weights, asset_names, weights_title, weights_save_path, show_fig
                )
            else:
                figures["weights"] = self._plot_portfolio_weights_mpl(
                    dates, weights, asset_names, weights_title, weights_save_path, show_fig
                )
        
        return figures
    
    def _calculate_drawdowns(self, values: np.ndarray) -> np.ndarray:
        """Calculate drawdowns from a series of values."""
        # Calculate running maximum
        running_max = np.maximum.accumulate(values)
        
        # Calculate drawdowns
        drawdowns = (values - running_max) / running_max
        
        return drawdowns
    
    def _plot_portfolio_weights_mpl(
        self,
        dates: pd.DatetimeIndex,
        weights: np.ndarray,
        asset_names: List[str],
        title: str,
        save_path: Optional[str],
        show_fig: bool
    ) -> Figure:
        """Plot portfolio weights using Matplotlib."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create stacked area chart
        ax.stackplot(
            dates,
            weights.T,
            labels=asset_names,
            alpha=0.8,
            colors=plt.cm.get_cmap(self.colormap, weights.shape[1])(np.linspace(0, 1, weights.shape[1]))
        )
        
        # Set labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight")
        ax.set_title(title)
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Add legend
        ax.legend(loc="upper right")
        
        # Rotate date labels for better readability
        fig.autofmt_xdate()
        
        # Adjust layout
        fig.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path)
            logger.info(f"Saved figure to {save_path}")
        
        # Show if requested
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _plot_portfolio_weights_plotly(
        self,
        dates: pd.DatetimeIndex,
        weights: np.ndarray,
        asset_names: List[str],
        title: str,
        save_path: Optional[str],
        show_fig: bool
    ) -> go.Figure:
        """Plot portfolio weights using Plotly."""
        import plotly.express as px
        
        # Convert data to format needed by Plotly
        df = pd.DataFrame(weights, index=dates, columns=asset_names)
        df = df.reset_index()
        df_melted = pd.melt(df, id_vars='index', var_name='Asset', value_name='Weight')
        
        # Create stacked area chart
        fig = px.area(
            df_melted,
            x='index',
            y='Weight',
            color='Asset',
            title=title,
            color_discrete_sequence=px.colors.sample_colorscale(
                self.colormap,
                np.linspace(0, 1, weights.shape[1])
            )
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Weight",
            legend_title="Asset"
        )
        
        # Save if path provided
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Saved figure to {save_path}")
        
        # Show if requested
        if show_fig:
            fig.show()
        
        return fig