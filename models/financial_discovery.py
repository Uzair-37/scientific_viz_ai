"""
Advanced pattern discovery and unsupervised mining module for financial data analysis.

This module implements techniques for discovering hidden patterns, anomalies,
and relationships in financial data without requiring labeled data.
"""
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import TSNE, MDS
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

logger = logging.getLogger(__name__)

class FinancialPatternDiscovery:
    """
    Advanced pattern discovery for financial data.
    
    This class implements various unsupervised learning techniques for discovering
    patterns in financial data, including market regimes, anomalies, correlations,
    and hidden factors.
    """
    
    def __init__(
        self,
        random_state: int = 42,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        Initialize the financial pattern discovery module.
        
        Args:
            random_state: Random seed for reproducibility
            device: Device to run computations on
        """
        self.random_state = random_state
        self.device = device
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Initialize standard preprocessor
        self.scaler = StandardScaler()
        
        logger.info(f"Initialized FinancialPatternDiscovery (device: {device})")
    
    def discover_market_regimes(
        self,
        returns: np.ndarray,
        volatility: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None,
        n_regimes: int = 3,
        method: str = "kmeans",
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Discover market regimes using unsupervised clustering.
        
        Args:
            returns: Array of asset returns
            volatility: Optional array of volatility measures
            volumes: Optional array of trading volumes
            n_regimes: Number of regimes/clusters to identify
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            **kwargs: Additional parameters for the clustering algorithm
            
        Returns:
            Tuple of (regime_labels, results_dict)
        """
        # Prepare features
        features = [returns]
        if volatility is not None:
            features.append(volatility)
        if volumes is not None:
            features.append(volumes)
            
        # Combine features
        X = np.column_stack(features)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply clustering
        if method == "kmeans":
            clusterer = KMeans(
                n_clusters=n_regimes,
                random_state=self.random_state,
                **kwargs
            )
            labels = clusterer.fit_predict(X_scaled)
            
            # Compute regime characteristics
            regime_centers = clusterer.cluster_centers_
            regime_centers = self.scaler.inverse_transform(regime_centers)
            
            # Calculate silhouette score
            silhouette = silhouette_score(X_scaled, labels) if n_regimes > 1 else 0
            
            # Calculate inertia
            inertia = clusterer.inertia_
            
            results = {
                "regime_centers": regime_centers,
                "silhouette_score": silhouette,
                "inertia": inertia,
                "clusterer": clusterer
            }
            
        elif method == "dbscan":
            eps = kwargs.get("eps", 0.5)
            min_samples = kwargs.get("min_samples", 5)
            
            clusterer = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                **kwargs
            )
            labels = clusterer.fit_predict(X_scaled)
            
            # Handle noise points (label -1)
            n_noise = np.sum(labels == -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Calculate regime characteristics
            regime_centers = []
            for i in range(n_clusters):
                mask = labels == i
                if np.any(mask):
                    center = X_scaled[mask].mean(axis=0)
                    regime_centers.append(center)
            
            if regime_centers:
                regime_centers = np.vstack(regime_centers)
                regime_centers = self.scaler.inverse_transform(regime_centers)
            else:
                regime_centers = np.array([])
            
            # Calculate silhouette score if multiple valid clusters
            if n_clusters > 1 and np.sum(labels != -1) > n_clusters:
                # Filter out noise points for silhouette calculation
                valid_mask = labels != -1
                silhouette = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
            else:
                silhouette = 0
            
            results = {
                "regime_centers": regime_centers,
                "silhouette_score": silhouette,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "clusterer": clusterer
            }
            
        elif method == "hierarchical":
            linkage = kwargs.get("linkage", "ward")
            
            clusterer = AgglomerativeClustering(
                n_clusters=n_regimes,
                linkage=linkage,
                **kwargs
            )
            labels = clusterer.fit_predict(X_scaled)
            
            # Calculate regime characteristics
            regime_centers = []
            for i in range(n_regimes):
                mask = labels == i
                center = X_scaled[mask].mean(axis=0)
                regime_centers.append(center)
            
            regime_centers = np.vstack(regime_centers)
            regime_centers = self.scaler.inverse_transform(regime_centers)
            
            # Calculate silhouette score
            silhouette = silhouette_score(X_scaled, labels) if n_regimes > 1 else 0
            
            # Calculate Davies-Bouldin score (lower is better)
            davies_bouldin = davies_bouldin_score(X_scaled, labels)
            
            results = {
                "regime_centers": regime_centers,
                "silhouette_score": silhouette,
                "davies_bouldin_score": davies_bouldin,
                "clusterer": clusterer
            }
            
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        # Add feature importance
        if returns.ndim > 1 and returns.shape[1] > 1:
            # Train a random forest to estimate feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            rf.fit(X, labels)
            results["feature_importance"] = rf.feature_importances_
        
        logger.info(f"Discovered {len(set(labels))} market regimes using {method}")
        
        return labels, results
    
    def detect_anomalies(
        self,
        data: np.ndarray,
        method: str = "isolation_forest",
        contamination: float = 0.05,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect anomalies in financial data.
        
        Args:
            data: Input data array (samples × features)
            method: Anomaly detection method
            contamination: Expected fraction of anomalies
            **kwargs: Additional parameters for the anomaly detection algorithm
            
        Returns:
            Tuple of (anomaly_labels, results_dict)
        """
        # Scale the data
        X_scaled = self.scaler.fit_transform(data)
        
        if method == "isolation_forest":
            detector = IsolationForest(
                contamination=contamination,
                random_state=self.random_state,
                **kwargs
            )
            scores = detector.fit(X_scaled).score_samples(X_scaled)
            
            # Convert scores to anomaly labels (True for anomalies)
            labels = scores < np.percentile(scores, contamination * 100)
            
            results = {
                "anomaly_scores": -scores,  # Convert to anomaly score (higher = more anomalous)
                "threshold": np.percentile(-scores, (1 - contamination) * 100),
                "detector": detector
            }
            
        elif method == "local_outlier_factor":
            from sklearn.neighbors import LocalOutlierFactor
            
            detector = LocalOutlierFactor(
                contamination=contamination,
                novelty=False,
                **kwargs
            )
            detector.fit(X_scaled)
            scores = -detector.negative_outlier_factor_
            
            # Convert scores to anomaly labels (True for anomalies)
            labels = scores > np.percentile(scores, (1 - contamination) * 100)
            
            results = {
                "anomaly_scores": scores,
                "threshold": np.percentile(scores, (1 - contamination) * 100),
                "detector": detector
            }
            
        elif method == "mahalanobis":
            # Calculate mean and covariance
            mean = np.mean(X_scaled, axis=0)
            cov = np.cov(X_scaled, rowvar=False)
            
            # Calculate Mahalanobis distance
            inv_cov = np.linalg.inv(cov)
            scores = np.zeros(X_scaled.shape[0])
            for i in range(X_scaled.shape[0]):
                diff = X_scaled[i] - mean
                scores[i] = np.sqrt(diff.dot(inv_cov).dot(diff.T))
            
            # Convert scores to anomaly labels (True for anomalies)
            labels = scores > np.percentile(scores, (1 - contamination) * 100)
            
            results = {
                "anomaly_scores": scores,
                "threshold": np.percentile(scores, (1 - contamination) * 100),
                "mean": mean,
                "cov": cov
            }
            
        elif method == "autoencoder":
            # Use a simple autoencoder for anomaly detection
            autoencoder, scores = self._train_anomaly_autoencoder(X_scaled, **kwargs)
            
            # Convert scores to anomaly labels (True for anomalies)
            labels = scores > np.percentile(scores, (1 - contamination) * 100)
            
            results = {
                "anomaly_scores": scores,
                "threshold": np.percentile(scores, (1 - contamination) * 100),
                "autoencoder": autoencoder
            }
            
        else:
            raise ValueError(f"Unsupported anomaly detection method: {method}")
        
        logger.info(f"Detected {np.sum(labels)} anomalies using {method} "
                   f"({np.sum(labels) / len(labels):.1%} of data)")
        
        return labels, results
    
    def _train_anomaly_autoencoder(
        self,
        X: np.ndarray,
        hidden_dim: int = 10,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        **kwargs
    ) -> Tuple[nn.Module, np.ndarray]:
        """Train an autoencoder for anomaly detection."""
        input_dim = X.shape[1]
        
        # Define a simple autoencoder
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        # Create and train the autoencoder
        autoencoder = Autoencoder(input_dim, hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Convert data to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train the autoencoder
        autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, _ in dataloader:
                optimizer.zero_grad()
                outputs = autoencoder(batch_X)
                loss = criterion(outputs, batch_X)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.6f}")
        
        # Compute reconstruction error for each sample
        autoencoder.eval()
        with torch.no_grad():
            reconstructions = autoencoder(X_tensor)
            mse = torch.mean((reconstructions - X_tensor) ** 2, dim=1)
            scores = mse.cpu().numpy()
        
        return autoencoder, scores
    
    def discover_latent_factors(
        self,
        returns: np.ndarray,
        n_factors: Optional[int] = None,
        method: str = "pca",
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Discover latent factors driving asset returns.
        
        Args:
            returns: Array of asset returns (time × assets)
            n_factors: Number of factors to extract (if None, determined automatically)
            method: Factor extraction method ('pca', 'ica', 'nmf')
            **kwargs: Additional parameters for the factor extraction algorithm
            
        Returns:
            Tuple of (factors, results_dict)
        """
        # Scale the returns
        X_scaled = self.scaler.fit_transform(returns)
        
        if method == "pca":
            # Determine number of factors if not specified
            if n_factors is None:
                n_factors = self._estimate_n_components(X_scaled, method="pca", **kwargs)
            
            # Apply PCA
            pca = PCA(n_components=n_factors, random_state=self.random_state)
            factors = pca.fit_transform(X_scaled)
            
            # Get loadings (correlation between factors and original variables)
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            
            results = {
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "explained_variance": pca.explained_variance_,
                "loadings": loadings,
                "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
                "model": pca
            }
            
        elif method == "ica":
            # Determine number of components if not specified
            if n_factors is None:
                n_factors = self._estimate_n_components(X_scaled, method="ica", **kwargs)
            
            # Apply ICA
            ica = FastICA(n_components=n_factors, random_state=self.random_state, **kwargs)
            factors = ica.fit_transform(X_scaled)
            
            # Get mixing matrix (how independent components mix to form observed data)
            mixing_matrix = ica.mixing_
            
            results = {
                "mixing_matrix": mixing_matrix,
                "components": ica.components_,
                "model": ica
            }
            
        elif method == "nmf":
            # NMF requires non-negative data
            X_pos = X_scaled - np.min(X_scaled)
            
            # Determine number of components if not specified
            if n_factors is None:
                n_factors = self._estimate_n_components(X_pos, method="nmf", **kwargs)
            
            # Apply NMF
            nmf = NMF(n_components=n_factors, random_state=self.random_state, **kwargs)
            factors = nmf.fit_transform(X_pos)
            
            results = {
                "components": nmf.components_,
                "reconstruction_err": nmf.reconstruction_err_,
                "model": nmf
            }
            
        else:
            raise ValueError(f"Unsupported factor extraction method: {method}")
        
        logger.info(f"Discovered {n_factors} latent factors using {method}")
        
        return factors, results
    
    def _estimate_n_components(
        self,
        X: np.ndarray,
        method: str = "pca",
        explained_variance_threshold: float = 0.9,
        max_components: int = 20,
        **kwargs
    ) -> int:
        """Estimate the optimal number of components/factors."""
        if method == "pca":
            # Use explained variance ratio
            pca = PCA(random_state=self.random_state)
            pca.fit(X)
            
            # Find number of components that explain threshold% of variance
            explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(explained_variance_ratio >= explained_variance_threshold) + 1
            
            # Limit to maximum number of components
            n_components = min(n_components, max_components)
            
        elif method == "ica" or method == "nmf":
            # For ICA and NMF, use reconstruction error
            reconstruction_errors = []
            max_test_components = min(max_components, min(X.shape) - 1)
            
            for n in range(1, max_test_components + 1):
                if method == "ica":
                    model = FastICA(n_components=n, random_state=self.random_state, **kwargs)
                else:  # NMF
                    model = NMF(n_components=n, random_state=self.random_state, **kwargs)
                
                model.fit(X)
                reconstruction = model.transform(X)
                reconstruction = model.inverse_transform(reconstruction)
                error = np.mean((X - reconstruction) ** 2)
                reconstruction_errors.append(error)
                
            # Find elbow point using the kneedle algorithm
            try:
                from kneed import KneeLocator
                k = KneeLocator(
                    range(1, len(reconstruction_errors) + 1),
                    reconstruction_errors,
                    curve="convex",
                    direction="decreasing"
                )
                n_components = k.elbow
                if n_components is None:
                    n_components = max(1, len(reconstruction_errors) // 3)
            except (ImportError, ValueError):
                # Fallback: use explained variance threshold
                pca = PCA(random_state=self.random_state)
                pca.fit(X)
                explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(explained_variance_ratio >= explained_variance_threshold) + 1
                
        # Ensure at least 1 component
        return max(1, n_components)
    
    def visualize_embedding(
        self,
        data: np.ndarray,
        method: str = "tsne",
        n_components: int = 2,
        labels: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create low-dimensional embedding for visualization.
        
        Args:
            data: Input data array (samples × features)
            method: Embedding method ('tsne', 'mds', 'umap')
            n_components: Number of dimensions in the embedding
            labels: Optional labels for coloring points
            **kwargs: Additional parameters for the embedding algorithm
            
        Returns:
            Tuple of (embedding, results_dict)
        """
        # Scale the data
        X_scaled = self.scaler.fit_transform(data)
        
        if method == "tsne":
            # Apply t-SNE
            perplexity = min(kwargs.get("perplexity", 30), X_scaled.shape[0] - 1)
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=self.random_state,
                **kwargs
            )
            embedding = tsne.fit_transform(X_scaled)
            
            results = {
                "kl_divergence": tsne.kl_divergence_,
                "model": tsne
            }
            
        elif method == "mds":
            # Apply MDS
            mds = MDS(
                n_components=n_components,
                random_state=self.random_state,
                **kwargs
            )
            embedding = mds.fit_transform(X_scaled)
            
            results = {
                "stress": mds.stress_,
                "model": mds
            }
            
        elif method == "umap":
            try:
                import umap
                # Apply UMAP
                reducer = umap.UMAP(
                    n_components=n_components,
                    random_state=self.random_state,
                    **kwargs
                )
                embedding = reducer.fit_transform(X_scaled)
                
                results = {
                    "model": reducer
                }
            except ImportError:
                logger.warning("UMAP not installed. Falling back to t-SNE.")
                # Fallback to t-SNE
                perplexity = min(kwargs.get("perplexity", 30), X_scaled.shape[0] - 1)
                tsne = TSNE(
                    n_components=n_components,
                    perplexity=perplexity,
                    random_state=self.random_state
                )
                embedding = tsne.fit_transform(X_scaled)
                
                results = {
                    "kl_divergence": tsne.kl_divergence_,
                    "model": tsne
                }
                
        else:
            raise ValueError(f"Unsupported embedding method: {method}")
        
        # Add label information if provided
        if labels is not None:
            # Calculate separation metrics
            if n_components == 2 and len(np.unique(labels)) > 1:
                silhouette = silhouette_score(embedding, labels)
                davies_bouldin = davies_bouldin_score(embedding, labels)
                
                results["silhouette_score"] = silhouette
                results["davies_bouldin_score"] = davies_bouldin
        
        logger.info(f"Created {n_components}D embedding using {method}")
        
        return embedding, results
    
    def analyze_correlations(
        self,
        data: np.ndarray,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
        threshold: float = 0.7,
        p_value_threshold: float = 0.05
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Analyze correlations between features.
        
        Args:
            data: Input data array (samples × features)
            columns: Optional column names for the features
            method: Correlation method ('pearson', 'spearman', 'kendall')
            threshold: Correlation coefficient magnitude threshold for significance
            p_value_threshold: P-value threshold for statistical significance
            
        Returns:
            Tuple of (correlation_matrix, results_dict)
        """
        # Calculate correlation matrix
        if isinstance(data, pd.DataFrame):
            if method == "pearson":
                corr_matrix = data.corr(method='pearson').values
            elif method == "spearman":
                corr_matrix = data.corr(method='spearman').values
            elif method == "kendall":
                corr_matrix = data.corr(method='kendall').values
            else:
                raise ValueError(f"Unsupported correlation method: {method}")
                
            if columns is None:
                columns = data.columns.tolist()
                
        else:
            if method == "pearson":
                corr_matrix = np.corrcoef(data.T)
            elif method == "spearman":
                from scipy.stats import spearmanr
                corr_matrix, p_matrix = spearmanr(data)
                if corr_matrix.size == 1:  # Handle single correlation value
                    corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])
                    p_matrix = np.array([[0.0, p_matrix], [p_matrix, 0.0]])
            elif method == "kendall":
                from scipy.stats import kendalltau
                n_features = data.shape[1]
                corr_matrix = np.zeros((n_features, n_features))
                p_matrix = np.zeros((n_features, n_features))
                
                for i in range(n_features):
                    for j in range(i, n_features):
                        tau, p_value = kendalltau(data[:, i], data[:, j])
                        corr_matrix[i, j] = tau
                        corr_matrix[j, i] = tau
                        p_matrix[i, j] = p_value
                        p_matrix[j, i] = p_value
            else:
                raise ValueError(f"Unsupported correlation method: {method}")
        
        # Find significant correlations
        if method != "pearson":
            significant_mask = (np.abs(corr_matrix) > threshold) & (p_matrix < p_value_threshold)
        else:
            significant_mask = np.abs(corr_matrix) > threshold
        
        # Zero out the diagonal and lower triangular part for better visualization
        np.fill_diagonal(significant_mask, False)
        significant_mask = np.triu(significant_mask, k=1)
        
        # Extract significant pairs
        significant_pairs = np.where(significant_mask)
        significant_correlations = corr_matrix[significant_pairs]
        
        # Create list of significant correlation details
        corr_details = []
        for idx in range(len(significant_pairs[0])):
            i, j = significant_pairs[0][idx], significant_pairs[1][idx]
            
            if method != "pearson":
                p_value = p_matrix[i, j]
            else:
                # Estimate p-value for Pearson correlation
                from scipy.stats import pearsonr
                _, p_value = pearsonr(data[:, i], data[:, j])
            
            feature_i = columns[i] if columns is not None else f"Feature {i}"
            feature_j = columns[j] if columns is not None else f"Feature {j}"
            
            corr_details.append({
                "feature_1": feature_i,
                "feature_2": feature_j,
                "correlation": corr_matrix[i, j],
                "p_value": p_value
            })
        
        # Sort by correlation magnitude
        corr_details = sorted(corr_details, key=lambda x: abs(x["correlation"]), reverse=True)
        
        results = {
            "significant_pairs": np.column_stack(significant_pairs) if len(significant_pairs[0]) > 0 else np.array([]),
            "significant_correlations": significant_correlations,
            "correlation_details": corr_details,
            "columns": columns
        }
        
        logger.info(f"Found {len(corr_details)} significant correlations using {method} method")
        
        return corr_matrix, results
    
    def detect_seasonality(
        self,
        time_series: np.ndarray,
        periods: List[int] = [5, 21, 63, 252],  # daily, weekly, monthly, yearly
        method: str = "fft",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect seasonality and cyclical patterns in time series data.
        
        Args:
            time_series: Input time series data (can be multi-dimensional)
            periods: List of periods to test for
            method: Method for seasonality detection ('fft', 'autocorr', 'stl')
            **kwargs: Additional parameters for the seasonality detection algorithm
            
        Returns:
            Dictionary with seasonality analysis results
        """
        # Ensure time_series is 2D
        if time_series.ndim == 1:
            time_series = time_series.reshape(-1, 1)
            
        n_series = time_series.shape[1]
        n_periods = len(periods)
        
        # Prepare results container
        seasonality_scores = np.zeros((n_series, n_periods))
        dominant_frequencies = []
        
        if method == "fft":
            # Use Fast Fourier Transform for spectral analysis
            from scipy.fftpack import fft
            
            for i in range(n_series):
                series = time_series[:, i]
                
                # Compute FFT
                n = len(series)
                yf = fft(series)
                power = np.abs(yf[:n//2]) ** 2
                freqs = np.arange(n//2) / n
                periods_freq = 1 / freqs[1:]  # Skip DC component
                
                # Find peaks in power spectrum
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(power[1:], height=np.mean(power[1:]) + 1 * np.std(power[1:]))
                
                # Get top peaks
                if len(peaks) > 0:
                    peak_periods = periods_freq[peaks]
                    peak_powers = power[1:][peaks]
                    
                    # Sort by power
                    sort_idx = np.argsort(peak_powers)[::-1]
                    peak_periods = peak_periods[sort_idx]
                    peak_powers = peak_powers[sort_idx]
                    
                    # Store top periods
                    dominant_frequencies.append({
                        "series": i,
                        "peak_periods": peak_periods[:5].tolist(),
                        "peak_powers": peak_powers[:5].tolist()
                    })
                    
                    # Score each target period
                    for j, period in enumerate(periods):
                        # Find closest peak
                        closest_idx = np.argmin(np.abs(peak_periods - period))
                        closest_period = peak_periods[closest_idx]
                        
                        # Score based on power and proximity
                        proximity = 1 - min(abs(closest_period - period) / period, 1)
                        power_ratio = peak_powers[closest_idx] / np.max(peak_powers)
                        
                        seasonality_scores[i, j] = proximity * power_ratio
                else:
                    dominant_frequencies.append({
                        "series": i,
                        "peak_periods": [],
                        "peak_powers": []
                    })
                    
        elif method == "autocorr":
            # Use autocorrelation
            from statsmodels.tsa.stattools import acf
            
            for i in range(n_series):
                series = time_series[:, i]
                
                # Compute autocorrelation
                max_lag = min(len(series) - 1, 2 * max(periods))
                autocorr = acf(series, nlags=max_lag, fft=True)
                
                # Find peaks in autocorrelation
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(autocorr, height=0)
                
                # Get peak periods and strengths
                if len(peaks) > 0:
                    peak_periods = peaks
                    peak_strengths = autocorr[peaks]
                    
                    # Sort by strength
                    sort_idx = np.argsort(peak_strengths)[::-1]
                    peak_periods = peak_periods[sort_idx]
                    peak_strengths = peak_strengths[sort_idx]
                    
                    # Store top periods
                    dominant_frequencies.append({
                        "series": i,
                        "peak_periods": peak_periods[:5].tolist(),
                        "peak_strengths": peak_strengths[:5].tolist()
                    })
                    
                    # Score each target period
                    for j, period in enumerate(periods):
                        if period <= max_lag:
                            # Use direct autocorrelation value
                            seasonality_scores[i, j] = max(0, autocorr[period])
                else:
                    dominant_frequencies.append({
                        "series": i,
                        "peak_periods": [],
                        "peak_strengths": []
                    })
                    
        elif method == "stl":
            # Use Seasonal-Trend-Decomposition
            try:
                from statsmodels.tsa.seasonal import STL
                
                for i in range(n_series):
                    series = time_series[:, i]
                    series_scores = []
                    
                    for period in periods:
                        if period < 2 or period >= len(series) // 2:
                            series_scores.append(0)
                            continue
                        
                        try:
                            # Decompose series
                            stl = STL(series, period=period)
                            result = stl.fit()
                            
                            # Calculate strength of seasonality
                            var_seasonal = np.var(result.seasonal)
                            var_residual = np.var(result.resid)
                            var_total = np.var(series)
                            
                            # Score formula: 1 - (var_resid / (var_seasonal + var_resid))
                            score = 1 - var_residual / max(var_seasonal + var_residual, 1e-10)
                            series_scores.append(max(0, score))
                        except:
                            series_scores.append(0)
                    
                    seasonality_scores[i, :] = series_scores
            except ImportError:
                logger.warning("statsmodels not installed or STL failed. Falling back to FFT.")
                return self.detect_seasonality(time_series, periods, method="fft", **kwargs)
                
        else:
            raise ValueError(f"Unsupported seasonality detection method: {method}")
        
        # Prepare results
        results = {
            "seasonality_scores": seasonality_scores,
            "periods": periods,
            "dominant_frequencies": dominant_frequencies,
            "method": method
        }
        
        logger.info(f"Detected seasonality patterns using {method} method")
        
        return results
    
    def detect_cointegration(
        self,
        time_series: np.ndarray,
        max_combinations: int = 100,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect cointegration relationships between time series.
        
        Args:
            time_series: Multi-dimensional time series data (samples × series)
            max_combinations: Maximum number of series combinations to test
            significance_level: Significance level for cointegration test
            
        Returns:
            Dictionary with cointegration test results
        """
        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            from statsmodels.tsa.stattools import coint
            import itertools
        except ImportError:
            logger.error("statsmodels not installed. Cannot run cointegration tests.")
            return {"error": "statsmodels not installed"}
        
        n_series = time_series.shape[1]
        
        # Test pairs for cointegration
        pairs_results = []
        
        # Generate all possible pairs
        series_indices = list(range(n_series))
        pairs = list(itertools.combinations(series_indices, 2))
        
        # Limit to max_combinations random pairs if there are too many
        if len(pairs) > max_combinations:
            import random
            random.seed(self.random_state)
            pairs = random.sample(pairs, max_combinations)
        
        # Test each pair
        for i, j in pairs:
            x = time_series[:, i]
            y = time_series[:, j]
            
            # Run Engle-Granger test
            score, p_value, _ = coint(x, y)
            
            # Store if significant
            if p_value < significance_level:
                pairs_results.append({
                    "series_1": i,
                    "series_2": j,
                    "test_statistic": score,
                    "p_value": p_value,
                    "is_cointegrated": True
                })
            else:
                pairs_results.append({
                    "series_1": i,
                    "series_2": j,
                    "test_statistic": score,
                    "p_value": p_value,
                    "is_cointegrated": False
                })
        
        # Also test for multivariate cointegration using Johansen test
        # This works best for small groups of related series
        group_results = []
        
        # Test combinations of 3-5 series
        for size in range(3, min(6, n_series + 1)):
            for subset in itertools.combinations(series_indices, size):
                # Skip if too many combinations
                if len(group_results) >= max_combinations:
                    break
                    
                subset_data = time_series[:, subset]
                
                try:
                    # Run Johansen test
                    result = coint_johansen(subset_data, det_order=0, k_ar_diff=1)
                    
                    # Process results for each possible rank
                    for r in range(size - 1):
                        trace_stat = result.lr1[r]
                        trace_crit = result.cvt[r, 0]  # 90% critical value
                        
                        max_stat = result.lr2[r]
                        max_crit = result.cvm[r, 0]  # 90% critical value
                        
                        # Store if significant
                        if trace_stat > trace_crit:
                            group_results.append({
                                "series_indices": subset,
                                "rank": r + 1,
                                "trace_statistic": trace_stat,
                                "trace_critical_value": trace_crit,
                                "max_statistic": max_stat,
                                "max_critical_value": max_crit,
                                "is_significant": True
                            })
                        else:
                            group_results.append({
                                "series_indices": subset,
                                "rank": r + 1,
                                "trace_statistic": trace_stat,
                                "trace_critical_value": trace_crit,
                                "max_statistic": max_stat,
                                "max_critical_value": max_crit,
                                "is_significant": False
                            })
                except:
                    logger.warning(f"Johansen test failed for subset {subset}")
        
        # Prepare results
        results = {
            "pairwise_results": pairs_results,
            "group_results": group_results,
            "significance_level": significance_level
        }
        
        logger.info(f"Detected {sum(r['is_cointegrated'] for r in pairs_results)} cointegrated pairs out of {len(pairs_results)} tested")
        
        return results