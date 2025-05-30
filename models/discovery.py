"""
Pattern discovery module for finding hidden relationships in scientific data.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

class PatternDiscovery:
    """
    Tools for discovering patterns in scientific data.
    """
    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        Initialize the pattern discovery module.
        
        Args:
            device: Device to run computations on
        """
        self.device = device
    
    def dimension_reduction(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        method: str = "pca",
        n_components: int = 2,
        **kwargs
    ) -> np.ndarray:
        """
        Reduce dimensionality of embeddings for visualization and pattern discovery.
        
        Args:
            embeddings: High-dimensional embeddings
            method: Dimension reduction method (pca, tsne, umap, svd, mds)
            n_components: Number of components in reduced space
            **kwargs: Additional arguments for the method
            
        Returns:
            Reduced embeddings
        """
        # Convert to numpy array if tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # Apply dimension reduction
        if method == "pca":
            reducer = PCA(n_components=n_components, **kwargs)
            reduced = reducer.fit_transform(embeddings)
            logger.info(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
            
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, **kwargs)
            reduced = reducer.fit_transform(embeddings)
            
        elif method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components, **kwargs)
                reduced = reducer.fit_transform(embeddings)
            except ImportError:
                logger.warning("UMAP not installed. Falling back to PCA.")
                reducer = PCA(n_components=n_components)
                reduced = reducer.fit_transform(embeddings)
            
        elif method == "svd":
            reducer = TruncatedSVD(n_components=n_components, **kwargs)
            reduced = reducer.fit_transform(embeddings)
            logger.info(f"SVD explained variance ratio: {reducer.explained_variance_ratio_}")
            
        elif method == "mds":
            reducer = MDS(n_components=n_components, **kwargs)
            reduced = reducer.fit_transform(embeddings)
            
        else:
            raise ValueError(f"Unsupported dimension reduction method: {method}")
        
        return reduced
    
    def cluster_data(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        method: str = "kmeans",
        n_clusters: int = 5,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Cluster data to discover groups.
        
        Args:
            embeddings: Data embeddings
            method: Clustering method (kmeans, dbscan, agglomerative)
            n_clusters: Number of clusters for methods that require it
            **kwargs: Additional arguments for the clustering method
            
        Returns:
            Tuple of (cluster_labels, evaluation_score)
        """
        # Convert to numpy array if tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # Apply clustering
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, **kwargs)
            labels = clusterer.fit_predict(embeddings)
            
            # Evaluate clustering
            if n_clusters > 1 and len(np.unique(labels)) > 1:
                score = silhouette_score(embeddings, labels)
                logger.info(f"K-means clustering silhouette score: {score:.4f}")
            else:
                score = None
            
        elif method == "dbscan":
            clusterer = DBSCAN(**kwargs)
            labels = clusterer.fit_predict(embeddings)
            
            # Evaluate clustering
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1 and -1 not in unique_labels:
                score = silhouette_score(embeddings, labels)
                logger.info(f"DBSCAN clustering silhouette score: {score:.4f}")
            else:
                score = None
            
            logger.info(f"DBSCAN found {len(unique_labels)} clusters")
            
        elif method == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering
            
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
            labels = clusterer.fit_predict(embeddings)
            
            # Evaluate clustering
            if n_clusters > 1:
                score = silhouette_score(embeddings, labels)
                logger.info(f"Agglomerative clustering silhouette score: {score:.4f}")
            else:
                score = None
                
        elif method == "gaussian_mixture":
            from sklearn.mixture import GaussianMixture
            
            clusterer = GaussianMixture(n_components=n_clusters, **kwargs)
            labels = clusterer.fit_predict(embeddings)
            
            # Evaluate clustering
            if n_clusters > 1:
                score = silhouette_score(embeddings, labels)
                logger.info(f"Gaussian mixture silhouette score: {score:.4f}")
            else:
                score = None
            
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        return labels, score
    
    def find_correlations(
        self,
        data: Union[np.ndarray, torch.Tensor, Dict[str, np.ndarray]],
        method: str = "pearson",
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find correlations between features in the data.
        
        Args:
            data: Data array or dictionary of arrays
            method: Correlation method (pearson, spearman, kendall)
            threshold: Threshold for significant correlations
            
        Returns:
            Tuple of (correlation_matrix, significant_pairs)
        """
        # Handle different input types
        if isinstance(data, dict):
            # Convert dictionary to array
            arrays = []
            feature_names = []
            
            for name, array in data.items():
                if isinstance(array, torch.Tensor):
                    array = array.detach().cpu().numpy()
                
                if array.ndim == 1:
                    arrays.append(array.reshape(-1, 1))
                    feature_names.append(name)
                else:
                    arrays.append(array)
                    feature_names.extend([f"{name}_{i}" for i in range(array.shape[1])])
            
            data = np.hstack(arrays)
        elif isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Calculate correlations
        if method == "pearson":
            from scipy.stats import pearsonr
            
            n_features = data.shape[1]
            corr_matrix = np.zeros((n_features, n_features))
            
            for i in range(n_features):
                for j in range(i, n_features):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        corr, _ = pearsonr(data[:, i], data[:, j])
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
        
        elif method == "spearman":
            from scipy.stats import spearmanr
            
            corr_matrix, _ = spearmanr(data)
            if np.isscalar(corr_matrix):
                corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])
                
        elif method == "kendall":
            from scipy.stats import kendalltau
            
            n_features = data.shape[1]
            corr_matrix = np.zeros((n_features, n_features))
            
            for i in range(n_features):
                for j in range(i, n_features):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        corr, _ = kendalltau(data[:, i], data[:, j])
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
                        
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        
        # Find significant correlations
        significant_indices = np.where(np.abs(corr_matrix) > threshold)
        significant_pairs = np.column_stack(significant_indices)
        
        # Remove self-correlations and duplicates
        significant_pairs = significant_pairs[significant_pairs[:, 0] != significant_pairs[:, 1]]
        
        # Remove duplicates (i,j) and (j,i)
        unique_pairs = set()
        filtered_pairs = []
        
        for i, j in significant_pairs:
            pair = tuple(sorted([i, j]))
            if pair not in unique_pairs:
                unique_pairs.add(pair)
                filtered_pairs.append([i, j])
        
        significant_pairs = np.array(filtered_pairs) if filtered_pairs else np.empty((0, 2), dtype=int)
        
        logger.info(f"Found {len(significant_pairs)} significant correlations")
        
        return corr_matrix, significant_pairs
    
    def find_anomalies(
        self,
        data: Union[np.ndarray, torch.Tensor],
        method: str = "isolation_forest",
        contamination: float = 0.05,
        **kwargs
    ) -> np.ndarray:
        """
        Find anomalies in the data.
        
        Args:
            data: Data array
            method: Anomaly detection method
            contamination: Expected fraction of anomalies
            **kwargs: Additional arguments for the method
            
        Returns:
            Boolean array indicating anomalies
        """
        # Convert to numpy array if tensor
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Apply anomaly detection
        if method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            
            detector = IsolationForest(contamination=contamination, **kwargs)
            predictions = detector.fit_predict(data)
            
            # Convert predictions to anomaly indicators (True for anomalies)
            anomalies = predictions == -1
            
        elif method == "local_outlier_factor":
            from sklearn.neighbors import LocalOutlierFactor
            
            detector = LocalOutlierFactor(contamination=contamination, **kwargs)
            predictions = detector.fit_predict(data)
            
            # Convert predictions to anomaly indicators (True for anomalies)
            anomalies = predictions == -1
            
        elif method == "one_class_svm":
            from sklearn.svm import OneClassSVM
            
            detector = OneClassSVM(**kwargs)
            predictions = detector.fit_predict(data)
            
            # Convert predictions to anomaly indicators (True for anomalies)
            anomalies = predictions == -1
            
        else:
            raise ValueError(f"Unsupported anomaly detection method: {method}")
        
        logger.info(f"Found {anomalies.sum()} anomalies out of {len(data)} samples "
                   f"({100 * anomalies.sum() / len(data):.2f}%)")
        
        return anomalies
    
    def find_feature_importance(
        self,
        data: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        method: str = "random_forest",
        **kwargs
    ) -> np.ndarray:
        """
        Find important features for predicting labels.
        
        Args:
            data: Feature data
            labels: Target labels
            method: Feature importance method
            **kwargs: Additional arguments for the method
            
        Returns:
            Array of feature importances
        """
        # Convert to numpy array if tensor
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
            
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Apply feature importance method
        if method == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            
            model = RandomForestClassifier(**kwargs)
            model.fit(data, labels)
            
            importances = model.feature_importances_
            
        elif method == "permutation":
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.inspection import permutation_importance
            
            model = RandomForestClassifier(**kwargs)
            model.fit(data, labels)
            
            result = permutation_importance(model, data, labels, n_repeats=10, random_state=42)
            importances = result.importances_mean
            
        elif method == "shap":
            try:
                import shap
                from sklearn.ensemble import RandomForestClassifier
                
                model = RandomForestClassifier(**kwargs)
                model.fit(data, labels)
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(data)
                
                if isinstance(shap_values, list):
                    # For multi-class problems, average the SHAP values
                    importances = np.abs(np.array(shap_values)).mean(axis=(0, 1))
                else:
                    importances = np.abs(shap_values).mean(axis=0)
                    
            except ImportError:
                logger.warning("SHAP not installed. Falling back to random forest importance.")
                model = RandomForestClassifier(**kwargs)
                model.fit(data, labels)
                importances = model.feature_importances_
                
        else:
            raise ValueError(f"Unsupported feature importance method: {method}")
        
        logger.info(f"Calculated feature importances using {method}")
        
        return importances