"""
Analysis page component for the multi-modal AI web application.

This module implements the analysis page UI that provides advanced
pattern discovery and analysis for both finance and climate domains.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import base64

def show_analysis_page():
    """Display the analysis page with advanced pattern discovery and visualization."""
    # Main title
    st.title("Advanced Pattern Discovery and Analysis")
    
    # Create tabs for different analysis sections
    tabs = st.tabs(["Dimensionality Reduction", "Clustering", "Correlation Analysis", "Anomaly Detection"])
    
    # Get current domain
    domain = st.session_state.get("domain", "finance")
    
    # Get data based on domain
    if domain == "finance":
        data = st.session_state.get("finance_data", None)
    else:  # climate
        data = st.session_state.get("climate_data", None)
    
    # Check if we have data
    if data is None:
        st.info("ℹ️ No data available. Please load data and run the model first.")
        return
    
    # Process data based on domain
    processed_data = _process_data_for_analysis(data, domain)
    
    # Dimensionality Reduction Tab
    with tabs[0]:
        _show_dimensionality_reduction_tab(processed_data, domain)
    
    # Clustering Tab
    with tabs[1]:
        _show_clustering_tab(processed_data, domain)
    
    # Correlation Analysis Tab
    with tabs[2]:
        _show_correlation_tab(processed_data, domain)
    
    # Anomaly Detection Tab
    with tabs[3]:
        _show_anomaly_detection_tab(processed_data, domain)

def _process_data_for_analysis(data, domain):
    """Process data for analysis based on domain."""
    if domain == "finance":
        # For finance, we have a DataFrame
        if isinstance(data, pd.DataFrame):
            # Get returns
            returns = data.pct_change().dropna()
            
            # Create features dataframe
            features = returns.copy()
            
            # Add rolling statistics
            for window in [5, 20]:
                features[f'volatility_{window}d'] = returns.rolling(window=window).std()
            
            # Fill NaN values
            features = features.fillna(0)
            
            # Create processed data dict
            processed_data = {
                "raw_data": data,
                "returns": returns,
                "features": features,
                "dates": features.index,
                "feature_names": features.columns
            }
            
            return processed_data
        else:
            # No proper finance data
            st.error("❌ Invalid finance data format. Please load data first.")
            return None
    else:  # climate domain
        # For climate, we have a dictionary with temperature and co2 data
        if isinstance(data, dict) and "temperature" in data:
            temp_df = data["temperature"]
            
            # Get features from temperature data
            features = temp_df.copy()
            
            # Add rolling statistics
            for window in [3, 12]:
                for col in temp_df.columns:
                    features[f'{col}_roll_{window}'] = temp_df[col].rolling(window=window).mean()
                    features[f'{col}_std_{window}'] = temp_df[col].rolling(window=window).std()
            
            # Add CO2 data if available
            if "co2" in data and not data["co2"].empty:
                co2_df = data["co2"]
                
                # Add CO2 columns to features
                for col in co2_df.columns:
                    # Align indexes
                    co2_series = co2_df[col].reindex(features.index, method="ffill")
                    features[f'CO2_{col}'] = co2_series
            
            # Fill NaN values
            features = features.fillna(0)
            
            # Create processed data dict
            processed_data = {
                "raw_data": data,
                "temperature": temp_df,
                "features": features,
                "dates": features.index,
                "feature_names": features.columns
            }
            
            return processed_data
        else:
            # No proper climate data
            st.error("❌ Invalid climate data format. Please load data first.")
            return None

def _show_dimensionality_reduction_tab(data, domain):
    """Show dimensionality reduction tab content."""
    st.header("Dimensionality Reduction")
    
    if data is None:
        st.info("ℹ️ No data available for analysis.")
        return
    
    # Get features data
    features_df = data["features"]
    
    # Dimensionality reduction options
    reduction_method = st.selectbox(
        "Select Dimensionality Reduction Method",
        ["PCA", "t-SNE"] + (["UMAP"] if UMAP_AVAILABLE else []),
        index=0
    )
    
    # Number of components
    n_components = st.slider(
        "Number of Components",
        min_value=2,
        max_value=3,
        value=2
    )
    
    # Feature selection
    with st.expander("Feature Selection"):
        # Select features to include
        all_features = list(features_df.columns)
        
        # Domain-specific defaults
        if domain == "finance":
            default_features = all_features[:min(5, len(all_features))]  # First 5 or fewer
        else:  # climate
            default_features = [col for col in all_features if not any(x in col for x in ["roll", "std"])]
            default_features = default_features[:min(5, len(default_features))]  # First 5 or fewer
        
        selected_features = st.multiselect(
            "Select Features",
            all_features,
            default=default_features
        )
    
    # Run dimensionality reduction
    if st.button("Run Dimensionality Reduction"):
        # Check if features are selected
        if not selected_features:
            st.warning("⚠️ Please select at least one feature.")
            return
        
        # Get selected features data
        X = features_df[selected_features].values
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply dimensionality reduction
        if reduction_method == "PCA":
            reducer = PCA(n_components=n_components)
            result = reducer.fit_transform(X_scaled)
            
            # Get explained variance
            explained_variance = reducer.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Show explained variance
            st.subheader("Explained Variance")
            
            # Create bar chart
            fig = go.Figure()
            
            # Add individual variance bars
            fig.add_trace(go.Bar(
                x=[f"Component {i+1}" for i in range(n_components)],
                y=explained_variance * 100,
                name="Explained Variance (%)"
            ))
            
            # Add cumulative variance line
            fig.add_trace(go.Scatter(
                x=[f"Component {i+1}" for i in range(n_components)],
                y=cumulative_variance * 100,
                mode="lines+markers",
                name="Cumulative Variance (%)",
                line=dict(color="red")
            ))
            
            # Update layout
            fig.update_layout(
                title="PCA Explained Variance",
                xaxis_title="Principal Component",
                yaxis_title="Variance (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show component loadings
            st.subheader("Component Loadings")
            
            # Get loadings
            loadings = reducer.components_
            
            # Create heatmap
            fig = px.imshow(
                loadings,
                labels=dict(x="Feature", y="Component"),
                x=selected_features,
                y=[f"Component {i+1}" for i in range(n_components)],
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            
            # Update layout
            fig.update_layout(
                title="PCA Component Loadings",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif reduction_method == "t-SNE":
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(30, len(X_scaled) - 1),
                random_state=42
            )
            result = reducer.fit_transform(X_scaled)
        
        elif reduction_method == "UMAP" and UMAP_AVAILABLE:
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=42
            )
            result = reducer.fit_transform(X_scaled)
        
        # Visualization
        st.subheader(f"{reduction_method} Visualization")
        
        # Create dataframe for plotting
        if n_components == 2:
            plot_df = pd.DataFrame({
                "Component 1": result[:, 0],
                "Component 2": result[:, 1],
                "Date": data["dates"]
            })
            
            # Create scatter plot
            fig = px.scatter(
                plot_df,
                x="Component 1",
                y="Component 2",
                color=plot_df.index if domain == "finance" else plot_df["Date"].dt.year,
                hover_name=plot_df["Date"].dt.strftime("%Y-%m-%d"),
                color_continuous_scale="Viridis"
            )
            
            # Update layout
            fig.update_layout(
                title=f"{reduction_method} Projection",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif n_components == 3:
            plot_df = pd.DataFrame({
                "Component 1": result[:, 0],
                "Component 2": result[:, 1],
                "Component 3": result[:, 2],
                "Date": data["dates"]
            })
            
            # Create 3D scatter plot
            fig = px.scatter_3d(
                plot_df,
                x="Component 1",
                y="Component 2",
                z="Component 3",
                color=plot_df.index if domain == "finance" else plot_df["Date"].dt.year,
                hover_name=plot_df["Date"].dt.strftime("%Y-%m-%d"),
                color_continuous_scale="Viridis"
            )
            
            # Update layout
            fig.update_layout(
                title=f"{reduction_method} Projection (3D)",
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Store results in session state
        st.session_state.reduction_result = result
        st.session_state.reduction_method = reduction_method
        st.session_state.reduction_dates = data["dates"]
        
        # Download button for reduced data
        if n_components == 2:
            download_df = pd.DataFrame({
                "Date": data["dates"],
                "Component_1": result[:, 0],
                "Component_2": result[:, 1]
            })
        else:  # 3 components
            download_df = pd.DataFrame({
                "Date": data["dates"],
                "Component_1": result[:, 0],
                "Component_2": result[:, 1],
                "Component_3": result[:, 2]
            })
        
        csv = download_df.to_csv(index=False)
        st.download_button(
            label=f"Download {reduction_method} Results",
            data=csv,
            file_name=f"{domain}_{reduction_method.lower()}_results.csv",
            mime="text/csv"
        )

def _show_clustering_tab(data, domain):
    """Show clustering tab content."""
    st.header("Clustering Analysis")
    
    if data is None:
        st.info("ℹ️ No data available for analysis.")
        return
    
    # Check if we have dimensionality reduction results
    if "reduction_result" not in st.session_state:
        st.info("ℹ️ Please run dimensionality reduction first to use those results for clustering.")
        
        # Get features data for direct clustering
        features_df = data["features"]
        
        # Feature selection
        with st.expander("Feature Selection"):
            # Select features to include
            all_features = list(features_df.columns)
            
            # Domain-specific defaults
            if domain == "finance":
                default_features = all_features[:min(5, len(all_features))]  # First 5 or fewer
            else:  # climate
                default_features = [col for col in all_features if not any(x in col for x in ["roll", "std"])]
                default_features = default_features[:min(5, len(default_features))]  # First 5 or fewer
            
            selected_features = st.multiselect(
                "Select Features for Clustering",
                all_features,
                default=default_features,
                key="clustering_features"
            )
        
        # Check if features are selected
        if not selected_features:
            st.warning("⚠️ Please select at least one feature or run dimensionality reduction first.")
            return
        
        # Get selected features data
        X = features_df[selected_features].values
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        # Use dimensionality reduction results
        X_scaled = st.session_state.reduction_result
        
        # Show info
        st.info(f"ℹ️ Using {st.session_state.reduction_method} results for clustering.")
    
    # Clustering method selection
    clustering_method = st.selectbox(
        "Select Clustering Method",
        ["K-Means", "DBSCAN"],
        index=0
    )
    
    # Clustering parameters
    if clustering_method == "K-Means":
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=10,
            value=3
        )
    else:  # DBSCAN
        col1, col2 = st.columns(2)
        
        with col1:
            eps = st.slider(
                "DBSCAN Epsilon",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1
            )
        
        with col2:
            min_samples = st.slider(
                "DBSCAN Min Samples",
                min_value=2,
                max_value=20,
                value=5
            )
    
    # Run clustering
    if st.button("Run Clustering"):
        # Apply clustering
        if clustering_method == "K-Means":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(X_scaled)
            
            # Get cluster centers
            cluster_centers = clusterer.cluster_centers_
            
            # Count samples in each cluster
            cluster_counts = np.bincount(cluster_labels)
            
            # Show cluster sizes
            st.subheader("Cluster Sizes")
            
            # Create bar chart
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(go.Bar(
                x=[f"Cluster {i+1}" for i in range(n_clusters)],
                y=cluster_counts,
                text=cluster_counts,
                textposition="auto"
            ))
            
            # Update layout
            fig.update_layout(
                title="Number of Samples in Each Cluster",
                xaxis_title="Cluster",
                yaxis_title="Count",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # DBSCAN
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clusterer.fit_predict(X_scaled)
            
            # Count samples in each cluster
            unique_labels = np.unique(cluster_labels)
            cluster_counts = np.bincount(cluster_labels[cluster_labels >= 0])
            
            # Show cluster sizes
            st.subheader("Cluster Sizes")
            
            # Create dataframe for plotting
            cluster_df = pd.DataFrame({
                "Cluster": [f"Cluster {i+1}" for i in range(len(cluster_counts))] + ["Noise"],
                "Count": list(cluster_counts) + [np.sum(cluster_labels == -1)]
            })
            
            # Create bar chart
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(go.Bar(
                x=cluster_df["Cluster"],
                y=cluster_df["Count"],
                text=cluster_df["Count"],
                textposition="auto",
                marker_color=["blue" if c != "Noise" else "red" for c in cluster_df["Cluster"]]
            ))
            
            # Update layout
            fig.update_layout(
                title="Number of Samples in Each Cluster",
                xaxis_title="Cluster",
                yaxis_title="Count",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Visualization
        st.subheader("Cluster Visualization")
        
        # Check dimensionality for visualization
        if X_scaled.shape[1] >= 2:
            # Create dataframe for plotting
            if X_scaled.shape[1] == 2:
                plot_df = pd.DataFrame({
                    "Component 1": X_scaled[:, 0],
                    "Component 2": X_scaled[:, 1],
                    "Cluster": cluster_labels,
                    "Date": data["dates"]
                })
                
                # Create scatter plot
                fig = px.scatter(
                    plot_df,
                    x="Component 1",
                    y="Component 2",
                    color="Cluster",
                    hover_name=plot_df["Date"].dt.strftime("%Y-%m-%d"),
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                
                # Update layout
                fig.update_layout(
                    title="Cluster Visualization (2D)",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif X_scaled.shape[1] >= 3:
                plot_df = pd.DataFrame({
                    "Component 1": X_scaled[:, 0],
                    "Component 2": X_scaled[:, 1],
                    "Component 3": X_scaled[:, 2],
                    "Cluster": cluster_labels,
                    "Date": data["dates"]
                })
                
                # Create 3D scatter plot
                fig = px.scatter_3d(
                    plot_df,
                    x="Component 1",
                    y="Component 2",
                    z="Component 3",
                    color="Cluster",
                    hover_name=plot_df["Date"].dt.strftime("%Y-%m-%d"),
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                
                # Update layout
                fig.update_layout(
                    title="Cluster Visualization (3D)",
                    height=700
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Temporal distribution of clusters
        st.subheader("Temporal Distribution of Clusters")
        
        # Create dataframe with dates and clusters
        temp_df = pd.DataFrame({
            "Date": data["dates"],
            "Cluster": cluster_labels
        })
        
        # Group by year
        temp_df["Year"] = temp_df["Date"].dt.year
        
        # Count clusters by year
        cluster_by_year = temp_df.groupby(["Year", "Cluster"]).size().unstack(fill_value=0)
        
        # Create stacked bar chart
        fig = go.Figure()
        
        # Add bars for each cluster
        for cluster in cluster_by_year.columns:
            cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster+1}"
            
            fig.add_trace(go.Bar(
                x=cluster_by_year.index,
                y=cluster_by_year[cluster],
                name=cluster_name
            ))
        
        # Update layout
        fig.update_layout(
            title="Cluster Distribution Over Time",
            xaxis_title="Year",
            yaxis_title="Count",
            barmode='stack',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Domain-specific cluster analysis
        st.subheader("Cluster Characteristics")
        
        # Get original features
        original_features = data["features"]
        
        # Calculate mean values for each cluster
        cluster_means = {}
        
        for cluster in np.unique(cluster_labels):
            if cluster == -1:  # Noise in DBSCAN
                cluster_name = "Noise"
            else:
                cluster_name = f"Cluster {cluster+1}"
            
            # Get samples in this cluster
            cluster_samples = original_features.iloc[cluster_labels == cluster]
            
            # Calculate mean
            cluster_means[cluster_name] = cluster_samples.mean()
        
        # Create dataframe
        means_df = pd.DataFrame(cluster_means)
        
        # Show table
        st.dataframe(means_df.T)
        
        # Create radar chart for top features
        if domain == "finance":
            # For finance, use returns features
            top_features = data["returns"].columns.tolist()[:5]  # Top 5 assets
        else:  # climate
            # For climate, use base temperature features
            top_features = [col for col in data["temperature"].columns if not any(x in col for x in ["roll", "std"])]
            top_features = top_features[:5]  # Top 5 features
        
        # Create radar chart
        fig = go.Figure()
        
        # Add traces for each cluster
        for cluster_name, means in cluster_means.items():
            fig.add_trace(go.Scatterpolar(
                r=[means[feature] for feature in top_features],
                theta=top_features,
                fill='toself',
                name=cluster_name
            ))
        
        # Update layout
        fig.update_layout(
            title="Cluster Profiles (Top Features)",
            polar=dict(
                radialaxis=dict(
                    visible=True
                )
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Store clustering results
        st.session_state.cluster_labels = cluster_labels
        
        # Download button for cluster results
        download_df = pd.DataFrame({
            "Date": data["dates"],
            "Cluster": cluster_labels
        })
        
        csv = download_df.to_csv(index=False)
        st.download_button(
            label=f"Download {clustering_method} Results",
            data=csv,
            file_name=f"{domain}_{clustering_method.lower()}_results.csv",
            mime="text/csv"
        )

def _show_correlation_tab(data, domain):
    """Show correlation analysis tab content."""
    st.header("Correlation Analysis")
    
    if data is None:
        st.info("ℹ️ No data available for analysis.")
        return
    
    # Get features data
    features_df = data["features"]
    
    # Correlation method selection
    correlation_method = st.selectbox(
        "Select Correlation Method",
        ["Pearson", "Spearman"],
        index=0
    )
    
    # Feature selection
    with st.expander("Feature Selection"):
        # Select features to include
        all_features = list(features_df.columns)
        
        # Domain-specific defaults
        if domain == "finance":
            default_features = all_features[:min(10, len(all_features))]  # First 10 or fewer
        else:  # climate
            default_features = [col for col in all_features if not any(x in col for x in ["roll", "std"])]
            default_features = default_features[:min(10, len(default_features))]  # First 10 or fewer
        
        selected_features = st.multiselect(
            "Select Features for Correlation Analysis",
            all_features,
            default=default_features,
            key="correlation_features"
        )
    
    # Correlation threshold
    correlation_threshold = st.slider(
        "Correlation Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )
    
    # Run correlation analysis
    if st.button("Run Correlation Analysis"):
        # Check if features are selected
        if not selected_features:
            st.warning("⚠️ Please select at least one feature.")
            return
        
        # Calculate correlation matrix
        correlation_matrix = features_df[selected_features].corr(method=correlation_method.lower())
        
        # Show correlation matrix
        st.subheader("Correlation Matrix")
        
        # Create heatmap
        fig = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=f"{correlation_method} Correlation Matrix"
        )
        
        # Update layout
        fig.update_layout(
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Find significant correlations
        st.subheader("Significant Correlations")
        
        # Create mask for upper triangle to avoid duplicates
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        
        # Apply mask and threshold
        significant_corr = correlation_matrix.mask(~mask).stack()
        significant_corr = significant_corr[abs(significant_corr) >= correlation_threshold]
        
        # Sort by absolute correlation
        significant_corr = significant_corr.sort_values(ascending=False)
        
        # Create dataframe
        if len(significant_corr) > 0:
            corr_df = pd.DataFrame({
                "Feature 1": [significant_corr.index[i][0] for i in range(len(significant_corr))],
                "Feature 2": [significant_corr.index[i][1] for i in range(len(significant_corr))],
                "Correlation": significant_corr.values
            })
            
            # Show table
            st.dataframe(corr_df)
            
            # Create bar chart
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(go.Bar(
                x=[f"{row['Feature 1']} - {row['Feature 2']}" for _, row in corr_df.iterrows()],
                y=corr_df["Correlation"],
                marker_color=[
                    "blue" if val > 0 else "red" 
                    for val in corr_df["Correlation"]
                ]
            ))
            
            # Update layout
            fig.update_layout(
                title="Significant Correlations",
                xaxis_title="Feature Pair",
                yaxis_title="Correlation",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plots for top correlations
            st.subheader("Top Correlation Scatter Plots")
            
            # Show top correlations
            top_n = min(5, len(significant_corr))
            
            # Create columns
            cols = st.columns(top_n)
            
            for i in range(top_n):
                with cols[i]:
                    # Get feature pair
                    feature1 = corr_df.iloc[i]["Feature 1"]
                    feature2 = corr_df.iloc[i]["Feature 2"]
                    corr_value = corr_df.iloc[i]["Correlation"]
                    
                    # Create scatter plot
                    fig = px.scatter(
                        x=features_df[feature1],
                        y=features_df[feature2],
                        title=f"Correlation: {corr_value:.2f}",
                        labels={"x": feature1, "y": feature2}
                    )
                    
                    # Add trendline
                    fig.add_trace(go.Scatter(
                        x=[features_df[feature1].min(), features_df[feature1].max()],
                        y=[
                            features_df[feature1].min() * corr_value * (features_df[feature2].std() / features_df[feature1].std()) + features_df[feature2].mean() - corr_value * features_df[feature1].mean() * (features_df[feature2].std() / features_df[feature1].std()),
                            features_df[feature1].max() * corr_value * (features_df[feature2].std() / features_df[feature1].std()) + features_df[feature2].mean() - corr_value * features_df[feature1].mean() * (features_df[feature2].std() / features_df[feature1].std())
                        ],
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        name="Trend"
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        height=300,
                        width=300
                    )
                    
                    st.plotly_chart(fig)
        else:
            st.info(f"No correlations above threshold {correlation_threshold} found.")
        
        # Domain-specific correlation analysis
        if domain == "finance":
            # For finance, show asset correlation network
            st.subheader("Asset Correlation Network")
            
            # Filter for just asset returns
            asset_features = [col for col in selected_features if not any(x in col for x in ["volatility", "roll", "std"])]
            
            if len(asset_features) >= 2:
                # Calculate asset correlation matrix
                asset_corr = features_df[asset_features].corr(method=correlation_method.lower())
                
                # Create network graph
                # Create nodes
                nodes = [{"id": asset, "label": asset, "size": 10} for asset in asset_features]
                
                # Create edges
                edges = []
                for i in range(len(asset_features)):
                    for j in range(i+1, len(asset_features)):
                        corr = asset_corr.iloc[i, j]
                        if abs(corr) >= correlation_threshold:
                            edges.append({
                                "source": asset_features[i],
                                "target": asset_features[j],
                                "value": abs(corr),
                                "color": "blue" if corr > 0 else "red"
                            })
                
                # Create network plot (simplified with scatter)
                # For a real network visualization, you would need a specialized library like networkx + plotly
                if len(edges) > 0:
                    st.info("Asset correlation network would be shown here with a specialized visualization library.")
                    
                    # Show table of significant correlations
                    edges_df = pd.DataFrame(edges)
                    st.dataframe(edges_df)
                else:
                    st.info(f"No asset correlations above threshold {correlation_threshold} found.")
            else:
                st.info("Select at least 2 asset features for network analysis.")
        
        elif domain == "climate":
            # For climate, show correlation over time
            st.subheader("Correlation Evolution Over Time")
            
            # Select time-based features
            time_features = [col for col in selected_features if not any(x in col for x in ["roll", "std"])]
            
            if len(time_features) >= 2:
                # Select pair of features
                col1, col2 = st.columns(2)
                
                with col1:
                    feature1 = st.selectbox(
                        "First Feature",
                        time_features,
                        index=0
                    )
                
                with col2:
                    feature2 = st.selectbox(
                        "Second Feature",
                        time_features,
                        index=min(1, len(time_features) - 1)
                    )
                
                # Calculate rolling correlation
                window_size = st.slider(
                    "Rolling Window (years)",
                    min_value=1,
                    max_value=20,
                    value=5
                )
                
                # Convert to months
                window_months = window_size * 12
                
                # Calculate rolling correlation
                rolling_corr = features_df[feature1].rolling(window=window_months).corr(features_df[feature2])
                
                # Create figure
                fig = go.Figure()
                
                # Add rolling correlation
                fig.add_trace(go.Scatter(
                    x=features_df.index,
                    y=rolling_corr,
                    mode="lines",
                    name=f"{feature1} - {feature2} Correlation"
                ))
                
                # Add horizontal line at zero
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="black"
                )
                
                # Update layout
                fig.update_layout(
                    title=f"{window_size}-Year Rolling Correlation",
                    xaxis_title="Year",
                    yaxis_title="Correlation",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least 2 time-based features for correlation evolution analysis.")
        
        # Download button for correlation matrix
        csv = correlation_matrix.to_csv()
        st.download_button(
            label="Download Correlation Matrix",
            data=csv,
            file_name=f"{domain}_{correlation_method.lower()}_correlation.csv",
            mime="text/csv"
        )

def _show_anomaly_detection_tab(data, domain):
    """Show anomaly detection tab content."""
    st.header("Anomaly Detection")
    
    if data is None:
        st.info("ℹ️ No data available for analysis.")
        return
    
    # Get features data
    features_df = data["features"]
    
    # Anomaly detection method selection
    anomaly_method = st.selectbox(
        "Select Anomaly Detection Method",
        ["Isolation Forest", "Statistical (Z-Score)"],
        index=0
    )
    
    # Feature selection
    with st.expander("Feature Selection"):
        # Select features to include
        all_features = list(features_df.columns)
        
        # Domain-specific defaults
        if domain == "finance":
            default_features = all_features[:min(5, len(all_features))]  # First 5 or fewer
        else:  # climate
            default_features = [col for col in all_features if not any(x in col for x in ["roll", "std"])]
            default_features = default_features[:min(5, len(default_features))]  # First 5 or fewer
        
        selected_features = st.multiselect(
            "Select Features for Anomaly Detection",
            all_features,
            default=default_features,
            key="anomaly_features"
        )
    
    # Anomaly threshold
    if anomaly_method == "Isolation Forest":
        contamination = st.slider(
            "Contamination (expected anomaly ratio)",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
            step=0.01
        )
    else:  # Statistical
        z_threshold = st.slider(
            "Z-Score Threshold",
            min_value=2.0,
            max_value=5.0,
            value=3.0,
            step=0.1
        )
    
    # Run anomaly detection
    if st.button("Run Anomaly Detection"):
        # Check if features are selected
        if not selected_features:
            st.warning("⚠️ Please select at least one feature.")
            return
        
        # Get selected features data
        X = features_df[selected_features].values
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply anomaly detection
        if anomaly_method == "Isolation Forest":
            detector = IsolationForest(
                contamination=contamination,
                random_state=42
            )
            # Fit and predict (-1 for anomalies, 1 for normal)
            anomaly_labels = detector.fit_predict(X_scaled)
            # Convert to boolean (True for anomalies)
            anomalies = anomaly_labels == -1
        else:  # Statistical
            # Calculate z-scores for each feature
            z_scores = np.abs(X_scaled)
            # Max z-score across features
            max_z = np.max(z_scores, axis=1)
            # Identify anomalies
            anomalies = max_z > z_threshold
        
        # Count anomalies
        num_anomalies = np.sum(anomalies)
        anomaly_ratio = num_anomalies / len(anomalies) * 100  # Percentage
        
        # Show results
        st.subheader("Anomaly Detection Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Number of Anomalies",
                value=str(num_anomalies)
            )
        
        with col2:
            st.metric(
                label="Anomaly Ratio",
                value=f"{anomaly_ratio:.2f}%"
            )
        
        # Show anomalies on timeline
        st.subheader("Anomalies Timeline")
        
        # Create figure
        fig = go.Figure()
        
        # Determine which features to plot
        plot_features = selected_features[:min(3, len(selected_features))]
        
        # Add lines for selected features
        for feature in plot_features:
            fig.add_trace(go.Scatter(
                x=features_df.index,
                y=features_df[feature],
                mode="lines",
                name=feature
            ))
        
        # Add anomaly markers
        anomaly_dates = features_df.index[anomalies]
        
        for feature in plot_features:
            fig.add_trace(go.Scatter(
                x=anomaly_dates,
                y=features_df.loc[anomaly_dates, feature],
                mode="markers",
                marker=dict(
                    color="red",
                    size=10,
                    symbol="x"
                ),
                name=f"{feature} Anomalies"
            ))
        
        # Update layout
        fig.update_layout(
            title="Feature Timeline with Anomalies",
            xaxis_title="Date",
            yaxis_title="Value",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show anomalies on scatter plot if using dimensionality reduction
        if "reduction_result" in st.session_state and st.session_state.reduction_result.shape[1] >= 2:
            st.subheader("Anomalies in Reduced Space")
            
            # Get reduction result
            result = st.session_state.reduction_result
            
            # Create dataframe for plotting
            if result.shape[1] == 2:
                plot_df = pd.DataFrame({
                    "Component 1": result[:, 0],
                    "Component 2": result[:, 1],
                    "Anomaly": anomalies,
                    "Date": data["dates"]
                })
                
                # Create scatter plot
                fig = px.scatter(
                    plot_df,
                    x="Component 1",
                    y="Component 2",
                    color="Anomaly",
                    hover_name=plot_df["Date"].dt.strftime("%Y-%m-%d"),
                    color_discrete_map={True: "red", False: "blue"},
                    category_orders={"Anomaly": [False, True]},
                    labels={"Anomaly": "Is Anomaly"}
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Anomalies in {st.session_state.reduction_method} Space (2D)",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif result.shape[1] >= 3:
                plot_df = pd.DataFrame({
                    "Component 1": result[:, 0],
                    "Component 2": result[:, 1],
                    "Component 3": result[:, 2],
                    "Anomaly": anomalies,
                    "Date": data["dates"]
                })
                
                # Create 3D scatter plot
                fig = px.scatter_3d(
                    plot_df,
                    x="Component 1",
                    y="Component 2",
                    z="Component 3",
                    color="Anomaly",
                    hover_name=plot_df["Date"].dt.strftime("%Y-%m-%d"),
                    color_discrete_map={True: "red", False: "blue"},
                    category_orders={"Anomaly": [False, True]},
                    labels={"Anomaly": "Is Anomaly"}
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Anomalies in {st.session_state.reduction_method} Space (3D)",
                    height=700
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Domain-specific anomaly analysis
        if domain == "finance":
            # For finance, show anomalies in returns
            st.subheader("Financial Anomalies Analysis")
            
            # Get returns data
            returns = data["returns"]
            
            # Calculate returns during anomalies
            anomaly_returns = returns.loc[anomaly_dates]
            
            # Calculate average returns during anomalies
            avg_anomaly_returns = anomaly_returns.mean()
            
            # Create bar chart
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(go.Bar(
                x=avg_anomaly_returns.index,
                y=avg_anomaly_returns.values * 100,  # Convert to percentage
                marker_color=[
                    "green" if val > 0 else "red" 
                    for val in avg_anomaly_returns.values
                ]
            ))
            
            # Update layout
            fig.update_layout(
                title="Average Returns During Anomalies",
                xaxis_title="Asset",
                yaxis_title="Return (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show anomaly table
            st.subheader("Anomaly Details")
            
            # Create dataframe
            anomaly_df = pd.DataFrame({
                "Date": anomaly_dates
            })
            
            # Add returns for top assets
            top_assets = returns.columns[:min(5, len(returns.columns))]
            
            for asset in top_assets:
                anomaly_df[f"{asset}_Return"] = returns.loc[anomaly_dates, asset] * 100  # Percentage
            
            # Format dates
            anomaly_df["Date"] = anomaly_df["Date"].dt.strftime("%Y-%m-%d")
            
            # Show table
            st.dataframe(anomaly_df)
        
        elif domain == "climate":
            # For climate, show anomalies in temperature
            st.subheader("Climate Anomalies Analysis")
            
            # Get temperature data
            temp_df = data["temperature"]
            
            # Calculate average anomaly magnitude
            if "Global" in temp_df.columns:
                global_temp = temp_df["Global"]
                
                # Create figure
                fig = go.Figure()
                
                # Add global temperature
                fig.add_trace(go.Scatter(
                    x=temp_df.index,
                    y=global_temp,
                    mode="lines",
                    name="Global Temperature"
                ))
                
                # Add anomalies
                fig.add_trace(go.Scatter(
                    x=anomaly_dates,
                    y=global_temp.loc[anomaly_dates],
                    mode="markers",
                    marker=dict(
                        color="red",
                        size=10,
                        symbol="x"
                    ),
                    name="Anomalies"
                ))
                
                # Add horizontal line at zero
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="black"
                )
                
                # Update layout
                fig.update_layout(
                    title="Global Temperature with Anomalies",
                    xaxis_title="Year",
                    yaxis_title="Temperature Anomaly (°C)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Show anomaly table
            st.subheader("Anomaly Details")
            
            # Create dataframe
            anomaly_df = pd.DataFrame({
                "Date": anomaly_dates
            })
            
            # Add temperature data for top variables
            top_vars = temp_df.columns[:min(5, len(temp_df.columns))]
            
            for var in top_vars:
                anomaly_df[var] = temp_df.loc[anomaly_dates, var]
            
            # Format dates
            anomaly_df["Date"] = anomaly_df["Date"].dt.strftime("%Y-%m-%d")
            
            # Show table
            st.dataframe(anomaly_df)
            
            # If CO2 data is available, show CO2 during anomalies
            if "co2" in data["raw_data"] and not data["raw_data"]["co2"].empty:
                co2_df = data["raw_data"]["co2"]
                
                # Get CO2 values during anomalies
                co2_anomalies = co2_df.loc[co2_df.index.isin(anomaly_dates)]
                
                if not co2_anomalies.empty:
                    # Create figure
                    fig = go.Figure()
                    
                    # Add CO2
                    fig.add_trace(go.Scatter(
                        x=co2_df.index,
                        y=co2_df.iloc[:, 0],
                        mode="lines",
                        name="CO2 Concentration"
                    ))
                    
                    # Add anomalies
                    fig.add_trace(go.Scatter(
                        x=co2_anomalies.index,
                        y=co2_anomalies.iloc[:, 0],
                        mode="markers",
                        marker=dict(
                            color="red",
                            size=10,
                            symbol="x"
                        ),
                        name="Anomalies"
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title="CO2 Concentration with Anomalies",
                        xaxis_title="Year",
                        yaxis_title="CO2 (ppm)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Store anomaly results
        st.session_state.anomaly_labels = anomalies
        
        # Download button for anomaly results
        download_df = pd.DataFrame({
            "Date": data["dates"],
            "Is_Anomaly": anomalies
        })
        
        csv = download_df.to_csv(index=False)
        st.download_button(
            label=f"Download {anomaly_method} Results",
            data=csv,
            file_name=f"{domain}_{anomaly_method.lower().replace(' ', '_')}_anomalies.csv",
            mime="text/csv"
        )

# Main function to test the component independently
if __name__ == "__main__":
    show_analysis_page()