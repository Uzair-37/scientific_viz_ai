"""
Diagnostics page component for the multi-modal AI web application.

This module implements the diagnostics page UI that provides
model evaluation, uncertainty calibration, and performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from io import BytesIO
import base64

def show_diagnostics_page():
    """Display the diagnostics page with model evaluation and uncertainty calibration."""
    # Main title
    st.title("Model Diagnostics and Evaluation")
    
    # Create tabs for different diagnostics sections
    tabs = st.tabs(["Model Performance", "Uncertainty Calibration", "Error Analysis", "Model Explanation"])
    
    # Get current domain
    domain = st.session_state.get("domain", "finance")
    
    # Check if predictions are available
    if domain == "finance":
        predictions = st.session_state.get("finance_predictions", None)
        data = st.session_state.get("finance_data", None)
    else:  # climate
        predictions = st.session_state.get("climate_predictions", None)
        data = st.session_state.get("climate_data", None)
    
    # Check if we have predictions
    if predictions is None:
        st.info("ℹ️ No model predictions available. Please run the model first.")
        return
    
    # Process predictions based on domain
    processed_preds = _process_predictions_for_diagnostics(predictions, data, domain)
    
    # Model Performance Tab
    with tabs[0]:
        _show_model_performance_tab(processed_preds, domain)
    
    # Uncertainty Calibration Tab
    with tabs[1]:
        _show_uncertainty_calibration_tab(processed_preds, domain)
    
    # Error Analysis Tab
    with tabs[2]:
        _show_error_analysis_tab(processed_preds, domain)
    
    # Model Explanation Tab
    with tabs[3]:
        _show_model_explanation_tab(processed_preds, domain)

def _process_predictions_for_diagnostics(predictions, data, domain):
    """Process model predictions for diagnostics based on domain."""
    if domain == "finance":
        # For finance, we expect predictions to contain:
        # - y_true: actual returns
        # - y_pred: predicted returns
        # - y_std: uncertainty estimates
        if not isinstance(predictions, dict) or "y_pred" not in predictions:
            st.error("❌ Invalid prediction format. Please run the model first.")
            return None
        
        # Create processed predictions dictionary
        processed = {
            "domain": "finance",
            "y_true": predictions.get("y_true", None),
            "y_pred": predictions.get("y_pred", None),
            "y_std": predictions.get("y_std", None),
            "dates": predictions.get("dates", None),
            "assets": predictions.get("assets", None),
            "horizons": predictions.get("horizons", None),
        }
        
        # Add metrics if predictions are available
        if processed["y_true"] is not None and processed["y_pred"] is not None:
            # Calculate metrics
            metrics = {}
            
            # Overall metrics
            y_true_flat = processed["y_true"].flatten()
            y_pred_flat = processed["y_pred"].flatten()
            
            metrics["mse"] = mean_squared_error(y_true_flat, y_pred_flat)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_true_flat, y_pred_flat)
            metrics["r2"] = r2_score(y_true_flat, y_pred_flat)
            
            # Per-asset metrics
            if processed["assets"] is not None:
                asset_metrics = {}
                
                for i, asset in enumerate(processed["assets"]):
                    asset_metrics[asset] = {
                        "mse": mean_squared_error(processed["y_true"][:, i], processed["y_pred"][:, i]),
                        "rmse": np.sqrt(mean_squared_error(processed["y_true"][:, i], processed["y_pred"][:, i])),
                        "mae": mean_absolute_error(processed["y_true"][:, i], processed["y_pred"][:, i]),
                        "r2": r2_score(processed["y_true"][:, i], processed["y_pred"][:, i])
                    }
                
                metrics["per_asset"] = asset_metrics
            
            # Per-horizon metrics
            if processed["horizons"] is not None:
                horizon_metrics = {}
                
                for i, horizon in enumerate(processed["horizons"]):
                    if len(processed["y_true"].shape) > 2:  # Multi-horizon predictions
                        horizon_metrics[horizon] = {
                            "mse": mean_squared_error(processed["y_true"][:, :, i].flatten(), processed["y_pred"][:, :, i].flatten()),
                            "rmse": np.sqrt(mean_squared_error(processed["y_true"][:, :, i].flatten(), processed["y_pred"][:, :, i].flatten())),
                            "mae": mean_absolute_error(processed["y_true"][:, :, i].flatten(), processed["y_pred"][:, :, i].flatten()),
                            "r2": r2_score(processed["y_true"][:, :, i].flatten(), processed["y_pred"][:, :, i].flatten())
                        }
                
                metrics["per_horizon"] = horizon_metrics
            
            processed["metrics"] = metrics
        
        return processed
    
    else:  # climate domain
        # For climate, we expect predictions to contain:
        # - temperature: dict with true, pred, std
        # - co2: dict with true, pred, std (optional)
        if not isinstance(predictions, dict) or "temperature" not in predictions:
            st.error("❌ Invalid prediction format. Please run the model first.")
            return None
        
        # Create processed predictions dictionary
        processed = {
            "domain": "climate",
            "temperature": predictions.get("temperature", {}),
            "co2": predictions.get("co2", {}),
            "spatial": predictions.get("spatial", {})
        }
        
        # Add metrics for temperature
        if "true" in processed["temperature"] and "pred" in processed["temperature"]:
            # Get data
            y_true = processed["temperature"]["true"]
            y_pred = processed["temperature"]["pred"]
            
            # Calculate metrics
            metrics = {}
            
            metrics["mse"] = mean_squared_error(y_true.flatten(), y_pred.flatten())
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_true.flatten(), y_pred.flatten())
            metrics["r2"] = r2_score(y_true.flatten(), y_pred.flatten())
            
            # Add to temperature dict
            processed["temperature"]["metrics"] = metrics
        
        # Add metrics for CO2
        if "co2" in processed and "true" in processed["co2"] and "pred" in processed["co2"]:
            # Get data
            y_true = processed["co2"]["true"]
            y_pred = processed["co2"]["pred"]
            
            # Calculate metrics
            metrics = {}
            
            metrics["mse"] = mean_squared_error(y_true.flatten(), y_pred.flatten())
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_true.flatten(), y_pred.flatten())
            metrics["r2"] = r2_score(y_true.flatten(), y_pred.flatten())
            
            # Add to CO2 dict
            processed["co2"]["metrics"] = metrics
        
        return processed

def _show_model_performance_tab(predictions, domain):
    """Show model performance tab content."""
    st.header("Model Performance Metrics")
    
    if predictions is None:
        st.info("ℹ️ No predictions available. Please run the model first.")
        return
    
    if domain == "finance":
        # Check if metrics are available
        if "metrics" not in predictions:
            st.info("ℹ️ No metrics available. Please run the model first.")
            return
        
        # Get metrics
        metrics = predictions["metrics"]
        
        # Show overall metrics
        st.subheader("Overall Forecast Performance")
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="MSE",
                value=f"{metrics['mse']:.6f}"
            )
        
        with col2:
            st.metric(
                label="RMSE",
                value=f"{metrics['rmse']:.6f}"
            )
        
        with col3:
            st.metric(
                label="MAE",
                value=f"{metrics['mae']:.6f}"
            )
        
        with col4:
            st.metric(
                label="R²",
                value=f"{metrics['r2']:.4f}"
            )
        
        # Show per-asset metrics if available
        if "per_asset" in metrics:
            st.subheader("Performance by Asset")
            
            # Create dataframe
            asset_metrics_df = pd.DataFrame({
                "Asset": list(metrics["per_asset"].keys()),
                "MSE": [metrics["per_asset"][asset]["mse"] for asset in metrics["per_asset"]],
                "RMSE": [metrics["per_asset"][asset]["rmse"] for asset in metrics["per_asset"]],
                "MAE": [metrics["per_asset"][asset]["mae"] for asset in metrics["per_asset"]],
                "R²": [metrics["per_asset"][asset]["r2"] for asset in metrics["per_asset"]]
            })
            
            # Show table
            st.dataframe(asset_metrics_df)
            
            # Create bar chart for RMSE
            fig = px.bar(
                asset_metrics_df,
                x="Asset",
                y="RMSE",
                title="RMSE by Asset"
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Asset",
                yaxis_title="RMSE",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create bar chart for R²
            fig = px.bar(
                asset_metrics_df,
                x="Asset",
                y="R²",
                title="R² by Asset"
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Asset",
                yaxis_title="R²",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show per-horizon metrics if available
        if "per_horizon" in metrics:
            st.subheader("Performance by Forecast Horizon")
            
            # Create dataframe
            horizon_metrics_df = pd.DataFrame({
                "Horizon": list(metrics["per_horizon"].keys()),
                "MSE": [metrics["per_horizon"][horizon]["mse"] for horizon in metrics["per_horizon"]],
                "RMSE": [metrics["per_horizon"][horizon]["rmse"] for horizon in metrics["per_horizon"]],
                "MAE": [metrics["per_horizon"][horizon]["mae"] for horizon in metrics["per_horizon"]],
                "R²": [metrics["per_horizon"][horizon]["r2"] for horizon in metrics["per_horizon"]]
            })
            
            # Show table
            st.dataframe(horizon_metrics_df)
            
            # Create line chart for RMSE
            fig = px.line(
                horizon_metrics_df,
                x="Horizon",
                y="RMSE",
                title="RMSE by Forecast Horizon",
                markers=True
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Forecast Horizon",
                yaxis_title="RMSE",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create line chart for R²
            fig = px.line(
                horizon_metrics_df,
                x="Horizon",
                y="R²",
                title="R² by Forecast Horizon",
                markers=True
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Forecast Horizon",
                yaxis_title="R²",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # climate domain
        # Show temperature metrics if available
        if "temperature" in predictions and "metrics" in predictions["temperature"]:
            st.subheader("Temperature Forecast Performance")
            
            # Get metrics
            metrics = predictions["temperature"]["metrics"]
            
            # Create columns for metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="MSE",
                    value=f"{metrics['mse']:.6f}"
                )
            
            with col2:
                st.metric(
                    label="RMSE",
                    value=f"{metrics['rmse']:.6f}"
                )
            
            with col3:
                st.metric(
                    label="MAE",
                    value=f"{metrics['mae']:.6f}"
                )
            
            with col4:
                st.metric(
                    label="R²",
                    value=f"{metrics['r2']:.4f}"
                )
            
            # Show true vs predicted scatter plot
            st.subheader("Temperature: True vs Predicted")
            
            # Get data
            y_true = predictions["temperature"]["true"].flatten()
            y_pred = predictions["temperature"]["pred"].flatten()
            
            # Create scatter plot
            fig = px.scatter(
                x=y_true,
                y=y_pred,
                title="True vs Predicted Temperature Anomalies",
                labels={"x": "True Values", "y": "Predicted Values"}
            )
            
            # Add identity line
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Perfect Prediction"
            ))
            
            # Update layout
            fig.update_layout(
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show CO2 metrics if available
        if "co2" in predictions and "metrics" in predictions["co2"]:
            st.subheader("CO₂ Forecast Performance")
            
            # Get metrics
            metrics = predictions["co2"]["metrics"]
            
            # Create columns for metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="MSE",
                    value=f"{metrics['mse']:.6f}"
                )
            
            with col2:
                st.metric(
                    label="RMSE",
                    value=f"{metrics['rmse']:.6f}"
                )
            
            with col3:
                st.metric(
                    label="MAE",
                    value=f"{metrics['mae']:.6f}"
                )
            
            with col4:
                st.metric(
                    label="R²",
                    value=f"{metrics['r2']:.4f}"
                )
            
            # Show true vs predicted scatter plot
            st.subheader("CO₂: True vs Predicted")
            
            # Get data
            y_true = predictions["co2"]["true"].flatten()
            y_pred = predictions["co2"]["pred"].flatten()
            
            # Create scatter plot
            fig = px.scatter(
                x=y_true,
                y=y_pred,
                title="True vs Predicted CO₂ Concentration",
                labels={"x": "True Values (ppm)", "y": "Predicted Values (ppm)"}
            )
            
            # Add identity line
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Perfect Prediction"
            ))
            
            # Update layout
            fig.update_layout(
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

def _show_uncertainty_calibration_tab(predictions, domain):
    """Show uncertainty calibration tab content."""
    st.header("Uncertainty Calibration")
    
    if predictions is None:
        st.info("ℹ️ No predictions available. Please run the model first.")
        return
    
    if domain == "finance":
        # Check if uncertainty estimates are available
        if "y_std" not in predictions or predictions["y_std"] is None:
            st.info("ℹ️ No uncertainty estimates available.")
            return
        
        # Get data
        y_true = predictions["y_true"]
        y_pred = predictions["y_pred"]
        y_std = predictions["y_std"]
        
        # Calculate standardized residuals
        z_scores = (y_true - y_pred) / y_std
        
        # Show histogram of standardized residuals
        st.subheader("Standardized Residuals Distribution")
        
        # Create histogram
        fig = px.histogram(
            z_scores.flatten(),
            title="Standardized Residuals (z-scores)",
            labels={"value": "z-score", "count": "Frequency"},
            nbins=50
        )
        
        # Add normal distribution line
        x = np.linspace(-4, 4, 1000)
        y = np.exp(-x**2/2) / np.sqrt(2*np.pi) * len(z_scores.flatten()) * (8/50)
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="red"),
            name="Normal Distribution"
        ))
        
        # Update layout
        fig.update_layout(
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate calibration curve
        st.subheader("Uncertainty Calibration Curve")
        
        # Define confidence levels
        confidence_levels = np.linspace(0.1, 0.9, 9)
        
        # Calculate expected vs actual coverage
        expected_coverage = []
        actual_coverage = []
        
        for p in confidence_levels:
            # Calculate z-score for this confidence level
            z = np.percentile(z_scores.flatten(), (1 - p) * 50)
            
            # Calculate interval width
            half_width = np.abs(z * y_std)
            
            # Calculate coverage
            coverage = np.mean(
                np.abs(y_true - y_pred) <= half_width
            )
            
            expected_coverage.append(p)
            actual_coverage.append(coverage)
        
        # Create dataframe
        calibration_df = pd.DataFrame({
            "Expected Coverage": expected_coverage,
            "Actual Coverage": actual_coverage
        })
        
        # Show table
        st.dataframe(calibration_df)
        
        # Create line chart
        fig = go.Figure()
        
        # Add identity line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Perfect Calibration"
        ))
        
        # Add actual calibration curve
        fig.add_trace(go.Scatter(
            x=expected_coverage,
            y=actual_coverage,
            mode="lines+markers",
            name="Model Calibration"
        ))
        
        # Update layout
        fig.update_layout(
            title="Uncertainty Calibration Curve",
            xaxis_title="Expected Coverage",
            yaxis_title="Actual Coverage",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate calibration error
        calibration_error = np.mean(
            np.abs(np.array(expected_coverage) - np.array(actual_coverage))
        )
        
        st.metric(
            label="Mean Calibration Error",
            value=f"{calibration_error:.4f}"
        )
    
    else:  # climate domain
        # Check if uncertainty estimates are available for temperature
        if "temperature" in predictions and "std" in predictions["temperature"]:
            st.subheader("Temperature Uncertainty Calibration")
            
            # Get data
            y_true = predictions["temperature"]["true"]
            y_pred = predictions["temperature"]["pred"]
            y_std = predictions["temperature"]["std"]
            
            # Calculate standardized residuals
            z_scores = (y_true - y_pred) / y_std
            
            # Show histogram of standardized residuals
            st.subheader("Standardized Residuals Distribution")
            
            # Create histogram
            fig = px.histogram(
                z_scores.flatten(),
                title="Standardized Temperature Residuals (z-scores)",
                labels={"value": "z-score", "count": "Frequency"},
                nbins=50
            )
            
            # Add normal distribution line
            x = np.linspace(-4, 4, 1000)
            y = np.exp(-x**2/2) / np.sqrt(2*np.pi) * len(z_scores.flatten()) * (8/50)
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color="red"),
                name="Normal Distribution"
            ))
            
            # Update layout
            fig.update_layout(
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate calibration curve
            st.subheader("Temperature Uncertainty Calibration Curve")
            
            # Define confidence levels
            confidence_levels = np.linspace(0.1, 0.9, 9)
            
            # Calculate expected vs actual coverage
            expected_coverage = []
            actual_coverage = []
            
            for p in confidence_levels:
                # Calculate z-score for this confidence level
                z = np.percentile(z_scores.flatten(), (1 - p) * 50)
                
                # Calculate interval width
                half_width = np.abs(z * y_std)
                
                # Calculate coverage
                coverage = np.mean(
                    np.abs(y_true - y_pred) <= half_width
                )
                
                expected_coverage.append(p)
                actual_coverage.append(coverage)
            
            # Create dataframe
            calibration_df = pd.DataFrame({
                "Expected Coverage": expected_coverage,
                "Actual Coverage": actual_coverage
            })
            
            # Show table
            st.dataframe(calibration_df)
            
            # Create line chart
            fig = go.Figure()
            
            # Add identity line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Perfect Calibration"
            ))
            
            # Add actual calibration curve
            fig.add_trace(go.Scatter(
                x=expected_coverage,
                y=actual_coverage,
                mode="lines+markers",
                name="Model Calibration"
            ))
            
            # Update layout
            fig.update_layout(
                title="Temperature Uncertainty Calibration Curve",
                xaxis_title="Expected Coverage",
                yaxis_title="Actual Coverage",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate calibration error
            calibration_error = np.mean(
                np.abs(np.array(expected_coverage) - np.array(actual_coverage))
            )
            
            st.metric(
                label="Mean Temperature Calibration Error",
                value=f"{calibration_error:.4f}"
            )
        
        # Check if uncertainty estimates are available for CO2
        if "co2" in predictions and "std" in predictions["co2"]:
            st.subheader("CO₂ Uncertainty Calibration")
            
            # Get data
            y_true = predictions["co2"]["true"]
            y_pred = predictions["co2"]["pred"]
            y_std = predictions["co2"]["std"]
            
            # Calculate standardized residuals
            z_scores = (y_true - y_pred) / y_std
            
            # Show histogram of standardized residuals
            st.subheader("Standardized CO₂ Residuals Distribution")
            
            # Create histogram
            fig = px.histogram(
                z_scores.flatten(),
                title="Standardized CO₂ Residuals (z-scores)",
                labels={"value": "z-score", "count": "Frequency"},
                nbins=50
            )
            
            # Add normal distribution line
            x = np.linspace(-4, 4, 1000)
            y = np.exp(-x**2/2) / np.sqrt(2*np.pi) * len(z_scores.flatten()) * (8/50)
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color="red"),
                name="Normal Distribution"
            ))
            
            # Update layout
            fig.update_layout(
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate calibration curve
            st.subheader("CO₂ Uncertainty Calibration Curve")
            
            # Define confidence levels
            confidence_levels = np.linspace(0.1, 0.9, 9)
            
            # Calculate expected vs actual coverage
            expected_coverage = []
            actual_coverage = []
            
            for p in confidence_levels:
                # Calculate z-score for this confidence level
                z = np.percentile(z_scores.flatten(), (1 - p) * 50)
                
                # Calculate interval width
                half_width = np.abs(z * y_std)
                
                # Calculate coverage
                coverage = np.mean(
                    np.abs(y_true - y_pred) <= half_width
                )
                
                expected_coverage.append(p)
                actual_coverage.append(coverage)
            
            # Create dataframe
            calibration_df = pd.DataFrame({
                "Expected Coverage": expected_coverage,
                "Actual Coverage": actual_coverage
            })
            
            # Show table
            st.dataframe(calibration_df)
            
            # Create line chart
            fig = go.Figure()
            
            # Add identity line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Perfect Calibration"
            ))
            
            # Add actual calibration curve
            fig.add_trace(go.Scatter(
                x=expected_coverage,
                y=actual_coverage,
                mode="lines+markers",
                name="Model Calibration"
            ))
            
            # Update layout
            fig.update_layout(
                title="CO₂ Uncertainty Calibration Curve",
                xaxis_title="Expected Coverage",
                yaxis_title="Actual Coverage",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate calibration error
            calibration_error = np.mean(
                np.abs(np.array(expected_coverage) - np.array(actual_coverage))
            )
            
            st.metric(
                label="Mean CO₂ Calibration Error",
                value=f"{calibration_error:.4f}"
            )

def _show_error_analysis_tab(predictions, domain):
    """Show error analysis tab content."""
    st.header("Error Analysis")
    
    if predictions is None:
        st.info("ℹ️ No predictions available. Please run the model first.")
        return
    
    if domain == "finance":
        # Check if predictions are available
        if "y_true" not in predictions or "y_pred" not in predictions:
            st.info("ℹ️ No predictions available for error analysis.")
            return
        
        # Get data
        y_true = predictions["y_true"]
        y_pred = predictions["y_pred"]
        
        # Calculate errors
        errors = y_true - y_pred
        
        # Show error distribution
        st.subheader("Error Distribution")
        
        # Create histogram
        fig = px.histogram(
            errors.flatten(),
            title="Prediction Error Distribution",
            labels={"value": "Error", "count": "Frequency"},
            nbins=50
        )
        
        # Add normal distribution line
        mu = np.mean(errors.flatten())
        sigma = np.std(errors.flatten())
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        y = np.exp(-(x - mu)**2/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi)) * len(errors.flatten()) * (8*sigma/50)
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="red"),
            name="Normal Distribution"
        ))
        
        # Update layout
        fig.update_layout(
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show errors over time if dates are available
        if "dates" in predictions and predictions["dates"] is not None:
            st.subheader("Errors Over Time")
            
            # Get dates
            dates = predictions["dates"]
            
            # Select asset for visualization
            if "assets" in predictions and predictions["assets"] is not None:
                assets = predictions["assets"]
                
                selected_asset = st.selectbox(
                    "Select Asset for Error Analysis",
                    assets
                )
                
                # Get asset index
                asset_idx = assets.index(selected_asset)
                
                # Get errors for this asset
                asset_errors = errors[:, asset_idx]
                
                # Create line chart
                fig = go.Figure()
                
                # Add error line
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=asset_errors,
                    mode="lines",
                    name="Prediction Error"
                ))
                
                # Add horizontal line at zero
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="black"
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Prediction Errors Over Time for {selected_asset}",
                    xaxis_title="Date",
                    yaxis_title="Error",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show error vs prediction magnitude
                st.subheader("Error vs Prediction Magnitude")
                
                # Create scatter plot
                fig = px.scatter(
                    x=y_pred[:, asset_idx],
                    y=asset_errors,
                    title=f"Error vs Prediction Magnitude for {selected_asset}",
                    labels={"x": "Predicted Value", "y": "Error"}
                )
                
                # Add horizontal line at zero
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="black"
                )
                
                # Update layout
                fig.update_layout(
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show error autocorrelation
                st.subheader("Error Autocorrelation")
                
                # Calculate autocorrelation
                max_lag = min(20, len(asset_errors) - 1)
                autocorr = [1]  # Lag 0
                
                for lag in range(1, max_lag + 1):
                    # Calculate autocorrelation
                    ac = np.corrcoef(asset_errors[:-lag], asset_errors[lag:])[0, 1]
                    autocorr.append(ac)
                
                # Create bar chart
                fig = go.Figure()
                
                # Add bars
                fig.add_trace(go.Bar(
                    x=list(range(max_lag + 1)),
                    y=autocorr
                ))
                
                # Add confidence bounds
                confidence = 1.96 / np.sqrt(len(asset_errors))
                
                fig.add_hline(
                    y=confidence,
                    line_dash="dash",
                    line_color="red"
                )
                
                fig.add_hline(
                    y=-confidence,
                    line_dash="dash",
                    line_color="red"
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Error Autocorrelation for {selected_asset}",
                    xaxis_title="Lag",
                    yaxis_title="Autocorrelation",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    else:  # climate domain
        # Check if temperature predictions are available
        if "temperature" in predictions and "true" in predictions["temperature"] and "pred" in predictions["temperature"]:
            st.subheader("Temperature Error Analysis")
            
            # Get data
            y_true = predictions["temperature"]["true"]
            y_pred = predictions["temperature"]["pred"]
            
            # Calculate errors
            errors = y_true - y_pred
            
            # Show error distribution
            st.subheader("Temperature Error Distribution")
            
            # Create histogram
            fig = px.histogram(
                errors.flatten(),
                title="Temperature Prediction Error Distribution",
                labels={"value": "Error (°C)", "count": "Frequency"},
                nbins=50
            )
            
            # Add normal distribution line
            mu = np.mean(errors.flatten())
            sigma = np.std(errors.flatten())
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
            y = np.exp(-(x - mu)**2/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi)) * len(errors.flatten()) * (8*sigma/50)
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color="red"),
                name="Normal Distribution"
            ))
            
            # Update layout
            fig.update_layout(
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show errors over time if dates are available
            if "dates" in predictions["temperature"]:
                st.subheader("Temperature Errors Over Time")
                
                # Get dates
                dates = predictions["temperature"]["dates"]
                
                # Create line chart
                fig = go.Figure()
                
                # Add error line
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=errors.flatten(),
                    mode="lines",
                    name="Temperature Prediction Error"
                ))
                
                # Add horizontal line at zero
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="black"
                )
                
                # Update layout
                fig.update_layout(
                    title="Temperature Prediction Errors Over Time",
                    xaxis_title="Date",
                    yaxis_title="Error (°C)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show error vs prediction magnitude
                st.subheader("Error vs Prediction Magnitude")
                
                # Create scatter plot
                fig = px.scatter(
                    x=y_pred.flatten(),
                    y=errors.flatten(),
                    title="Error vs Prediction Magnitude",
                    labels={"x": "Predicted Temperature (°C)", "y": "Error (°C)"}
                )
                
                # Add horizontal line at zero
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="black"
                )
                
                # Update layout
                fig.update_layout(
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Check if CO2 predictions are available
        if "co2" in predictions and "true" in predictions["co2"] and "pred" in predictions["co2"]:
            st.subheader("CO₂ Error Analysis")
            
            # Get data
            y_true = predictions["co2"]["true"]
            y_pred = predictions["co2"]["pred"]
            
            # Calculate errors
            errors = y_true - y_pred
            
            # Show error distribution
            st.subheader("CO₂ Error Distribution")
            
            # Create histogram
            fig = px.histogram(
                errors.flatten(),
                title="CO₂ Prediction Error Distribution",
                labels={"value": "Error (ppm)", "count": "Frequency"},
                nbins=50
            )
            
            # Add normal distribution line
            mu = np.mean(errors.flatten())
            sigma = np.std(errors.flatten())
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
            y = np.exp(-(x - mu)**2/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi)) * len(errors.flatten()) * (8*sigma/50)
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color="red"),
                name="Normal Distribution"
            ))
            
            # Update layout
            fig.update_layout(
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show errors over time if dates are available
            if "dates" in predictions["co2"]:
                st.subheader("CO₂ Errors Over Time")
                
                # Get dates
                dates = predictions["co2"]["dates"]
                
                # Create line chart
                fig = go.Figure()
                
                # Add error line
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=errors.flatten(),
                    mode="lines",
                    name="CO₂ Prediction Error"
                ))
                
                # Add horizontal line at zero
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="black"
                )
                
                # Update layout
                fig.update_layout(
                    title="CO₂ Prediction Errors Over Time",
                    xaxis_title="Date",
                    yaxis_title="Error (ppm)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

def _show_model_explanation_tab(predictions, domain):
    """Show model explanation tab content."""
    st.header("Model Explanation")
    
    # Placeholder for model-specific explanations
    st.info("ℹ️ Model explanation features will be implemented in a future version. This would include feature importance, SHAP values, and attention visualizations.")
    
    # Show architecture description
    st.subheader("Model Architecture")
    
    # Description based on domain
    if domain == "finance":
        st.markdown("""
        ### Finance Adapter Architecture
        
        The finance adapter uses a specialized architecture for financial time series forecasting with uncertainty quantification:
        
        - **Time Series Encoder**: Processes historical price data using temporal convolutional networks and self-attention
        - **Return Forecaster**: Predicts future returns with multi-head forecasting for different time horizons
        - **Uncertainty Quantification**: Implements multiple methods for estimating uncertainty:
            - Heteroscedastic uncertainty (direct variance prediction)
            - Monte Carlo dropout sampling
            - Bayesian neural network with variational inference
            - Ensemble methods (when available)
        
        The model is designed to capture complex temporal patterns in financial data while providing well-calibrated uncertainty estimates.
        """)
    else:  # climate
        st.markdown("""
        ### Climate Adapter Architecture
        
        The climate adapter uses a specialized architecture for climate variable forecasting with uncertainty quantification:
        
        - **Spatial-Temporal Encoder**: Processes historical climate data using CNN layers for spatial patterns and self-attention for temporal dependencies
        - **Temperature Predictor**: Forecasts future temperature anomalies with trend and seasonality decomposition
        - **CO₂ Predictor**: Models carbon dioxide concentration with autoregressive components
        - **Uncertainty Quantification**: Implements multiple methods for estimating uncertainty:
            - Heteroscedastic uncertainty (direct variance prediction)
            - Monte Carlo dropout sampling
            - Bayesian neural network with variational inference
            - Ensemble methods (when available)
        
        The model is designed to capture both spatial and temporal patterns in climate data while providing well-calibrated uncertainty estimates.
        """)
    
    # Show example uncertainty visualization
    st.subheader("Uncertainty Visualization Example")
    
    # Create example plot based on domain
    if domain == "finance":
        # Create dummy data
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.1 * x
        y_pred = y + np.random.normal(0, 0.1, size=len(x))
        y_std = 0.1 + 0.05 * np.abs(x - 5)
        
        # Create figure
        fig = go.Figure()
        
        # Add true values
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="black"),
            name="True Returns"
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=x,
            y=y_pred,
            mode="lines",
            line=dict(color="blue"),
            name="Predicted Returns"
        ))
        
        # Add uncertainty
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y_pred + 2*y_std, (y_pred - 2*y_std)[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line=dict(color="rgba(0, 0, 255, 0)"),
            name="95% Confidence Interval"
        ))
        
        # Update layout
        fig.update_layout(
            title="Example Finance Prediction with Uncertainty",
            xaxis_title="Time",
            yaxis_title="Returns",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:  # climate
        # Create dummy data
        x = np.linspace(0, 10, 100)
        y = 0.1 * x + 0.05 * x**2 + 0.2 * np.sin(x)
        y_pred = y + np.random.normal(0, 0.1, size=len(x))
        y_std = 0.1 + 0.05 * np.abs(x - 5)
        
        # Create figure
        fig = go.Figure()
        
        # Add true values
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="black"),
            name="True Temperature"
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=x,
            y=y_pred,
            mode="lines",
            line=dict(color="red"),
            name="Predicted Temperature"
        ))
        
        # Add uncertainty
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y_pred + 2*y_std, (y_pred - 2*y_std)[::-1]]),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(color="rgba(255, 0, 0, 0)"),
            name="95% Confidence Interval"
        ))
        
        # Update layout
        fig.update_layout(
            title="Example Temperature Prediction with Uncertainty",
            xaxis_title="Time",
            yaxis_title="Temperature Anomaly (°C)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Main function to test the component independently
if __name__ == "__main__":
    show_diagnostics_page()