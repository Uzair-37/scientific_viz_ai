"""
Climate page component for the multi-modal AI web application.

This module implements the climate page UI that allows users to
analyze climate data with uncertainty quantification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import torch
from io import BytesIO
import base64
import os
import tempfile
from scipy import stats

# Import domain-specific components
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from models.domain_adapters.climate_adapter import ClimateAdapter
except ImportError:
    # Fallback for different deployment environments
    ClimateAdapter = None

try:
    from web_app.utils.download import download_button
except ImportError:
    # Fallback function if download utility is not available
    def download_button(data, file_name, label="Download", mime="text/plain"):
        st.download_button(label=label, data=data, file_name=file_name, mime=mime)

def show_climate_page():
    """Display the climate page with temperature forecasting and uncertainty visualization."""
    # Main title
    st.title("Climate Prediction with Uncertainty Quantification")
    
    # Check if ClimateAdapter is available
    if ClimateAdapter is None:
        st.error("ClimateAdapter module is not available. Please check the installation.")
        st.info("Using demo mode with limited functionality.")
        _show_demo_mode()
        return
    
    # Create tabs for different climate sections
    tabs = st.tabs(["Data", "Temperature", "CO₂ Prediction", "Spatial Analysis"])
    
    # Check if model needs to be run
    run_model = st.session_state.get("run_model", False)
    
    # Get data source
    data_source = st.session_state.get("data_source", "sample")
    
    # Get or generate data based on source
    if data_source == "sample":
        data = _get_sample_data()
    elif data_source == "noaa_data" or data_source == "berkeley_earth":
        data = _get_climate_api_data()
    elif data_source == "upload_data":
        data = _get_uploaded_data()
    else:
        data = _get_sample_data()  # Fallback to sample data
    
    # Store data in session state
    st.session_state.climate_data = data
    
    # Run model if triggered
    if run_model:
        with st.spinner("Running climate model with uncertainty quantification..."):
            results = _run_climate_model(data)
            st.session_state.results = results
            st.session_state.run_model = False  # Reset flag
    
    # Get results from session state
    results = st.session_state.get("results", None)
    
    # Data Tab
    with tabs[0]:
        _show_climate_data_tab(data)
    
    # Temperature Tab
    with tabs[1]:
        _show_temperature_tab(data, results)
    
    # CO₂ Tab
    with tabs[2]:
        _show_co2_tab(data, results)
    
    # Spatial Analysis Tab
    with tabs[3]:
        _show_spatial_tab(data, results)

def _get_sample_data():
    """Get sample climate data for demonstration."""
    # Create temperature time series data
    # Start date: January 1, 1950
    start_date = datetime(1950, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Create date range with monthly frequency
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create temperature anomaly data with trend and seasonality
    n_months = len(date_range)
    
    # Create time index for trend
    time_idx = np.arange(n_months) / 12  # Convert to years
    
    # Create trend component (warming trend)
    # Accelerating warming trend
    trend = 0.2 * time_idx + 0.003 * time_idx**2
    
    # Create seasonal component
    seasonality = 0.8 * np.sin(2 * np.pi * np.arange(n_months) / 12)
    
    # Create noise component
    noise = np.random.normal(0, 0.2, n_months)
    
    # Combine components
    temperature_anomaly = trend + seasonality + noise
    
    # Create dataframe
    temp_df = pd.DataFrame({
        'Date': date_range,
        'Global': temperature_anomaly,
        'Northern_Hemisphere': temperature_anomaly + 0.2 + np.random.normal(0, 0.1, n_months),
        'Southern_Hemisphere': temperature_anomaly - 0.1 + np.random.normal(0, 0.1, n_months),
        'Land': temperature_anomaly + 0.3 + np.random.normal(0, 0.15, n_months),
        'Ocean': temperature_anomaly - 0.2 + np.random.normal(0, 0.1, n_months)
    })
    
    # Set date as index
    temp_df.set_index('Date', inplace=True)
    
    # Create CO2 data
    # Start with pre-industrial level around 280 ppm
    base_co2 = 280
    
    # Create time series for CO2
    time_idx_years = time_idx
    
    # Exponential growth with seasonal oscillation
    co2 = base_co2 + 0.5 * np.exp(0.015 * time_idx_years) + 20 * (1 - np.exp(-0.05 * time_idx_years))
    seasonal_co2 = 2 * np.sin(2 * np.pi * np.arange(n_months) / 12)
    
    co2_with_seasonality = co2 + seasonal_co2
    
    # Add noise
    co2_with_noise = co2_with_seasonality + np.random.normal(0, 0.2, n_months)
    
    # Create CO2 dataframe
    co2_df = pd.DataFrame({
        'Date': date_range,
        'CO2_Concentration': co2_with_noise
    })
    
    # Set date as index
    co2_df.set_index('Date', inplace=True)
    
    # Combine data
    climate_data = {
        'temperature': temp_df,
        'co2': co2_df,
        'metadata': {
            'source': 'synthetic',
            'description': 'Synthetic climate data for demonstration purposes',
            'variables': {
                'temperature': 'Temperature anomaly in °C relative to 1951-1980 average',
                'co2': 'CO2 concentration in parts per million (ppm)'
            }
        }
    }
    
    # Create spatial data (simplified grid)
    # Create a 10x10 grid of temperature anomalies
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Create spatial data with latitudinal gradient and warming trend
    spatial_data = []
    
    # Generate spatial data for selected years
    years = [1950, 1980, 2000, 2023]
    
    for year in years:
        # Get year index
        year_idx = (year - 1950)
        
        # Base warming value from the trend
        base_warming = 0.2 * year_idx / 12 + 0.003 * (year_idx / 12)**2
        
        # Create grid with latitudinal gradient (more warming at poles)
        grid_data = base_warming + 0.5 * np.abs(lat_grid / 90) + np.random.normal(0, 0.3, (10, 10))
        
        # Store data
        spatial_data.append({
            'year': year,
            'lat': lat,
            'lon': lon,
            'data': grid_data
        })
    
    # Add spatial data to climate data
    climate_data['spatial'] = spatial_data
    
    # Success message
    st.success("✅ Sample climate data loaded successfully")
    
    return climate_data

def _get_climate_api_data():
    """Get climate data from API based on user selection."""
    # In a real implementation, this would connect to climate data APIs
    # For now, we'll return sample data with a message
    
    # Get climate data type
    data_type = st.session_state.get("climate_data_type", "Temperature")
    
    # Get region
    region = st.session_state.get("climate_region", "Global")
    
    # Get year range
    start_year = st.session_state.get("climate_start_year", 1990)
    end_year = st.session_state.get("climate_end_year", 2020)
    
    # Display message
    st.info(f"📊 Loading {data_type} data for {region} from {start_year} to {end_year}")
    
    # Return sample data
    return _get_sample_data()

def _get_uploaded_data():
    """Process uploaded climate data."""
    # Get uploaded file from session state
    uploaded_file = st.session_state.get("uploaded_file", None)
    
    if uploaded_file is not None:
        try:
            # Process based on file type
            if uploaded_file.name.endswith('.csv'):
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                # Check if there's a date column
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                
                if date_cols:
                    # Set date column as index
                    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                    df.set_index(date_cols[0], inplace=True)
                
                # Determine if it's temperature or CO2 data
                if any('temp' in col.lower() for col in df.columns):
                    # Temperature data
                    temp_df = df
                    co2_df = pd.DataFrame()  # Empty DataFrame
                    
                    climate_data = {
                        'temperature': temp_df,
                        'co2': co2_df,
                        'metadata': {
                            'source': 'user_upload',
                            'description': 'User uploaded temperature data',
                            'variables': {
                                'temperature': 'Temperature data from user upload'
                            }
                        }
                    }
                
                elif any('co2' in col.lower() for col in df.columns):
                    # CO2 data
                    co2_df = df
                    temp_df = pd.DataFrame()  # Empty DataFrame
                    
                    climate_data = {
                        'temperature': temp_df,
                        'co2': co2_df,
                        'metadata': {
                            'source': 'user_upload',
                            'description': 'User uploaded CO2 data',
                            'variables': {
                                'co2': 'CO2 data from user upload'
                            }
                        }
                    }
                
                else:
                    # Unclear data type, assume temperature
                    temp_df = df
                    co2_df = pd.DataFrame()  # Empty DataFrame
                    
                    climate_data = {
                        'temperature': temp_df,
                        'co2': co2_df,
                        'metadata': {
                            'source': 'user_upload',
                            'description': 'User uploaded data (assumed temperature)',
                            'variables': {
                                'temperature': 'Data from user upload'
                            }
                        }
                    }
                
                # No spatial data for CSV
                climate_data['spatial'] = []
                
                # Success message
                st.success(f"✅ Uploaded data processed successfully")
                return climate_data
            
            elif uploaded_file.name.endswith('.nc'):
                # NetCDF file
                st.warning("⚠️ NetCDF processing not implemented in demo")
                
                # Return sample data
                return _get_sample_data()
            
            else:
                # Unsupported file type
                st.error(f"❌ Unsupported file type: {uploaded_file.name}")
                return _get_sample_data()
        
        except Exception as e:
            # Error message
            st.error(f"❌ Error processing uploaded file: {str(e)}")
            return _get_sample_data()
    
    else:
        # No file uploaded
        st.warning("⚠️ No file uploaded, using sample data instead")
        return _get_sample_data()

def _prepare_climate_features(data):
    """Prepare features for climate forecasting."""
    # Get temperature data
    temp_df = data['temperature']
    
    # Create copy to avoid modifying original
    df = temp_df.copy()
    
    # Add time features
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['day_of_year'] = df.index.dayofyear
    
    # Add lag features
    for col in temp_df.columns:
        for lag in [1, 3, 6, 12]:  # 1, 3, 6, 12 month lags
            df[f'{col}_lag_{lag}'] = temp_df[col].shift(lag)
    
    # Add rolling mean features
    for col in temp_df.columns:
        for window in [3, 6, 12, 60]:  # 3, 6, 12, 60 month windows
            df[f'{col}_roll_{window}'] = temp_df[col].rolling(window=window).mean()
    
    # Add rolling std features
    for col in temp_df.columns:
        for window in [12, 60]:  # 12, 60 month windows
            df[f'{col}_std_{window}'] = temp_df[col].rolling(window=window).std()
    
    # Add CO2 data if available
    if 'co2' in data and not data['co2'].empty:
        co2_df = data['co2']
        
        # Merge with temperature data
        for col in co2_df.columns:
            df[col] = co2_df[col]
        
        # Add CO2 lag features
        for col in co2_df.columns:
            for lag in [1, 3, 6, 12]:  # 1, 3, 6, 12 month lags
                df[f'{col}_lag_{lag}'] = co2_df[col].shift(lag)
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    return df

def _run_climate_model(data):
    """Run climate model with uncertainty quantification."""
    # Get model type
    model_type = st.session_state.get("model_type", "heteroscedastic")
    
    # Prepare features
    features_df = _prepare_climate_features(data)
    
    # Define forecast horizon
    horizon = st.session_state.get("climate_horizon", 12)
    
    # Get temperature data
    temp_df = data['temperature']
    
    # Get CO2 data
    co2_df = data['co2']
    
    # Extract the target variables
    if 'Global' in temp_df.columns:
        temp_target = temp_df['Global']
    else:
        temp_target = temp_df.iloc[:, 0]  # Use first column
    
    if not co2_df.empty and 'CO2_Concentration' in co2_df.columns:
        co2_target = co2_df['CO2_Concentration']
    elif not co2_df.empty:
        co2_target = co2_df.iloc[:, 0]  # Use first column
    else:
        # Generate synthetic CO2 data if not available
        co2_target = pd.Series(
            np.linspace(350, 420, len(temp_target)) + np.random.normal(0, 1, len(temp_target)),
            index=temp_target.index
        )
    
    # Create sequences for time series forecasting
    sequence_length = 60  # Use last 60 months to predict next 'horizon' months
    
    # Create sequences from the features dataframe
    X, y_temp, y_co2 = _create_climate_sequences(features_df, temp_target, co2_target, sequence_length, horizon)
    
    # Split into train and test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_temp_train, y_temp_test = y_temp[:train_size], y_temp[train_size:]
    y_co2_train, y_co2_test = y_co2[:train_size], y_co2[train_size:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_temp_train_tensor = torch.tensor(y_temp_train, dtype=torch.float32)
    y_co2_train_tensor = torch.tensor(y_co2_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_temp_test_tensor = torch.tensor(y_temp_test, dtype=torch.float32)
    y_co2_test_tensor = torch.tensor(y_co2_test, dtype=torch.float32)
    
    # Define model parameters
    input_dim = X_train.shape[2]  # Number of features
    embed_dim = st.session_state.get("embedding_dim", 256)
    hidden_dim = st.session_state.get("hidden_dim", 128)
    dropout_rate = st.session_state.get("dropout_rate", 0.1)
    activation = st.session_state.get("activation", "relu")
    
    # Create metadata features (simplified)
    metadata = np.zeros((1, 8))  # Placeholder metadata
    metadata_tensor = torch.tensor(metadata, dtype=torch.float32)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClimateAdapter(
        input_channels=1,  # Not used in this demo
        time_series_dim=input_dim,
        metadata_dim=metadata.shape[1],
        spatial_dim=(10, 10),  # Not used in this demo
        temporal_dim=sequence_length,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=1,  # Single output dimension
        horizon=horizon,
        uncertainty_type=model_type,
        dropout_rate=dropout_rate,
        activation=activation,
        data_type="station"  # Using station data for time series
    ).to(device)
    
    # For demo purposes, we'll just make predictions without training
    # In a real implementation, you would train the model here
    
    # Prepare test input
    test_input = {
        "time_series": X_test_tensor.to(device),
        "metadata": metadata_tensor.to(device)
    }
    
    # Make predictions with uncertainty
    with torch.no_grad():
        predictions = model.predict_with_uncertainty(test_input)
    
    # Process predictions
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        temp_mean = predictions["temperature_mean"].cpu().numpy()
        temp_var = predictions["temperature_var"].cpu().numpy()
        co2_mean = predictions["co2_mean"].cpu().numpy()
        co2_var = predictions["co2_var"].cpu().numpy()
    else:
        # No uncertainty model
        temp_mean = predictions["temperature"].cpu().numpy()
        temp_var = np.zeros_like(temp_mean)
        co2_mean = predictions["co2"].cpu().numpy()
        co2_var = np.zeros_like(co2_mean)
    
    # Get last date from data
    last_date = temp_df.index[-1]
    
    # Create forecast dates
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq='MS'  # Month start
    )
    
    # Calculate standard deviations
    temp_std = np.sqrt(temp_var)
    co2_std = np.sqrt(co2_var)
    
    # Create confidence intervals
    temp_lower = temp_mean - 2 * temp_std
    temp_upper = temp_mean + 2 * temp_std
    co2_lower = co2_mean - 2 * co2_std
    co2_upper = co2_mean + 2 * co2_std
    
    # Create results dictionary
    results = {
        "input_data": data,
        "features": features_df,
        "forecast_dates": forecast_dates,
        "temperature_mean": temp_mean[-1, :, 0],  # Last batch, all horizons, first output
        "temperature_std": temp_std[-1, :, 0],
        "temperature_lower": temp_lower[-1, :, 0],
        "temperature_upper": temp_upper[-1, :, 0],
        "co2_mean": co2_mean[-1, :, 0],
        "co2_std": co2_std[-1, :, 0],
        "co2_lower": co2_lower[-1, :, 0],
        "co2_upper": co2_upper[-1, :, 0],
        "model_type": model_type,
        "horizon": horizon
    }
    
    return results

def _create_climate_sequences(features, temp_target, co2_target, sequence_length, horizon):
    """Create sequences for climate time series forecasting."""
    X, y_temp, y_co2 = [], [], []
    
    for i in range(len(features) - sequence_length - horizon + 1):
        # Features sequence
        X.append(features.iloc[i:i+sequence_length].values)
        
        # Temperature target sequence
        y_temp.append(temp_target.iloc[i+sequence_length:i+sequence_length+horizon].values)
        
        # CO2 target sequence
        y_co2.append(co2_target.iloc[i+sequence_length:i+sequence_length+horizon].values)
    
    # Reshape targets to [batch_size, horizon, 1]
    y_temp = np.array(y_temp).reshape(-1, horizon, 1)
    y_co2 = np.array(y_co2).reshape(-1, horizon, 1)
    
    return np.array(X), y_temp, y_co2

def _show_climate_data_tab(data):
    """Show climate data tab content."""
    st.header("Climate Data")
    
    # Data overview
    st.subheader("Temperature Data")
    
    temp_df = data['temperature']
    st.dataframe(temp_df.head())
    
    # Temperature plot
    st.subheader("Temperature Anomaly")
    
    # Create figure
    fig = go.Figure()
    
    # Add time series for each temperature variable
    for col in temp_df.columns:
        fig.add_trace(go.Scatter(
            x=temp_df.index,
            y=temp_df[col],
            mode='lines',
            name=col.replace('_', ' ')
        ))
    
    # Update layout
    fig.update_layout(
        title="Temperature Anomaly (°C relative to 1951-1980 average)",
        xaxis_title="Year",
        yaxis_title="Temperature Anomaly (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # CO2 data
    st.subheader("CO₂ Concentration Data")
    
    co2_df = data['co2']
    
    if not co2_df.empty:
        st.dataframe(co2_df.head())
        
        # CO2 plot
        st.subheader("CO₂ Concentration")
        
        # Create figure
        fig = go.Figure()
        
        # Add time series for CO2
        for col in co2_df.columns:
            fig.add_trace(go.Scatter(
                x=co2_df.index,
                y=co2_df[col],
                mode='lines',
                name=col.replace('_', ' ')
            ))
        
        # Update layout
        fig.update_layout(
            title="CO₂ Concentration (ppm)",
            xaxis_title="Year",
            yaxis_title="CO₂ (ppm)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No CO₂ data available.")
    
    # Spatial data if available
    st.subheader("Spatial Data")
    
    if 'spatial' in data and data['spatial']:
        # Create year selector
        spatial_data = data['spatial']
        years = [item['year'] for item in spatial_data]
        
        selected_year = st.selectbox(
            "Select Year",
            years,
            index=len(years) - 1,  # Default to last year
            key="co2_year_selector"
        )
        
        # Get data for selected year
        selected_data = next((item for item in spatial_data if item['year'] == selected_year), None)
        
        if selected_data:
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=selected_data['data'],
                x=selected_data['lon'],
                y=selected_data['lat'],
                colorscale='RdBu_r',
                colorbar=dict(title='Temperature<br>Anomaly (°C)'),
                zmin=-2,
                zmax=2
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Temperature Anomaly Map ({selected_year})",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No spatial data available.")
    
    # Download data buttons
    col1, col2 = st.columns(2)
    
    with col1:
        temp_csv = temp_df.to_csv()
        st.download_button(
            label="Download Temperature Data",
            data=temp_csv,
            file_name="temperature_data.csv",
            mime="text/csv"
        )
    
    with col2:
        if not co2_df.empty:
            co2_csv = co2_df.to_csv()
            st.download_button(
                label="Download CO₂ Data",
                data=co2_csv,
                file_name="co2_data.csv",
                mime="text/csv"
            )

def _show_temperature_tab(data, results):
    """Show temperature tab content."""
    st.header("Temperature Forecasting")
    
    if results is None:
        st.info("ℹ️ Run the model to see temperature forecasting results.")
        return
    
    # Get temperature data
    temp_df = data['temperature']
    
    # Get the target variable (Global or first column)
    if 'Global' in temp_df.columns:
        temp_var = 'Global'
    else:
        temp_var = temp_df.columns[0]
    
    # Get results data
    forecast_dates = results["forecast_dates"]
    temp_mean = results["temperature_mean"]
    temp_std = results["temperature_std"]
    temp_lower = results["temperature_lower"]
    temp_upper = results["temperature_upper"]
    model_type = results["model_type"]
    horizon = results["horizon"]
    
    # Temperature forecast plot
    st.subheader("Temperature Anomaly Forecast")
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=temp_df.index[-60:],  # Last 5 years (60 months)
        y=temp_df[temp_var].values[-60:],
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=temp_mean,
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add confidence intervals if available
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=temp_upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=temp_lower,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(width=0),
            name='95% Confidence Interval'
        ))
    
    # Add vertical line at forecast start
    fig.add_vline(
        x=temp_df.index[-1],
        line_dash="dash",
        line_color="black",
        annotation_text="Forecast Start",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title=f"{temp_var.replace('_', ' ')} Temperature Anomaly Forecast ({horizon} Months)",
        xaxis_title="Year",
        yaxis_title="Temperature Anomaly (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast metrics
    st.subheader("Forecast Metrics")
    
    # Calculate metrics
    last_temp = temp_df[temp_var].values[-1]
    mean_forecast = np.mean(temp_mean)
    end_forecast = temp_mean[-1]
    warming_rate = (end_forecast - last_temp) * (12 / horizon)  # Annualized rate
    
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        mean_uncertainty = np.mean(temp_std) * 2  # 2 standard deviations (95% CI)
    else:
        mean_uncertainty = 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Anomaly",
            value=f"{last_temp:.2f}°C"
        )
    
    with col2:
        st.metric(
            label="Mean Forecast",
            value=f"{mean_forecast:.2f}°C",
            delta=f"{mean_forecast - last_temp:.2f}°C"
        )
    
    with col3:
        st.metric(
            label="End of Forecast",
            value=f"{end_forecast:.2f}°C",
            delta=f"{end_forecast - last_temp:.2f}°C"
        )
    
    with col4:
        st.metric(
            label="Uncertainty (±2σ)",
            value=f"{mean_uncertainty:.2f}°C"
        )
    
    # Warming rate
    st.metric(
        label="Annualized Warming Rate",
        value=f"{warming_rate:.2f}°C/year",
        delta=f"{warming_rate:.2f}°C/year"
    )
    
    # Probability analysis
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        st.subheader("Probability Analysis")
        
        # Define thresholds
        thresholds = [1.5, 2.0, 2.5]
        
        # Calculate probabilities
        probabilities = []
        
        for threshold in thresholds:
            # Probability that any month exceeds threshold
            probs = []
            
            for i in range(len(temp_mean)):
                # Calculate probability using normal CDF
                prob = 1 - stats.norm.cdf(threshold, loc=temp_mean[i], scale=temp_std[i])
                probs.append(prob)
            
            # Probability of exceeding threshold at least once
            prob_any = 1 - np.prod(1 - np.array(probs))
            
            probabilities.append({
                "threshold": threshold,
                "probability": prob_any * 100  # Convert to percentage
            })
        
        # Display probabilities
        col1, col2, col3 = st.columns(3)
        
        for i, col in enumerate([col1, col2, col3]):
            with col:
                st.metric(
                    label=f"Prob. of Exceeding {thresholds[i]}°C",
                    value=f"{probabilities[i]['probability']:.1f}%"
                )
        
        # Create probability distribution plot
        st.subheader("Forecast Distribution")
        
        # Create figure
        fig = go.Figure()
        
        # Plot normal distributions for selected forecast points
        x = np.linspace(-1, 5, 1000)
        
        # Beginning, middle, and end of forecast
        indices = [0, horizon//2, horizon-1]
        labels = ["First Month", "Middle", "Last Month"]
        colors = ['blue', 'green', 'red']
        
        for i, idx in enumerate(indices):
            # Calculate normal PDF
            pdf = stats.norm.pdf(x, loc=temp_mean[idx], scale=temp_std[idx])
            
            # Add distribution curve
            fig.add_trace(go.Scatter(
                x=x,
                y=pdf,
                mode='lines',
                name=f"{labels[i]} ({forecast_dates[idx].strftime('%b %Y')})",
                line=dict(color=colors[i])
            ))
            
            # Add vertical line at the mean
            fig.add_vline(
                x=temp_mean[idx],
                line_dash="dash",
                line_color=colors[i],
                annotation_text=f"{temp_mean[idx]:.2f}°C",
                annotation_position="top right"
            )
        
        # Add thresholds
        for threshold in thresholds:
            fig.add_vline(
                x=threshold,
                line_dash="dot",
                line_color="black",
                annotation_text=f"{threshold}°C",
                annotation_position="bottom right"
            )
        
        # Update layout
        fig.update_layout(
            title="Forecast Probability Distributions",
            xaxis_title="Temperature Anomaly (°C)",
            yaxis_title="Probability Density",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Annual average comparison
    st.subheader("Annual Average Comparison")
    
    # Calculate annual averages
    annual_avg = temp_df[temp_var].resample('Y').mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add annual averages
    fig.add_trace(go.Bar(
        x=annual_avg.index.year,
        y=annual_avg.values,
        name='Annual Average',
        marker_color='blue'
    ))
    
    # Add horizontal line at zero
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="black"
    )
    
    # Calculate current year's average if available
    current_year = datetime.now().year
    if current_year in annual_avg.index.year:
        current_year_avg = annual_avg[annual_avg.index.year == current_year].values[0]
    else:
        # Use last available year
        current_year = annual_avg.index.year[-1]
        current_year_avg = annual_avg.values[-1]
    
    # Add horizontal line at current year's average
    fig.add_hline(
        y=current_year_avg,
        line_dash="solid",
        line_color="red",
        annotation_text=f"{current_year} Avg: {current_year_avg:.2f}°C",
        annotation_position="right"
    )
    
    # Update layout
    fig.update_layout(
        title=f"{temp_var.replace('_', ' ')} Annual Average Temperature Anomaly",
        xaxis_title="Year",
        yaxis_title="Temperature Anomaly (°C)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download forecast button
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': temp_mean
    })
    
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        forecast_df['Lower_95'] = temp_lower
        forecast_df['Upper_95'] = temp_upper
        forecast_df['Std_Dev'] = temp_std
    
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="Download Temperature Forecast",
        data=csv,
        file_name="temperature_forecast.csv",
        mime="text/csv"
    )

def _show_co2_tab(data, results):
    """Show CO2 tab content."""
    st.header("CO₂ Concentration Prediction")
    
    if results is None:
        st.info("ℹ️ Run the model to see CO₂ prediction results.")
        return
    
    # Get CO2 data
    co2_df = data['co2']
    
    # Check if CO2 data is available
    if co2_df.empty:
        st.warning("⚠️ No historical CO₂ data available. Showing forecast only.")
        
        # Create synthetic historical data for visualization
        dates = pd.date_range(end=results["forecast_dates"][0] - pd.DateOffset(months=1), periods=60, freq='MS')
        co2_values = np.linspace(380, 420, 60) + np.random.normal(0, 1, 60)
        co2_df = pd.DataFrame({
            'CO2_Concentration': co2_values
        }, index=dates)
    
    # Get the target variable (CO2_Concentration or first column)
    if 'CO2_Concentration' in co2_df.columns:
        co2_var = 'CO2_Concentration'
    else:
        co2_var = co2_df.columns[0]
    
    # Get results data
    forecast_dates = results["forecast_dates"]
    co2_mean = results["co2_mean"]
    co2_std = results["co2_std"]
    co2_lower = results["co2_lower"]
    co2_upper = results["co2_upper"]
    model_type = results["model_type"]
    horizon = results["horizon"]
    
    # CO2 forecast plot
    st.subheader("CO₂ Concentration Forecast")
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=co2_df.index[-60:],  # Last 5 years (60 months)
        y=co2_df[co2_var].values[-60:],
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=co2_mean,
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add confidence intervals if available
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=co2_upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=co2_lower,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(width=0),
            name='95% Confidence Interval'
        ))
    
    # Add vertical line at forecast start
    fig.add_vline(
        x=co2_df.index[-1],
        line_dash="dash",
        line_color="black",
        annotation_text="Forecast Start",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title=f"CO₂ Concentration Forecast ({horizon} Months)",
        xaxis_title="Year",
        yaxis_title="CO₂ (ppm)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast metrics
    st.subheader("CO₂ Forecast Metrics")
    
    # Calculate metrics
    last_co2 = co2_df[co2_var].values[-1]
    mean_forecast = np.mean(co2_mean)
    end_forecast = co2_mean[-1]
    growth_rate = (end_forecast - last_co2) * (12 / horizon)  # Annualized rate
    
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        mean_uncertainty = np.mean(co2_std) * 2  # 2 standard deviations (95% CI)
    else:
        mean_uncertainty = 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current CO₂",
            value=f"{last_co2:.1f} ppm"
        )
    
    with col2:
        st.metric(
            label="Mean Forecast",
            value=f"{mean_forecast:.1f} ppm",
            delta=f"{mean_forecast - last_co2:.1f} ppm"
        )
    
    with col3:
        st.metric(
            label="End of Forecast",
            value=f"{end_forecast:.1f} ppm",
            delta=f"{end_forecast - last_co2:.1f} ppm"
        )
    
    with col4:
        st.metric(
            label="Uncertainty (±2σ)",
            value=f"{mean_uncertainty:.1f} ppm"
        )
    
    # Growth rate
    st.metric(
        label="Annualized Growth Rate",
        value=f"{growth_rate:.1f} ppm/year",
        delta=f"{growth_rate:.1f} ppm/year"
    )
    
    # CO2 threshold analysis
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        st.subheader("Threshold Analysis")
        
        # Define thresholds
        thresholds = [420, 425, 430]
        
        # Calculate probabilities
        probabilities = []
        
        for threshold in thresholds:
            # Probability that any month exceeds threshold
            probs = []
            
            for i in range(len(co2_mean)):
                # Calculate probability using normal CDF
                prob = 1 - stats.norm.cdf(threshold, loc=co2_mean[i], scale=co2_std[i])
                probs.append(prob)
            
            # Probability of exceeding threshold at least once
            prob_any = 1 - np.prod(1 - np.array(probs))
            
            # Calculate time to threshold
            time_to_threshold = None
            for i in range(len(co2_mean)):
                if co2_mean[i] >= threshold:
                    time_to_threshold = i + 1
                    break
            
            probabilities.append({
                "threshold": threshold,
                "probability": prob_any * 100,  # Convert to percentage
                "time_to_threshold": time_to_threshold
            })
        
        # Display probabilities
        st.subheader("Probability of Exceeding Thresholds")
        
        probs_df = pd.DataFrame(probabilities)
        probs_df['time_to_threshold'] = probs_df['time_to_threshold'].apply(
            lambda x: f"{x} months" if x is not None else "Not within forecast horizon"
        )
        probs_df.columns = ['Threshold (ppm)', 'Probability (%)', 'Time to Threshold']
        st.table(probs_df)
    
    # Historical CO2 trend
    st.subheader("Long-term CO₂ Trend")
    
    # Calculate annual averages
    annual_avg = co2_df[co2_var].resample('Y').mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add annual averages
    fig.add_trace(go.Scatter(
        x=annual_avg.index,
        y=annual_avg.values,
        mode='lines+markers',
        name='Annual Average',
        marker_color='blue'
    ))
    
    # Add trend line
    years = np.array([(date - annual_avg.index[0]).days / 365.25 for date in annual_avg.index])
    z = np.polyfit(years, annual_avg.values, 1)
    p = np.poly1d(z)
    
    trend_values = p(years)
    
    fig.add_trace(go.Scatter(
        x=annual_avg.index,
        y=trend_values,
        mode='lines',
        name=f'Trend: {z[0]:.2f} ppm/year',
        line=dict(color='red', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title="Long-term CO₂ Concentration Trend",
        xaxis_title="Year",
        yaxis_title="CO₂ (ppm)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly variation
    st.subheader("Seasonal CO₂ Variation")
    
    # Calculate monthly averages
    monthly_avg = co2_df[co2_var].groupby(co2_df.index.month).mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add monthly averages
    fig.add_trace(go.Bar(
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=monthly_avg.values,
        name='Monthly Average',
        marker_color='green'
    ))
    
    # Update layout
    fig.update_layout(
        title="Seasonal CO₂ Variation",
        xaxis_title="Month",
        yaxis_title="CO₂ (ppm)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download forecast button
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': co2_mean
    })
    
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        forecast_df['Lower_95'] = co2_lower
        forecast_df['Upper_95'] = co2_upper
        forecast_df['Std_Dev'] = co2_std
    
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="Download CO₂ Forecast",
        data=csv,
        file_name="co2_forecast.csv",
        mime="text/csv"
    )

def _show_spatial_tab(data, results):
    """Show spatial analysis tab content."""
    st.header("Spatial Climate Analysis")
    
    # Check if spatial data is available
    if 'spatial' not in data or not data['spatial']:
        st.warning("⚠️ No spatial climate data available for analysis.")
        return
    
    # Get spatial data
    spatial_data = data['spatial']
    
    # Temporal evolution
    st.subheader("Temporal Evolution of Spatial Patterns")
    
    # Year selector
    years = [item['year'] for item in spatial_data]
    
    # Create comparison mode
    comparison_mode = st.radio(
        "Display Mode",
        ["Single Year", "Compare Years"],
        horizontal=True,
        key="spatial_comparison_mode"
    )
    
    if comparison_mode == "Single Year":
        # Single year selection
        selected_year = st.selectbox(
            "Select Year",
            years,
            index=len(years) - 1,  # Default to last year
            key="spatial_single_year"
        )
        
        # Get data for selected year
        selected_data = next((item for item in spatial_data if item['year'] == selected_year), None)
        
        if selected_data:
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=selected_data['data'],
                x=selected_data['lon'],
                y=selected_data['lat'],
                colorscale='RdBu_r',
                colorbar=dict(title='Temperature<br>Anomaly (°C)'),
                zmin=-2,
                zmax=2
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Temperature Anomaly Map ({selected_year})",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True, key="spatial_single_year_map")
    else:
        # Compare years
        col1, col2 = st.columns(2)
        
        with col1:
            year1 = st.selectbox(
                "First Year",
                years,
                index=0,  # Default to first year
                key="spatial_year1"
            )
        
        with col2:
            year2 = st.selectbox(
                "Second Year",
                years,
                index=len(years) - 1,  # Default to last year
                key="spatial_year2"
            )
        
        # Get data for selected years
        data1 = next((item for item in spatial_data if item['year'] == year1), None)
        data2 = next((item for item in spatial_data if item['year'] == year2), None)
        
        if data1 and data2:
            # Calculate difference
            diff_data = data2['data'] - data1['data']
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    f"{year1} Temperature Anomaly",
                    f"{year2} Temperature Anomaly",
                    f"Difference ({year2} - {year1})",
                    "Latitudinal Averages"
                ),
                specs=[
                    [{"type": "heatmap"}, {"type": "heatmap"}],
                    [{"type": "heatmap"}, {"type": "scatter"}]
                ]
            )
            
            # Add first year heatmap
            fig.add_trace(
                go.Heatmap(
                    z=data1['data'],
                    x=data1['lon'],
                    y=data1['lat'],
                    colorscale='RdBu_r',
                    zmin=-2,
                    zmax=2,
                    showscale=True,
                    colorbar=dict(title='°C', x=0.46, y=0.8, len=0.4)
                ),
                row=1, col=1
            )
            
            # Add second year heatmap
            fig.add_trace(
                go.Heatmap(
                    z=data2['data'],
                    x=data2['lon'],
                    y=data2['lat'],
                    colorscale='RdBu_r',
                    zmin=-2,
                    zmax=2,
                    showscale=True,
                    colorbar=dict(title='°C', x=0.96, y=0.8, len=0.4)
                ),
                row=1, col=2
            )
            
            # Add difference heatmap
            fig.add_trace(
                go.Heatmap(
                    z=diff_data,
                    x=data1['lon'],
                    y=data1['lat'],
                    colorscale='RdBu_r',
                    zmin=-2,
                    zmax=2,
                    showscale=True,
                    colorbar=dict(title='°C', x=0.46, y=0.2, len=0.4)
                ),
                row=2, col=1
            )
            
            # Calculate latitudinal averages
            lat_avg1 = np.mean(data1['data'], axis=1)
            lat_avg2 = np.mean(data2['data'], axis=1)
            
            # Add latitudinal averages
            fig.add_trace(
                go.Scatter(
                    x=data1['lat'],
                    y=lat_avg1,
                    mode='lines',
                    name=str(year1),
                    line=dict(color='blue')
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data2['lat'],
                    y=lat_avg2,
                    mode='lines',
                    name=str(year2),
                    line=dict(color='red')
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                title_text=f"Temperature Anomaly Comparison: {year1} vs {year2}"
            )
            
            # Update axes
            fig.update_xaxes(title_text="Longitude", row=1, col=1)
            fig.update_xaxes(title_text="Longitude", row=1, col=2)
            fig.update_xaxes(title_text="Longitude", row=2, col=1)
            fig.update_xaxes(title_text="Latitude", row=2, col=2)
            
            fig.update_yaxes(title_text="Latitude", row=1, col=1)
            fig.update_yaxes(title_text="Latitude", row=1, col=2)
            fig.update_yaxes(title_text="Latitude", row=2, col=1)
            fig.update_yaxes(title_text="Temperature Anomaly (°C)", row=2, col=2)
            
            st.plotly_chart(fig, use_container_width=True, key="spatial_comparison_maps")
    
    # Polar amplification
    st.subheader("Polar Amplification Analysis")
    
    # Create figure for polar amplification
    fig = go.Figure()
    
    # Calculate latitudinal averages for each year
    for item in spatial_data:
        year = item['year']
        lat = item['lat']
        
        # Calculate latitudinal average
        lat_avg = np.mean(item['data'], axis=1)
        
        # Add line for this year
        fig.add_trace(go.Scatter(
            x=lat,
            y=lat_avg,
            mode='lines',
            name=str(year)
        ))
    
    # Update layout
    fig.update_layout(
        title="Latitudinal Temperature Anomaly Profiles",
        xaxis_title="Latitude",
        yaxis_title="Temperature Anomaly (°C)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate polar amplification metric
    st.subheader("Polar Amplification Metric")
    
    # Calculate ratio of polar (>60°N) to tropical (±30°) warming for each year
    pa_metrics = []
    
    for item in spatial_data:
        year = item['year']
        lat = item['lat']
        data_grid = item['data']
        
        # Get indices for polar and tropical regions
        polar_indices = [i for i, l in enumerate(lat) if abs(l) > 60]
        tropical_indices = [i for i, l in enumerate(lat) if abs(l) <= 30]
        
        # Calculate average warming in each region
        if polar_indices and tropical_indices:
            polar_warming = np.mean(data_grid[polar_indices, :])
            tropical_warming = np.mean(data_grid[tropical_indices, :])
            
            # Calculate ratio if tropical warming is not zero
            if abs(tropical_warming) > 1e-6:
                pa_ratio = polar_warming / tropical_warming
            else:
                pa_ratio = np.nan
            
            pa_metrics.append({
                "Year": year,
                "Polar Warming": polar_warming,
                "Tropical Warming": tropical_warming,
                "PA Ratio": pa_ratio
            })
    
    # Create dataframe
    pa_df = pd.DataFrame(pa_metrics)
    
    # Display as table
    st.table(pa_df)
    
    # Create plot of PA ratio over time
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pa_df["Year"],
        y=pa_df["PA Ratio"],
        mode='lines+markers',
        name='Polar Amplification Ratio',
        line=dict(color='purple')
    ))
    
    # Add horizontal line at ratio=1 (no amplification)
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="black"
    )
    
    # Update layout
    fig.update_layout(
        title="Polar Amplification Ratio Over Time",
        xaxis_title="Year",
        yaxis_title="Polar/Tropical Warming Ratio",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Land vs. Ocean warming
    st.subheader("Land vs. Ocean Warming")
    
    # Check if we have separate land and ocean data
    if 'Land' in data['temperature'].columns and 'Ocean' in data['temperature'].columns:
        # Get land and ocean data
        land_data = data['temperature']['Land']
        ocean_data = data['temperature']['Ocean']
        
        # Create annual averages
        land_annual = land_data.resample('Y').mean()
        ocean_annual = ocean_data.resample('Y').mean()
        
        # Create figure
        fig = go.Figure()
        
        # Add land data
        fig.add_trace(go.Scatter(
            x=land_annual.index,
            y=land_annual.values,
            mode='lines',
            name='Land',
            line=dict(color='brown')
        ))
        
        # Add ocean data
        fig.add_trace(go.Scatter(
            x=ocean_annual.index,
            y=ocean_annual.values,
            mode='lines',
            name='Ocean',
            line=dict(color='blue')
        ))
        
        # Add land/ocean ratio
        ratio = land_annual / ocean_annual
        
        fig.add_trace(go.Scatter(
            x=ratio.index,
            y=ratio.values,
            mode='lines',
            name='Land/Ocean Ratio',
            line=dict(color='purple', dash='dash'),
            yaxis="y2"
        ))
        
        # Update layout with secondary y-axis
        fig.update_layout(
            title="Land vs. Ocean Warming Comparison",
            xaxis_title="Year",
            yaxis_title="Temperature Anomaly (°C)",
            yaxis2=dict(
                title="Land/Ocean Ratio",
                overlaying="y",
                side="right"
            ),
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Separate Land and Ocean data not available for comparison.")
    
    # Download spatial data button
    if len(spatial_data) > 0:
        # Create a download button for the most recent spatial data
        latest_spatial = spatial_data[-1]
        
        # Convert to CSV format
        spatial_csv = f"# Spatial Temperature Anomaly Data for {latest_spatial['year']}\n"
        spatial_csv += f"# Latitude: {latest_spatial['lat'].tolist()}\n"
        spatial_csv += f"# Longitude: {latest_spatial['lon'].tolist()}\n\n"
        
        # Add data grid
        for i, lat_val in enumerate(latest_spatial['lat']):
            for j, lon_val in enumerate(latest_spatial['lon']):
                spatial_csv += f"{lat_val},{lon_val},{latest_spatial['data'][i, j]}\n"
        
        st.download_button(
            label="Download Spatial Data (CSV)",
            data=spatial_csv,
            file_name=f"spatial_temperature_{latest_spatial['year']}.csv",
            mime="text/csv"
        )

def _show_demo_mode():
    """Show a demo version when ClimateAdapter is not available."""
    st.markdown("""
    ### Demo Mode - Basic Climate Analysis
    
    Advanced ML models are not available, but you can still:
    - Upload your own climate data
    - View basic statistical analysis
    - Create interactive visualizations
    """)
    
    # Data source selection
    data_source = st.radio(
        "Choose data source:",
        ["Sample Data", "Upload CSV"],
        horizontal=True,
        key="climate_demo_data_source"
    )
    
    df = None
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file with climate data", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    else:  # Sample Data
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        temp = 15 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) + np.random.randn(len(dates)) * 2
        
        df = pd.DataFrame({
            'Date': dates,
            'Temperature': temp
        })
    
    if df is not None:
        # Display data info
        st.subheader("Data Overview")
        st.metric("Rows", len(df))
        st.metric("Columns", len(df.columns))
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Simple visualization
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            st.subheader("Data Visualization")
            
            # Create simple plot
            if 'Date' in df.columns:
                fig = px.line(df, x='Date', y=numeric_cols[0], title=f'{numeric_cols[0]} Over Time')
            else:
                fig = px.line(df, y=numeric_cols[0], title=f'{numeric_cols[0]} Over Time')
                
            st.plotly_chart(fig, use_container_width=True, key="climate_demo_chart")
            
            # Basic statistics
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
        else:
            st.warning("No numeric columns found for analysis")
    
    st.info("For advanced climate modeling with uncertainty quantification, install the full model dependencies.")

# Main function to test the component independently
if __name__ == "__main__":
    show_climate_page()