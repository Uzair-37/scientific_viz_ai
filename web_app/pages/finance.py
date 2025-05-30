"""
Finance page component for the multi-modal AI web application.

This module implements the finance page UI that allows users to
analyze financial time series data with uncertainty quantification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import torch
from io import BytesIO
import base64

# Import domain-specific components
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from models.domain_adapters.finance_adapter import FinanceAdapter
except ImportError:
    # Fallback for different deployment environments
    FinanceAdapter = None

from web_app.utils.download import download_button

def show_finance_page():
    """Display the finance page with time series forecasting and uncertainty visualization."""
    # Main title
    st.title("Financial Forecasting with Uncertainty Quantification")
    
    # Check if FinanceAdapter is available
    if FinanceAdapter is None:
        st.error("FinanceAdapter module is not available. Please check the installation.")
        st.info("Using demo mode with limited functionality.")
        _show_demo_mode()
        return
    
    # Create tabs for different finance sections
    tabs = st.tabs(["Data", "Forecasting", "Volatility", "Risk Analysis"])
    
    # Check if model needs to be run
    run_model = st.session_state.get("run_model", False)
    
    # Get data source
    data_source = st.session_state.get("data_source", "sample")
    
    # Get or generate data based on source
    if data_source == "sample":
        data = _get_sample_data()
    elif data_source == "yahoo_finance_api":
        data = _get_yahoo_finance_data()
    elif data_source == "upload_data":
        data = _get_uploaded_data()
    else:
        data = _get_sample_data()  # Fallback to sample data
    
    # Store data in session state
    st.session_state.finance_data = data
    
    # Run model if triggered
    if run_model:
        with st.spinner("Running financial model with uncertainty quantification..."):
            results = _run_finance_model(data)
            st.session_state.results = results
            st.session_state.run_model = False  # Reset flag
    
    # Get results from session state
    results = st.session_state.get("results", None)
    
    # Data Tab
    with tabs[0]:
        _show_finance_data_tab(data)
    
    # Forecasting Tab
    with tabs[1]:
        _show_finance_forecasting_tab(data, results)
    
    # Volatility Tab
    with tabs[2]:
        _show_finance_volatility_tab(data, results)
    
    # Risk Analysis Tab
    with tabs[3]:
        _show_finance_risk_tab(data, results)

def _get_sample_data():
    """Get sample finance data for demonstration."""
    # Define start and end date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years of data
    
    # Define tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    try:
        # Try to fetch actual data (if internet connection available)
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        # Use only Adj Close for simplicity
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Adj Close']
        
        # Return with success message
        st.success("✅ Sample data loaded successfully")
        return data
    
    except Exception as e:
        # Generate synthetic data if fetch fails
        st.warning(f"⚠️ Could not fetch real data, using synthetic data instead: {str(e)}")
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Create synthetic data
        np.random.seed(42)
        
        # Start with random values
        synth_data = {ticker: 100 + np.random.randn() * 10 for ticker in tickers}
        
        # Generate random walk
        data_dict = {ticker: [] for ticker in tickers}
        for _ in range(len(date_range)):
            for ticker in tickers:
                # Random walk with drift and volatility
                synth_data[ticker] *= np.exp(np.random.normal(0.0002, 0.02))
                data_dict[ticker].append(synth_data[ticker])
        
        # Create DataFrame
        df = pd.DataFrame(data_dict, index=date_range)
        return df

def _get_yahoo_finance_data():
    """Get data from Yahoo Finance API based on user selection."""
    # Get ticker from session state
    ticker = st.session_state.get("ticker", "AAPL")
    
    # Get date range from session state
    start_date = st.session_state.get("start_date", datetime.now() - timedelta(days=365))
    end_date = st.session_state.get("end_date", datetime.now())
    
    try:
        # Fetch data from Yahoo Finance
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        # Success message
        st.success(f"✅ Data for {ticker} loaded successfully")
        return data
    
    except Exception as e:
        # Error message
        st.error(f"❌ Error fetching data: {str(e)}")
        return _get_sample_data()  # Fallback to sample data

def _get_uploaded_data():
    """Process uploaded financial data."""
    # Get uploaded file from session state
    uploaded_file = st.session_state.get("uploaded_file", None)
    
    if uploaded_file is not None:
        try:
            # Read data based on file type
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            else:  # Excel
                data = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
            
            # Success message
            st.success(f"✅ Uploaded data processed successfully")
            return data
        
        except Exception as e:
            # Error message
            st.error(f"❌ Error processing uploaded file: {str(e)}")
            return _get_sample_data()  # Fallback to sample data
    
    else:
        # No file uploaded
        st.warning("⚠️ No file uploaded, using sample data instead")
        return _get_sample_data()  # Fallback to sample data

def _prepare_finance_features(data):
    """Prepare features for financial forecasting."""
    # Convert to DataFrame if Series
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    # Create copy to avoid modifying original
    df = data.copy()
    
    # Handle MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Adj Close']
    
    # Calculate returns
    returns = df.pct_change().fillna(0)
    
    # Calculate log returns
    log_returns = np.log(df/df.shift(1)).fillna(0)
    
    # Add technical indicators if enabled
    if st.session_state.get("use_technical_indicators", True):
        # Moving averages
        for ma_window in [5, 10, 20, 50]:
            df[f'MA_{ma_window}'] = df.rolling(window=ma_window).mean()
            
            # Calculate MA ratio (current price / MA)
            for col in df.columns[:len(data.columns)]:  # Only for original price columns
                df[f'MA_{ma_window}_ratio_{col}'] = df[col] / df[f'MA_{ma_window}'].iloc[:, df.columns.get_loc(col)]
        
        # Volatility (standard deviation)
        for vol_window in [10, 20]:
            returns_cols = returns.columns
            for col in returns_cols:
                df[f'Volatility_{vol_window}_{col}'] = returns[col].rolling(window=vol_window).std()
        
        # RSI (Relative Strength Index)
        for rsi_window in [14]:
            delta = df.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window=rsi_window).mean()
            avg_loss = loss.rolling(window=rsi_window).mean()
            
            rs = avg_gain / avg_loss
            for col in df.columns[:len(data.columns)]:
                rs_col = rs.iloc[:, df.columns.get_loc(col)]
                df[f'RSI_{rsi_window}_{col}'] = 100 - (100 / (1 + rs_col))
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    # Return both dataframes
    return df, returns, log_returns

def _run_finance_model(data):
    """Run finance model with uncertainty quantification."""
    # Get model type
    model_type = st.session_state.get("model_type", "heteroscedastic")
    
    # Prepare features
    features_df, returns_df, log_returns_df = _prepare_finance_features(data)
    
    # Define forecast horizon
    horizon = st.session_state.get("finance_horizon", 5)
    
    # Create sequences for time series forecasting
    sequence_length = 20  # Use last 20 days to predict next 'horizon' days
    
    # Prepare sequences
    X, y = _create_sequences(log_returns_df, sequence_length, horizon)
    
    # Split into train and test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Define model parameters
    input_dim = X_train.shape[2]  # Number of features
    output_dim = y_train.shape[2]  # Number of assets
    embed_dim = st.session_state.get("embedding_dim", 256)
    hidden_dim = st.session_state.get("hidden_dim", 128)
    dropout_rate = st.session_state.get("dropout_rate", 0.1)
    activation = st.session_state.get("activation", "relu")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinanceAdapter(
        time_series_dim=input_dim,
        fundamental_dim=0,  # No fundamental data for now
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        horizon=horizon,
        uncertainty_type=model_type,
        dropout_rate=dropout_rate,
        activation=activation
    ).to(device)
    
    # For demo purposes, we'll just make predictions without training
    # In a real implementation, you would train the model here
    
    # Prepare test input
    test_input = {
        "time_series": X_test_tensor.to(device)
    }
    
    # Make predictions with uncertainty
    with torch.no_grad():
        predictions = model.predict_with_uncertainty(test_input)
    
    # Process predictions
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        returns_mean = predictions["returns_mean"].cpu().numpy()
        returns_var = predictions["returns_var"].cpu().numpy()
        volatility_mean = predictions["volatility_mean"].cpu().numpy()
        volatility_var = predictions["volatility_var"].cpu().numpy()
    else:
        # No uncertainty model
        returns_mean = predictions["returns"].cpu().numpy()
        returns_var = np.zeros_like(returns_mean)
        volatility_mean = predictions["volatility"].cpu().numpy()
        volatility_var = np.zeros_like(volatility_mean)
    
    # Convert log returns predictions back to prices
    last_prices = data.iloc[-sequence_length:].values
    predicted_prices = []
    predicted_lower = []
    predicted_upper = []
    
    # For each asset
    for i in range(output_dim):
        # Get last price
        last_price = last_prices[-1, i]
        
        # Calculate predicted prices
        pred_returns = returns_mean[-1, :, i]
        pred_std = np.sqrt(returns_var[-1, :, i])
        
        # Calculate price series
        price_series = [last_price]
        lower_series = [last_price]
        upper_series = [last_price]
        
        for j in range(horizon):
            # Calculate next price based on predicted return
            next_price = price_series[-1] * (1 + pred_returns[j])
            price_series.append(next_price)
            
            # Calculate confidence intervals
            lower_bound = price_series[-1] * (1 + pred_returns[j] - 2 * pred_std[j])
            upper_bound = price_series[-1] * (1 + pred_returns[j] + 2 * pred_std[j])
            lower_series.append(lower_bound)
            upper_series.append(upper_bound)
        
        # Store predicted price series
        predicted_prices.append(price_series[1:])  # Skip first value (last actual price)
        predicted_lower.append(lower_series[1:])
        predicted_upper.append(upper_series[1:])
    
    # Create forecast dates
    last_date = data.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq='B'  # Business days
    )
    
    # Create results dictionary
    results = {
        "input_data": data,
        "features": features_df,
        "returns": returns_df,
        "log_returns": log_returns_df,
        "forecast_dates": forecast_dates,
        "predicted_prices": predicted_prices,
        "predicted_lower": predicted_lower,
        "predicted_upper": predicted_upper,
        "predicted_returns": returns_mean[-1],  # Last batch, all horizons, all assets
        "predicted_returns_var": returns_var[-1],
        "predicted_volatility": volatility_mean[-1],
        "predicted_volatility_var": volatility_var[-1],
        "model_type": model_type,
        "horizon": horizon,
        "asset_names": data.columns
    }
    
    return results

def _create_sequences(data, sequence_length, horizon):
    """Create sequences for time series forecasting."""
    X, y = [], []
    
    for i in range(len(data) - sequence_length - horizon + 1):
        X.append(data.iloc[i:i+sequence_length].values)
        y.append(data.iloc[i+sequence_length:i+sequence_length+horizon].values)
    
    return np.array(X), np.array(y)

def _show_finance_data_tab(data):
    """Show finance data tab content."""
    st.header("Financial Data")
    
    # Data overview
    st.subheader("Data Overview")
    st.dataframe(data.head())
    
    # Data statistics
    st.subheader("Data Statistics")
    st.dataframe(data.describe())
    
    # Time series plot
    st.subheader("Price Time Series")
    
    # Create figure
    fig = go.Figure()
    
    # Add time series for each asset
    for col in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[col],
            mode='lines',
            name=col
        ))
    
    # Update layout
    fig.update_layout(
        title="Asset Price Time Series",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Returns plot
    st.subheader("Daily Returns")
    
    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Create figure
    fig = go.Figure()
    
    # Add time series for each asset
    for col in returns.columns:
        fig.add_trace(go.Scatter(
            x=returns.index,
            y=returns[col],
            mode='lines',
            name=col
        ))
    
    # Update layout
    fig.update_layout(
        title="Daily Returns",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.subheader("Asset Correlation Matrix")
    
    # Calculate correlation matrix
    corr_matrix = data.pct_change().corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Asset Return Correlation Matrix"
    )
    
    # Update layout
    fig.update_layout(
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download data button
    csv = data.to_csv()
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="finance_data.csv",
        mime="text/csv"
    )

def _show_finance_forecasting_tab(data, results):
    """Show finance forecasting tab content."""
    st.header("Return Forecasting")
    
    if results is None:
        st.info("ℹ️ Run the model to see forecasting results.")
        return
    
    # Get results data
    asset_names = results["asset_names"]
    forecast_dates = results["forecast_dates"]
    predicted_prices = results["predicted_prices"]
    predicted_lower = results["predicted_lower"]
    predicted_upper = results["predicted_upper"]
    model_type = results["model_type"]
    horizon = results["horizon"]
    
    # Asset selector
    asset_index = st.selectbox(
        "Select Asset",
        range(len(asset_names)),
        format_func=lambda i: asset_names[i]
    )
    
    # Get selected asset data
    asset_data = data[asset_names[asset_index]]
    asset_pred_prices = predicted_prices[asset_index]
    asset_pred_lower = predicted_lower[asset_index]
    asset_pred_upper = predicted_upper[asset_index]
    
    # Forecast plot
    st.subheader(f"{asset_names[asset_index]} Price Forecast")
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=asset_data.index[-30:],  # Last 30 days
        y=asset_data.values[-30:],
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=asset_pred_prices,
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add confidence intervals if available
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=asset_pred_upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=asset_pred_lower,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(width=0),
            name='95% Confidence Interval'
        ))
    
    # Add vertical line at forecast start
    fig.add_vline(
        x=asset_data.index[-1],
        line_dash="dash",
        line_color="black",
        annotation_text="Forecast Start",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title=f"{asset_names[asset_index]} Price Forecast ({horizon} Days)",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast metrics
    st.subheader("Forecast Metrics")
    
    # Calculate metrics
    last_price = asset_data.values[-1]
    expected_return = (asset_pred_prices[-1] / last_price - 1) * 100  # Percentage
    
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        uncertainty = (asset_pred_upper[-1] - asset_pred_lower[-1]) / (2 * asset_pred_prices[-1]) * 100  # Percentage
    else:
        uncertainty = 0
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Expected Return",
            value=f"{expected_return:.2f}%",
            delta=f"{expected_return:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Forecast Horizon",
            value=f"{horizon} days"
        )
    
    with col3:
        st.metric(
            label="Uncertainty (±2σ)",
            value=f"{uncertainty:.2f}%"
        )
    
    # Return distribution
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        st.subheader("Return Distribution")
        
        # Get predicted returns and uncertainty
        returns_mean = results["predicted_returns"][:, asset_index]  # All horizons for selected asset
        returns_std = np.sqrt(results["predicted_returns_var"][:, asset_index])
        
        # Create figure
        fig = go.Figure()
        
        # Add bar chart for returns
        fig.add_trace(go.Bar(
            x=[f"Day {i+1}" for i in range(horizon)],
            y=returns_mean * 100,  # Convert to percentage
            name="Expected Return (%)",
            error_y=dict(
                type='data',
                array=returns_std * 100 * 2,  # 2 standard deviations (95% CI)
                visible=True
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{asset_names[asset_index]} Expected Daily Returns with Uncertainty",
            xaxis_title="Forecast Horizon",
            yaxis_title="Return (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Download forecast button
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': asset_pred_prices
    })
    
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        forecast_df['Lower_95'] = asset_pred_lower
        forecast_df['Upper_95'] = asset_pred_upper
    
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="Download Forecast as CSV",
        data=csv,
        file_name=f"{asset_names[asset_index]}_forecast.csv",
        mime="text/csv"
    )

def _show_finance_volatility_tab(data, results):
    """Show finance volatility tab content."""
    st.header("Volatility Analysis")
    
    if results is None:
        st.info("ℹ️ Run the model to see volatility results.")
        return
    
    # Get results data
    asset_names = results["asset_names"]
    forecast_dates = results["forecast_dates"]
    predicted_volatility = results["predicted_volatility"]
    predicted_volatility_var = results["predicted_volatility_var"]
    model_type = results["model_type"]
    horizon = results["horizon"]
    
    # Asset selector
    asset_index = st.selectbox(
        "Select Asset",
        range(len(asset_names)),
        format_func=lambda i: asset_names[i],
        key="volatility_asset_selector"
    )
    
    # Calculate historical volatility
    returns = results["returns"][asset_names[asset_index]]
    historical_volatility = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
    
    # Get selected asset volatility predictions
    asset_pred_vol = predicted_volatility[:, asset_index] * np.sqrt(252)  # Annualized
    
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        asset_pred_vol_std = np.sqrt(predicted_volatility_var[:, asset_index]) * np.sqrt(252)  # Annualized
        asset_pred_vol_lower = asset_pred_vol - 2 * asset_pred_vol_std
        asset_pred_vol_upper = asset_pred_vol + 2 * asset_pred_vol_std
        
        # Ensure lower bound is not negative
        asset_pred_vol_lower = np.maximum(asset_pred_vol_lower, 0)
    
    # Historical volatility plot
    st.subheader("Historical Volatility")
    
    # Create figure
    fig = go.Figure()
    
    # Add historical volatility
    fig.add_trace(go.Scatter(
        x=historical_volatility.index,
        y=historical_volatility.values * 100,  # Convert to percentage
        mode='lines',
        name='20-Day Rolling Volatility',
        line=dict(color='blue')
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{asset_names[asset_index]} Historical Volatility (Annualized)",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volatility forecast plot
    st.subheader("Volatility Forecast")
    
    # Create figure
    fig = go.Figure()
    
    # Add current volatility point
    current_vol = historical_volatility.iloc[-1] * 100  # Convert to percentage
    fig.add_trace(go.Scatter(
        x=[historical_volatility.index[-1]],
        y=[current_vol],
        mode='markers',
        name='Current Volatility',
        marker=dict(color='blue', size=10)
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=asset_pred_vol * 100,  # Convert to percentage
        mode='lines+markers',
        name='Volatility Forecast',
        line=dict(color='red')
    ))
    
    # Add confidence intervals if available
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=asset_pred_vol_upper * 100,  # Convert to percentage
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=asset_pred_vol_lower * 100,  # Convert to percentage
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(width=0),
            name='95% Confidence Interval'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{asset_names[asset_index]} Volatility Forecast (Annualized)",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volatility metrics
    st.subheader("Volatility Metrics")
    
    # Calculate metrics
    current_volatility = historical_volatility.iloc[-1] * 100  # Percentage
    forecast_volatility = asset_pred_vol[-1] * 100  # Last horizon, percentage
    
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        volatility_uncertainty = asset_pred_vol_std[-1] * 100  # Last horizon, percentage
    else:
        volatility_uncertainty = 0
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Current Volatility",
            value=f"{current_volatility:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Forecast Volatility",
            value=f"{forecast_volatility:.2f}%",
            delta=f"{forecast_volatility - current_volatility:.2f}%"
        )
    
    with col3:
        st.metric(
            label="Uncertainty (±2σ)",
            value=f"{volatility_uncertainty * 2:.2f}%"
        )
    
    # Volatility term structure
    st.subheader("Volatility Term Structure")
    
    # Create figure
    fig = go.Figure()
    
    # Add volatility term structure
    fig.add_trace(go.Scatter(
        x=[f"Day {i+1}" for i in range(horizon)],
        y=asset_pred_vol * 100,  # Convert to percentage
        mode='lines+markers',
        name='Volatility',
        line=dict(color='red')
    ))
    
    # Add confidence intervals if available
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        fig.add_trace(go.Scatter(
            x=[f"Day {i+1}" for i in range(horizon)],
            y=asset_pred_vol_upper * 100,  # Convert to percentage
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[f"Day {i+1}" for i in range(horizon)],
            y=asset_pred_vol_lower * 100,  # Convert to percentage
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(width=0),
            name='95% Confidence Interval'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{asset_names[asset_index]} Volatility Term Structure",
        xaxis_title="Forecast Horizon",
        yaxis_title="Volatility (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download volatility forecast button
    vol_forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Volatility': asset_pred_vol * 100  # Convert to percentage
    })
    
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        vol_forecast_df['Lower_95'] = asset_pred_vol_lower * 100  # Convert to percentage
        vol_forecast_df['Upper_95'] = asset_pred_vol_upper * 100  # Convert to percentage
    
    csv = vol_forecast_df.to_csv(index=False)
    st.download_button(
        label="Download Volatility Forecast as CSV",
        data=csv,
        file_name=f"{asset_names[asset_index]}_volatility_forecast.csv",
        mime="text/csv"
    )

def _show_finance_risk_tab(data, results):
    """Show finance risk analysis tab content."""
    st.header("Risk Analysis")
    
    if results is None:
        st.info("ℹ️ Run the model to see risk analysis results.")
        return
    
    # Get results data
    asset_names = results["asset_names"]
    forecast_dates = results["forecast_dates"]
    predicted_returns = results["predicted_returns"]
    predicted_returns_var = results["predicted_returns_var"]
    predicted_volatility = results["predicted_volatility"]
    model_type = results["model_type"]
    horizon = results["horizon"]
    
    # Portfolio allocation
    st.subheader("Portfolio Allocation")
    
    # Create columns for portfolio weights
    cols = st.columns(min(4, len(asset_names)))
    
    # Initialize weights
    weights = {}
    total_weight = 0
    
    # Add sliders for each asset
    for i, col in enumerate(cols):
        if i < len(asset_names):
            weight = col.slider(
                f"{asset_names[i]} Weight",
                min_value=0.0,
                max_value=1.0,
                value=1.0/len(asset_names),
                step=0.01,
                key=f"weight_{i}"
            )
            weights[asset_names[i]] = weight
            total_weight += weight
    
    # Normalize weights
    if total_weight > 0:
        for asset in weights:
            weights[asset] /= total_weight
    
    # Display normalized weights
    st.subheader("Normalized Portfolio Weights")
    
    # Create pie chart
    fig = px.pie(
        values=list(weights.values()),
        names=list(weights.keys()),
        title="Portfolio Allocation"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate portfolio metrics
    st.subheader("Portfolio Risk Metrics")
    
    # Create weight vector
    weight_vector = np.array([weights.get(asset, 0) for asset in asset_names])
    
    # Calculate historical portfolio returns
    returns = results["returns"]
    portfolio_returns = returns.dot(weight_vector)
    
    # Historical statistics
    annual_return = portfolio_returns.mean() * 252 * 100  # Annualized, percentage
    annual_volatility = portfolio_returns.std() * np.sqrt(252) * 100  # Annualized, percentage
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    max_drawdown = (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min() * 100  # Percentage
    
    # Display historical metrics
    st.markdown("### Historical Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Annual Return",
            value=f"{annual_return:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Annual Volatility",
            value=f"{annual_volatility:.2f}%"
        )
    
    with col3:
        st.metric(
            label="Sharpe Ratio",
            value=f"{sharpe_ratio:.2f}"
        )
    
    with col4:
        st.metric(
            label="Max Drawdown",
            value=f"{max_drawdown:.2f}%"
        )
    
    # Calculate forecast portfolio returns
    forecast_returns = np.sum(predicted_returns * weight_vector, axis=1)
    
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        # Simplified variance calculation (ignoring covariance for demo)
        forecast_variance = np.sum((predicted_returns_var * (weight_vector ** 2)), axis=1)
        forecast_std = np.sqrt(forecast_variance)
        
        # Calculate VaR and CVaR
        confidence = 0.95
        z_score = 1.645  # 95% confidence
        
        # Value at Risk (VaR)
        var_95 = -(forecast_returns.mean() + z_score * forecast_std.mean()) * 100  # Percentage
        
        # Conditional Value at Risk (CVaR) approximation
        cvar_95 = -(forecast_returns.mean() + forecast_std.mean() * 
                   (np.exp(-0.5 * z_score**2) / (np.sqrt(2 * np.pi) * (1 - confidence)))) * 100  # Percentage
    else:
        # Simple estimates without uncertainty
        forecast_std = np.sqrt(np.sum((predicted_volatility ** 2) * (weight_vector ** 2), axis=1))
        var_95 = -(forecast_returns.mean() + 1.645 * forecast_std.mean()) * 100  # Percentage
        cvar_95 = -(forecast_returns.mean() + 2.0 * forecast_std.mean()) * 100  # Percentage
    
    # Display forecast metrics
    st.markdown("### Forecast Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Expected Return",
            value=f"{forecast_returns.mean() * 100:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Expected Volatility",
            value=f"{forecast_std.mean() * 100:.2f}%"
        )
    
    with col3:
        st.metric(
            label="Value at Risk (95%)",
            value=f"{var_95:.2f}%"
        )
    
    with col4:
        st.metric(
            label="Conditional VaR (95%)",
            value=f"{cvar_95:.2f}%"
        )
    
    # Return distribution plot
    st.subheader("Portfolio Return Distribution")
    
    # Create normal distribution based on forecast mean and variance
    if model_type in ["heteroscedastic", "mc_dropout", "bayesian", "ensemble"]:
        # Calculate mean and standard deviation
        mean_return = forecast_returns.mean() * 100  # Percentage
        std_return = forecast_std.mean() * 100  # Percentage
        
        # Create distribution
        x = np.linspace(mean_return - 4 * std_return, mean_return + 4 * std_return, 1000)
        pdf = stats.norm.pdf(x, mean_return, std_return)
        
        # Create figure
        fig = go.Figure()
        
        # Add distribution curve
        fig.add_trace(go.Scatter(
            x=x,
            y=pdf,
            mode='lines',
            name='Return Distribution',
            line=dict(color='blue')
        ))
        
        # Add VaR line
        fig.add_trace(go.Scatter(
            x=[-var_95, -var_95],
            y=[0, stats.norm.pdf(-var_95, mean_return, std_return)],
            mode='lines',
            name='95% VaR',
            line=dict(color='red', dash='dash')
        ))
        
        # Add CVaR region
        x_cvar = np.linspace(mean_return - 4 * std_return, -var_95, 100)
        y_cvar = stats.norm.pdf(x_cvar, mean_return, std_return)
        
        fig.add_trace(go.Scatter(
            x=x_cvar,
            y=y_cvar,
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(width=0),
            name='Expected Shortfall'
        ))
        
        # Update layout
        fig.update_layout(
            title="Portfolio Return Distribution with Risk Measures",
            xaxis_title="Return (%)",
            yaxis_title="Probability Density",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Stress test
    st.subheader("Stress Test")
    
    # Stress scenarios
    scenarios = {
        "Market Crash (-20%)": -0.20,
        "Severe Correction (-10%)": -0.10,
        "Moderate Correction (-5%)": -0.05,
        "Flat Market (0%)": 0.0,
        "Bull Market (+5%)": 0.05,
        "Strong Bull Market (+10%)": 0.10
    }
    
    # Create stress test results
    stress_results = []
    
    for scenario, market_return in scenarios.items():
        # Calculate beta for each asset (simplified)
        betas = np.ones(len(asset_names))  # Assume beta=1 for demo
        
        # Calculate expected returns under scenario
        scenario_returns = betas * market_return
        
        # Calculate portfolio return
        portfolio_return = np.sum(scenario_returns * weight_vector) * 100  # Percentage
        
        # Store results
        stress_results.append({
            "Scenario": scenario,
            "Market Return": f"{market_return * 100:.1f}%",
            "Portfolio Return": f"{portfolio_return:.2f}%"
        })
    
    # Display stress test results
    st.table(pd.DataFrame(stress_results))
    
    # Download risk report button
    report_data = f"""# Portfolio Risk Report

## Portfolio Allocation
{', '.join([f"{asset}: {weight:.1%}" for asset, weight in weights.items()])}

## Historical Performance
- Annual Return: {annual_return:.2f}%
- Annual Volatility: {annual_volatility:.2f}%
- Sharpe Ratio: {sharpe_ratio:.2f}
- Maximum Drawdown: {max_drawdown:.2f}%

## Risk Forecast
- Expected Return: {forecast_returns.mean() * 100:.2f}%
- Expected Volatility: {forecast_std.mean() * 100:.2f}%
- Value at Risk (95%): {var_95:.2f}%
- Conditional VaR (95%): {cvar_95:.2f}%

## Stress Test Results
{pd.DataFrame(stress_results).to_markdown(index=False)}

Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}
"""
    
    st.download_button(
        label="Download Risk Report",
        data=report_data,
        file_name="portfolio_risk_report.md",
        mime="text/markdown"
    )

def _show_demo_mode():
    """Show a demo version when FinanceAdapter is not available."""
    st.markdown("""
    ### Demo Mode - Basic Financial Analysis
    
    Advanced ML models are not available, but you can still:
    - Upload your own financial data
    - View basic statistical analysis
    - Create interactive visualizations
    - Download analysis results
    """)
    
    # Data source selection
    data_source = st.radio(
        "Choose data source:",
        ["Sample Data", "Upload CSV", "Yahoo Finance (Basic)"],
        horizontal=True
    )
    
    df = None
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
    
    elif data_source == "Yahoo Finance (Basic)":
        symbol = st.text_input("Enter stock symbol:", value="AAPL")
        if st.button("Fetch Data"):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="1y")
                df = df.reset_index()
                st.success(f"Data fetched for {symbol}")
            except Exception as e:
                st.error(f"Error fetching data: {e}")
    
    else:  # Sample Data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices
        })
    
    if df is not None:
        # Display data info
        st.subheader("Data Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Rows", len(df))
            st.metric("Columns", len(df.columns))
        
        with col2:
            if 'Close' in df.columns:
                st.metric("Latest Price", f"${df['Close'].iloc[-1]:.2f}")
                returns = df['Close'].pct_change().dropna()
                st.metric("Daily Volatility", f"{returns.std():.3f}")
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Basic visualization
        st.subheader("Price Chart")
        if 'Date' in df.columns and 'Close' in df.columns:
            fig = px.line(df, x='Date', y='Close', title='Price Over Time')
        elif 'Close' in df.columns:
            fig = px.line(df, y='Close', title='Price Over Time')
        else:
            # Try to plot first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig = px.line(df, y=numeric_cols[0], title=f'{numeric_cols[0]} Over Time')
            else:
                st.warning("No numeric columns found for plotting")
                return
                
        st.plotly_chart(fig, use_container_width=True)
        
        # Basic statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    st.info("For advanced forecasting with uncertainty quantification, install the full model dependencies.")

# Main function to test the component independently
if __name__ == "__main__":
    show_finance_page()