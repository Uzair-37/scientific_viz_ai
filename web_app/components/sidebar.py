"""
Sidebar component for the multi-modal AI web application.

This module implements the sidebar UI elements that control domain selection,
model settings, data sources, and other application parameters.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Import utility functions with error handling
try:
    from web_app.utils.session_state import get_state, set_state
except ImportError:
    def get_state(key, default=None):
        return st.session_state.get(key, default)
    def set_state(key, value):
        st.session_state[key] = value

try:
    from web_app.utils.download import download_button
except ImportError:
    def download_button(data, file_name, label="Download", mime="text/plain"):
        st.download_button(label=label, data=data, file_name=file_name, mime=mime)

def create_sidebar():
    """Create the sidebar with domain selection and model settings."""
    with st.sidebar:
        st.title("Model Settings")
        
        # Navigation menu
        st.header("Navigation")
        pages = {
            "Home": "home",
            "Finance": "finance",
            "Climate": "climate",
            "Analysis": "analysis",
            "Diagnostics": "diagnostics"
        }
        
        selected_page = st.radio(
            "Page",
            list(pages.keys()),
            index=list(pages.values()).index(st.session_state.page) if st.session_state.page in pages.values() else 0,
            help="Navigate to different application pages."
        )
        
        # Update session state when page changes
        if st.session_state.page != pages[selected_page]:
            st.session_state.page = pages[selected_page]
        
        # Domain selection
        st.header("Domain Selection")
        domain = st.radio(
            "Choose Domain",
            ["Finance", "Climate"],
            index=0 if st.session_state.domain == "finance" else 1,
            help="Select the domain for data analysis and prediction."
        )
        
        # Update session state when domain changes
        if (domain == "Finance" and st.session_state.domain != "finance") or \
           (domain == "Climate" and st.session_state.domain != "climate"):
            st.session_state.domain = domain.lower()
            # Reset model-specific settings when domain changes
            st.session_state.model_type = "heteroscedastic"  # Default to heteroscedastic
        
        # Model type selection
        st.header("Model Settings")
        
        model_types = {
            "No Uncertainty": "none",
            "Heteroscedastic": "heteroscedastic",
            "MC Dropout": "mc_dropout",
            "Bayesian Neural Network": "bayesian",
            "Ensemble Model": "ensemble"
        }
        
        model_explanations = {
            "No Uncertainty": "Standard model without uncertainty quantification.",
            "Heteroscedastic": "Models data-dependent uncertainty by predicting both mean and variance.",
            "MC Dropout": "Uses dropout during inference for approximate Bayesian uncertainty.",
            "Bayesian Neural Network": "Full Bayesian treatment with distributions over weights.",
            "Ensemble Model": "Combines multiple models for robust uncertainty estimation."
        }
        
        # Show model selection with explanations
        model_type_display = st.selectbox(
            "Model Type",
            list(model_types.keys()),
            index=list(model_types.values()).index(st.session_state.model_type),
            help="Select the type of model and uncertainty quantification method."
        )
        
        # Show explanation for selected model
        st.markdown(f"<div style='font-size: 0.9rem; color: #666;'>{model_explanations[model_type_display]}</div>", unsafe_allow_html=True)
        
        # Update session state
        st.session_state.model_type = model_types[model_type_display]
        
        # Data source selection
        st.header("Data Source")
        
        # Different data sources based on domain
        if st.session_state.domain == "finance":
            data_sources = ["Upload Data", "Sample Data", "Yahoo Finance API"]
        else:  # climate
            data_sources = ["Upload Data", "Sample Data", "NOAA Data", "Berkeley Earth"]
        
        data_source = st.radio(
            "Select Data Source",
            data_sources,
            index=data_sources.index("Sample Data") if st.session_state.data_source == "sample" else 0,
            help="Choose where to get data for analysis."
        )
        
        # Update session state
        st.session_state.data_source = data_source.lower().replace(" ", "_")
        
        # Show appropriate data input based on selection
        if st.session_state.data_source == "upload_data":
            _show_data_upload()
        elif st.session_state.data_source == "yahoo_finance_api":
            _show_yahoo_finance_selector()
        elif st.session_state.data_source == "noaa_data" or st.session_state.data_source == "berkeley_earth":
            _show_climate_data_selector()
        
        # Advanced settings expandable section
        with st.expander("Advanced Settings"):
            _show_advanced_settings()
        
        # Run model button
        if st.button("Run Model", key="run_model_btn"):
            # Set flag to run the model
            st.session_state.run_model = True
            
            # Show a spinner while model runs
            with st.spinner("Running model..."):
                # Note: Actual model running happens in the main app
                pass
        
        # Reset button
        if st.button("Reset Settings", key="reset_settings_btn"):
            # Reset session state
            st.session_state.domain = "finance"
            st.session_state.model_type = "heteroscedastic"
            st.session_state.data_source = "sample"
            st.session_state.run_model = False
            st.session_state.results = None
            
            # Show success message
            st.success("Settings reset to defaults.")

def _show_data_upload():
    """Show UI for data file upload."""
    if st.session_state.domain == "finance":
        # Finance data upload
        uploaded_file = st.file_uploader(
            "Upload Financial Data (CSV)",
            type=["csv", "xlsx"],
            help="Upload time series financial data. File should contain timestamp and price/return columns."
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            
            # Show preview button
            if st.button("Preview Data"):
                try:
                    # Try to read the file
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:  # Excel
                        df = pd.read_excel(uploaded_file)
                    
                    # Show preview
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error reading file: {e}")
    else:
        # Climate data upload
        uploaded_file = st.file_uploader(
            "Upload Climate Data",
            type=["csv", "xlsx", "nc"],
            help="Upload climate data. CSV/Excel for time series or NetCDF for gridded data."
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            
            # Show preview button
            if st.button("Preview Data"):
                try:
                    # Try to read the file
                    if uploaded_file.name.endswith('.nc'):
                        st.info("NetCDF preview not available in sidebar. Data will be processed when running the model.")
                    elif uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        st.dataframe(df.head())
                    else:  # Excel
                        df = pd.read_excel(uploaded_file)
                        st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error reading file: {e}")

def _show_yahoo_finance_selector():
    """Show UI for selecting Yahoo Finance ticker and date range."""
    # Ticker input
    ticker = st.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, GOOG)."
    )
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            help="Select start date for historical data."
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            help="Select end date for historical data."
        )
    
    # Store in session state
    st.session_state.ticker = ticker
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    
    # Fetch button
    if st.button("Fetch Data"):
        try:
            # Show spinner while fetching
            with st.spinner(f"Fetching data for {ticker}..."):
                # Download data from Yahoo Finance
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                # Store data in session state
                st.session_state.finance_data = data
                
                # Show success and data preview
                st.success(f"Data for {ticker} fetched successfully!")
                st.dataframe(data.head())
        except Exception as e:
            st.error(f"Error fetching data: {e}")

def _show_climate_data_selector():
    """Show UI for selecting climate data sources and parameters."""
    # Data type selection
    data_type = st.selectbox(
        "Climate Data Type",
        ["Temperature", "Precipitation", "CO2 Concentration", "Sea Level"],
        help="Select the type of climate data to analyze."
    )
    
    # Store in session state
    st.session_state.climate_data_type = data_type
    
    # Region selection
    region = st.selectbox(
        "Region",
        ["Global", "North America", "Europe", "Asia", "Africa", "Australia", "South America"],
        help="Select geographical region for data analysis."
    )
    
    # Store in session state
    st.session_state.climate_region = region
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input(
            "Start Year",
            min_value=1900,
            max_value=2023,
            value=1990,
            help="Select start year for climate data."
        )
    with col2:
        end_year = st.number_input(
            "End Year",
            min_value=1900,
            max_value=2023,
            value=2020,
            help="Select end year for climate data."
        )
    
    # Store in session state
    st.session_state.climate_start_year = start_year
    st.session_state.climate_end_year = end_year
    
    # Fetch button
    if st.button("Fetch Climate Data"):
        # In a real implementation, this would connect to climate data APIs
        # For now, we'll simulate success with a message
        st.success(f"Climate data for {data_type} in {region} from {start_year}-{end_year} will be fetched when running the model.")

def _show_advanced_settings():
    """Show advanced model settings UI."""
    # Domain-specific advanced settings
    if st.session_state.domain == "finance":
        # Finance model settings
        st.subheader("Finance Model Parameters")
        
        # Time horizon for forecasting
        horizon = st.slider(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=30,
            value=5,
            help="Number of days to forecast into the future."
        )
        
        # Feature settings
        use_technical_indicators = st.checkbox(
            "Use Technical Indicators",
            value=True,
            help="Include technical indicators as features (e.g., MA, RSI, MACD)."
        )
        
        use_sentiment = st.checkbox(
            "Use Sentiment Analysis",
            value=False,
            help="Include news sentiment as a feature (requires additional API)."
        )
        
        # Store in session state
        st.session_state.finance_horizon = horizon
        st.session_state.use_technical_indicators = use_technical_indicators
        st.session_state.use_sentiment = use_sentiment
        
    else:
        # Climate model settings
        st.subheader("Climate Model Parameters")
        
        # Time horizon for forecasting
        horizon = st.slider(
            "Forecast Horizon (months)",
            min_value=1,
            max_value=60,
            value=12,
            help="Number of months to forecast into the future."
        )
        
        # Spatial resolution
        spatial_resolution = st.selectbox(
            "Spatial Resolution",
            ["Low (5°)", "Medium (1°)", "High (0.25°)"],
            index=1,
            help="Spatial resolution for gridded climate data."
        )
        
        # Additional features
        use_emissions_scenario = st.checkbox(
            "Include Emissions Scenarios",
            value=True,
            help="Include different CO2 emissions scenarios in the forecast."
        )
        
        # Store in session state
        st.session_state.climate_horizon = horizon
        st.session_state.spatial_resolution = spatial_resolution
        st.session_state.use_emissions_scenario = use_emissions_scenario
    
    # Common model hyperparameters
    st.subheader("Model Hyperparameters")
    
    # Model architecture
    embedding_dim = st.select_slider(
        "Embedding Dimension",
        options=[32, 64, 128, 256, 512],
        value=256,
        help="Size of the embedding dimension in the model."
    )
    
    hidden_dim = st.select_slider(
        "Hidden Dimension",
        options=[32, 64, 128, 256, 512],
        value=128,
        help="Size of the hidden layers in the model."
    )
    
    dropout_rate = st.slider(
        "Dropout Rate",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.05,
        help="Dropout rate for regularization and uncertainty estimation."
    )
    
    activation = st.selectbox(
        "Activation Function",
        ["ReLU", "GELU", "SiLU"],
        index=0,
        help="Activation function used in the model."
    )
    
    # Store in session state
    st.session_state.embedding_dim = embedding_dim
    st.session_state.hidden_dim = hidden_dim
    st.session_state.dropout_rate = dropout_rate
    st.session_state.activation = activation.lower()
    
    # Training settings
    st.subheader("Training Settings")
    
    # Only show if not using pre-trained model
    if st.session_state.data_source != "sample":
        # Number of epochs
        epochs = st.slider(
            "Training Epochs",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Number of training epochs."
        )
        
        # Batch size
        batch_size = st.select_slider(
            "Batch Size",
            options=[8, 16, 32, 64, 128],
            value=32,
            help="Batch size for training."
        )
        
        # Learning rate
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001,
            format_func=lambda x: f"{x:.4f}",
            help="Learning rate for optimizer."
        )
        
        # Store in session state
        st.session_state.epochs = epochs
        st.session_state.batch_size = batch_size
        st.session_state.learning_rate = learning_rate