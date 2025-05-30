"""
Home page component for the multi-modal AI web application.

This module implements the home page UI that introduces the system
and allows users to select their domain of interest.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO

def show_home_page():
    """Display the home page with introduction and domain selection."""
    # Main title
    st.title("Multi-Modal AI with Uncertainty Quantification")
    
    # Introduction
    st.markdown("""
    This application demonstrates advanced AI techniques for financial and climate data analysis
    with robust uncertainty quantification. Select a domain to explore state-of-the-art machine
    learning models for prediction and pattern discovery.
    """)
    
    # Domain selection cards in a two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        _show_finance_card()
    
    with col2:
        _show_climate_card()
    
    # Horizontal rule
    st.markdown("---")
    
    # Key features section
    st.header("Key Features")
    
    # Features in a three-column layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Advanced Uncertainty
        - Heteroscedastic models
        - MC Dropout estimation
        - Bayesian neural networks
        - Ensemble methods
        - Calibrated intervals
        """)
    
    with col2:
        st.markdown("""
        ### 🧠 Multi-Modal Fusion
        - Time series processing
        - Spatial-temporal analysis
        - Cross-attention mechanism
        - Domain-specific adapters
        - Transformer architecture
        """)
    
    with col3:
        st.markdown("""
        ### 🔍 Pattern Discovery
        - Dimensionality reduction
        - Cluster analysis
        - Anomaly detection
        - Correlation discovery
        - Interactive exploration
        """)
    
    # Horizontal rule
    st.markdown("---")
    
    # Technical overview section
    with st.expander("Technical Overview", expanded=False):
        _show_technical_overview()
    
    # Usage guide
    with st.expander("Usage Guide", expanded=False):
        _show_usage_guide()
    
    # Project information
    with st.expander("About This Project", expanded=False):
        _show_project_info()

def _show_finance_card():
    """Display the finance domain card."""
    # Create a styled container with CSS
    st.markdown("""
    <div style="
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        height: 100%;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0;
            right: 0;
            width: 40px;
            height: 40px;
            background-color: #0068c9;
            border-radius: 0 0 0 10px;
        ">
            <div style="
                color: white;
                font-size: 20px;
                text-align: center;
                line-height: 40px;
            ">📈</div>
        </div>
        <h2 style="margin-top: 0; color: #0068c9;">Finance</h2>
        <p style="margin-bottom: 1rem;">
            Financial time series forecasting with uncertainty quantification for stocks, 
            cryptocurrencies, and economic indicators.
        </p>
        <ul style="margin-bottom: 1.5rem;">
            <li>Stock return prediction</li>
            <li>Volatility forecasting</li>
            <li>Risk assessment</li>
            <li>Anomaly detection</li>
        </ul>
        <div style="text-align: center;">
            <button style="
                background-color: #0068c9;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                transition: background-color 0.2s;
            " onclick="document.querySelector('[data-baseweb=tab]').click();">
                Explore Finance
            </button>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a clickable area to select finance domain
    if st.button("Select Finance Domain", key="select_finance"):
        st.session_state.domain = "finance"
        st.session_state.page = "finance"
        st.experimental_rerun()

def _show_climate_card():
    """Display the climate domain card."""
    # Create a styled container with CSS
    st.markdown("""
    <div style="
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        height: 100%;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0;
            right: 0;
            width: 40px;
            height: 40px;
            background-color: #2e7d32;
            border-radius: 0 0 0 10px;
        ">
            <div style="
                color: white;
                font-size: 20px;
                text-align: center;
                line-height: 40px;
            ">🌍</div>
        </div>
        <h2 style="margin-top: 0; color: #2e7d32;">Climate</h2>
        <p style="margin-bottom: 1rem;">
            Climate and environmental data analysis with uncertainty quantification for
            temperature, CO₂, and spatial-temporal patterns.
        </p>
        <ul style="margin-bottom: 1.5rem;">
            <li>Temperature forecasting</li>
            <li>CO₂ concentration prediction</li>
            <li>Spatial pattern analysis</li>
            <li>Climate trend modeling</li>
        </ul>
        <div style="text-align: center;">
            <button style="
                background-color: #2e7d32;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                transition: background-color 0.2s;
            " onclick="document.querySelectorAll('[data-baseweb=tab]')[2].click();">
                Explore Climate
            </button>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a clickable area to select climate domain
    if st.button("Select Climate Domain", key="select_climate"):
        st.session_state.domain = "climate"
        st.session_state.page = "climate"
        st.experimental_rerun()

def _show_technical_overview():
    """Display technical overview of the system architecture."""
    st.subheader("System Architecture")
    
    # Architecture description
    st.markdown("""
    This application is built on a modular architecture with several key components:
    
    1. **Multi-Modal Transformer**: Processes and aligns different types of data (time series, tabular, text)
    using specialized encoders and cross-attention mechanisms.
    
    2. **Domain-Specific Adapters**: Tailored components for finance and climate domains that
    interface with the core transformer architecture.
    
    3. **Uncertainty Quantification**: Multiple methods to estimate prediction uncertainty:
        - **Heteroscedastic Model**: Outputs both mean and variance
        - **MC Dropout**: Uses dropout during inference for Bayesian approximation
        - **Bayesian Neural Network**: Places distributions over weights
        - **Ensemble Model**: Combines multiple models for robust estimates
    
    4. **Pattern Discovery**: Tools for dimensionality reduction, clustering, and anomaly detection
    to uncover hidden patterns in the data.
    
    5. **Interactive Visualization**: Rich visualization components with uncertainty representation.
    """)
    
    # Architecture diagram
    st.markdown("""
    ```
                           ┌────────────────────┐
                           │    Data Sources    │
                           │ (Finance, Climate) │
                           └─────────┬──────────┘
                                     │
                                     ▼
    ┌────────────────────────────────────────────────────────────┐
    │                       Data Processing                       │
    │                                                            │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
    │  │ Finance Loader  │  │ Climate Loader  │  │ Processors  │ │
    │  └─────────────────┘  └─────────────────┘  └─────────────┘ │
    └────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌────────────────────────────────────────────────────────────┐
    │                   Multi-Modal Transformer                   │
    │                                                            │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
    │  │  Time Series    │  │     Tabular     │  │    Text     │ │
    │  │    Encoder      │  │     Encoder     │  │   Encoder   │ │
    │  └─────────────────┘  └─────────────────┘  └─────────────┘ │
    │                                                            │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │              Cross-Attention Fusion                 │   │
    │  └─────────────────────────────────────────────────────┘   │
    └────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌────────────────────────────────────────────────────────────┐
    │                    Domain Adapters                          │
    │                                                            │
    │  ┌─────────────────┐         ┌─────────────────────────┐   │
    │  │ Finance Adapter │         │    Climate Adapter      │   │
    │  └─────────────────┘         └─────────────────────────┘   │
    └────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌────────────────────────────────────────────────────────────┐
    │                 Uncertainty Quantification                  │
    │                                                            │
    │  ┌───────────────┐ ┌──────────────┐ ┌───────────────────┐  │
    │  │Heteroscedastic│ │  MC Dropout  │ │   Bayesian NN     │  │
    │  └───────────────┘ └──────────────┘ └───────────────────┘  │
    └────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌────────────────────────────────────────────────────────────┐
    │                   Analysis & Visualization                  │
    │                                                            │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
    │  │  Time Series    │  │  Uncertainty    │  │  Pattern    │ │
    │  │  Visualization  │  │  Visualization  │  │  Discovery  │ │
    │  └─────────────────┘  └─────────────────┘  └─────────────┘ │
    └────────────────────────────────────────────────────────────┘
    ```
    """)

def _show_usage_guide():
    """Display a guide on how to use the application."""
    st.subheader("How to Use This Application")
    
    st.markdown("""
    ### Step 1: Select a Domain
    Choose between Finance and Climate domains to access specialized models and visualizations.
    
    ### Step 2: Configure Model Settings
    Use the sidebar to:
    - Select an uncertainty quantification method
    - Choose a data source (upload, sample, or API)
    - Configure advanced model parameters
    
    ### Step 3: Run Analysis
    Click the "Run Model" button to start the analysis process.
    
    ### Step 4: Explore Results
    - View forecasts with confidence intervals
    - Examine uncertainty calibration
    - Discover patterns in the data
    - Compare different model types
    
    ### Step 5: Export Results
    Download predictions, visualizations, and reports for further use.
    """)
    
    # Add a simple illustration
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Create a simple flow diagram
        st.markdown("""
        ```
        ┌───────────┐    ┌───────────┐    ┌───────────┐
        │  Select   │ → │ Configure │ → │   Run     │
        │  Domain   │    │  Model    │    │  Analysis │
        └───────────┘    └───────────┘    └───────────┘
               │                               │
               └───────────┐     ┌─────────────┘
                           ▼     ▼
                      ┌───────────────┐
                      │    Explore    │
                      │    Results    │
                      └───────────────┘
                             │
                             ▼
                      ┌───────────────┐
                      │    Export     │
                      │    Data       │
                      └───────────────┘
        ```
        """)

def _show_project_info():
    """Display information about the project."""
    st.subheader("About This Project")
    
    st.markdown("""
    This project demonstrates advanced machine learning techniques for financial and climate data
    analysis with robust uncertainty quantification. It was developed as part of an MS admissions
    application to showcase expertise in:
    
    - Multi-modal machine learning architectures
    - Advanced uncertainty quantification methods
    - Domain-specific modeling for finance and climate science
    - Interactive data visualization and exploration
    
    ### Technologies Used
    
    - **PyTorch**: Deep learning models and uncertainty quantification
    - **Streamlit**: Interactive web application
    - **Plotly**: Advanced data visualization
    - **Pandas & NumPy**: Data processing and analysis
    - **yfinance & xarray**: Domain-specific data sources
    
    ### Data Sources
    
    - **Finance**: Yahoo Finance, FRED Economic Data
    - **Climate**: NOAA, Berkeley Earth, Climate Reanalyzer
    
    ### References
    
    The methods implemented in this project are based on research from:
    
    1. Kendall, A., & Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*
    2. Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*
    3. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*
    """)
    
    # Author information
    st.markdown("""
    ### Author
    
    **Uzair Shaik**  
    MS Admissions Project, 2025
    """)

# Main function to test the component independently
if __name__ == "__main__":
    show_home_page()