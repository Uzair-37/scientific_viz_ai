"""
Main Streamlit application for Multi-Modal AI for Finance and Climate Science.

This module implements the web interface for the multi-modal AI system
with uncertainty quantification, supporting both finance and climate domains.
"""

import os
import sys
import logging
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import torch

# Add parent directory to path to allow importing from other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import domain-specific components
from web_app.components.sidebar import create_sidebar
from web_app.pages.home import show_home_page
from web_app.pages.finance import show_finance_page
from web_app.pages.climate import show_climate_page
from web_app.pages.analysis import show_analysis_page
from web_app.pages.diagnostics import show_diagnostics_page

# Import model components
try:
    from Models.domain_adapters.finance_adapter import FinanceAdapter
    from Models.domain_adapters.climate_adapter import ClimateAdapter
except ImportError:
    FinanceAdapter = None
    ClimateAdapter = None

try:
    from utils.config import get_config, save_config
except ImportError:
    def get_config():
        return {}
    def save_config(config):
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Define app constants
APP_TITLE = "Multi-Modal AI for Finance and Climate Science"
APP_ICON = "🧠"
PRIMARY_COLOR = "#0068c9"
SECONDARY_COLOR = "#5c5c5c"
ACCENT_COLOR = "#ff5757"

def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

def setup_session_state():
    """Initialize session state variables if they don't exist."""
    # Domain selection
    if "domain" not in st.session_state:
        st.session_state.domain = "finance"  # Default domain
    
    # Active page
    if "page" not in st.session_state:
        st.session_state.page = "home"  # Default page
    
    # Model settings
    if "model_type" not in st.session_state:
        st.session_state.model_type = "heteroscedastic"  # Default model type
    
    # Data settings
    if "data_source" not in st.session_state:
        st.session_state.data_source = "sample"  # Default data source
    
    # Results storage
    if "results" not in st.session_state:
        st.session_state.results = None

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
        /* Primary color adjustments */
        .main .block-container {
            padding-top: 2rem;
        }
        .st-emotion-cache-16txtl3 h1, 
        .st-emotion-cache-16txtl3 h2, 
        .st-emotion-cache-16txtl3 h3 {
            color: #0068c9;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #0068c9;
            color: white;
            border-radius: 4px;
            padding: 0.25rem 1rem;
            border: none;
        }
        .stButton > button:hover {
            background-color: #0051a8;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0 0;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1rem;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #0068c9;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] [data-testid="stMarkdownContainer"] p {
            color: white;
        }
        
        /* Tooltip styling */
        .tooltip {
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted #0068c9;
            cursor: help;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 300px;
            background-color: #555;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -150px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.9rem;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Card styling */
        .card {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Metric styling */
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            text-align: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #0068c9;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #5c5c5c;
        }
        
        /* Progress bar */
        .stProgress .st-cl {
            background-color: #0068c9;
        }
    </style>
    """, unsafe_allow_html=True)

def create_tooltip(text, explanation):
    """Create a tooltip with an explanation."""
    return f"""
    <span class="tooltip">{text}
        <span class="tooltiptext">{explanation}</span>
    </span>
    """

def create_navigation():
    """Create the main navigation tabs."""
    tabs = st.tabs([
        "🏠 Home", 
        "📈 Finance", 
        "🌍 Climate", 
        "🔍 Analysis", 
        "📊 Diagnostics"
    ])
    
    # Set active page if clicked on a tab
    if "tab_clicked" not in st.session_state:
        st.session_state.tab_clicked = False
    
    # Map tab indices to page names
    tab_to_page = {
        0: "home",
        1: "finance",
        2: "climate",
        3: "analysis",
        4: "diagnostics"
    }
    
    return tabs

def create_domain_specific_model(domain, model_type, model_params):
    """Create domain-specific model based on user selections."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if domain == "finance":
            # Create finance adapter
            model = FinanceAdapter(
                time_series_dim=model_params.get("time_series_dim", 10),
                fundamental_dim=model_params.get("fundamental_dim", 5),
                embed_dim=model_params.get("embed_dim", 256),
                hidden_dim=model_params.get("hidden_dim", 128),
                output_dim=model_params.get("output_dim", 1),
                horizon=model_params.get("horizon", 5),
                uncertainty_type=model_type,
                dropout_rate=model_params.get("dropout_rate", 0.1),
                activation=model_params.get("activation", "relu")
            )
        elif domain == "climate":
            # Create climate adapter
            model = ClimateAdapter(
                input_channels=model_params.get("input_channels", 5),
                time_series_dim=model_params.get("time_series_dim", 10),
                metadata_dim=model_params.get("metadata_dim", 8),
                spatial_dim=model_params.get("spatial_dim", (32, 32)),
                temporal_dim=model_params.get("temporal_dim", 48),
                embed_dim=model_params.get("embed_dim", 256),
                hidden_dim=model_params.get("hidden_dim", 128),
                output_dim=model_params.get("output_dim", 1),
                horizon=model_params.get("horizon", 12),
                uncertainty_type=model_type,
                dropout_rate=model_params.get("dropout_rate", 0.1),
                activation=model_params.get("activation", "relu"),
                data_type=model_params.get("data_type", "gridded")
            )
        else:
            raise ValueError(f"Unsupported domain: {domain}")
        
        # Move model to device
        model = model.to(device)
        
        return model
    
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        st.error(f"Error creating model: {e}")
        return None

def main():
    """Main function to run the Streamlit app."""
    # Setup page config
    setup_page_config()
    
    # Setup session state
    setup_session_state()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Create sidebar
    create_sidebar()
    
    # Create main navigation
    tabs = create_navigation()
    
    # Get current page
    current_page = st.session_state.page
    
    # Show appropriate page based on selected tab
    if current_page == "home":
        with tabs[0]:  # Home tab
            show_home_page()
    elif current_page == "finance":
        with tabs[1]:  # Finance tab
            show_finance_page()
    elif current_page == "climate":
        with tabs[2]:  # Climate tab
            show_climate_page()
    elif current_page == "analysis":
        with tabs[3]:  # Analysis tab
            show_analysis_page()
    elif current_page == "diagnostics":
        with tabs[4]:  # Diagnostics tab
            show_diagnostics_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Multi-Modal AI for Finance and Climate Science with Uncertainty Quantification** | "
        "Created by Uzair Shaik | MS Admissions Project"
    )

if __name__ == "__main__":
    main()