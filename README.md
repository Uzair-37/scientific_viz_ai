# Multi-Modal AI for Finance and Climate Science with Uncertainty Quantification

A sophisticated machine learning system that processes financial and climate data to produce forecasts with quantified uncertainty, discover patterns, and visualize insights. This project is designed as a modular, extensible platform that demonstrates modern AI techniques for practical domain-specific applications.

## Project Overview

This system introduces a novel multi-modal architecture for finance and climate data analysis that combines:

1. **Domain Adapters**: Specialized adapters for finance and climate domains.
2. **Uncertainty Quantification**: Multiple methods to estimate and visualize prediction uncertainty.
3. **Multi-Modal Processing**: Specialized encoders for each data modality with cross-attention for alignment.
4. **Pattern Discovery**: Advanced analysis techniques to identify hidden relationships and anomalies.
5. **Interactive Interface**: Streamlit-based UI for data exploration and visualization.

## Key Features

- **Multi-Modal Transformer Architecture**: Processes different types of financial and climate data
- **Domain-Specific Adapters**: Specialized for finance and climate domains
- **Uncertainty Quantification**: Heteroscedastic, Monte Carlo Dropout, Bayesian NN, and Ensemble methods
- **Pattern Discovery Tools**: Dimensionality reduction, clustering, correlation analysis, anomaly detection
- **Interactive Web Interface**: Explore data, forecasts, and uncertainty visualizations in real-time
- **Comprehensive Diagnostics**: Model evaluation, calibration assessment, and error analysis

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scientific_viz_ai.git
cd scientific_viz_ai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Web Application

Launch the interactive Streamlit web application:

```bash
# Start the Streamlit app locally
streamlit run web_app/app.py

# Or use the convenience script
./run_app.sh
```

### Online Demo

An online demo of the application is available at:

[https://scientific-viz-ai.streamlit.app](https://scientific-viz-ai.streamlit.app)

You can also deploy your own instance by following the instructions in [DEPLOYMENT.md](DEPLOYMENT.md).

## Web Application Components

The web application is structured around five main pages:

### 1. Home Page
- System overview and architecture explanation
- Interactive architecture diagram
- Quick links to different domain pages

### 2. Finance Page
- Financial time series forecasting with uncertainty quantification
- Stock price and volatility prediction
- Interactive charts with confidence intervals
- Multiple uncertainty visualization options

### 3. Climate Page
- Temperature and CO2 forecasting with uncertainty quantification
- Spatial-temporal climate variable visualization
- Multiple timescale analysis
- Uncertainty visualization for climate predictions

### 4. Analysis Page
- Advanced pattern discovery
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Clustering (K-Means, DBSCAN)
- Correlation analysis
- Anomaly detection

### 5. Diagnostics Page
- Model performance metrics
- Uncertainty calibration analysis
- Error distribution and analysis
- Model explanation tools

## Uncertainty Quantification Methods

The system implements multiple approaches to uncertainty quantification:

1. **Heteroscedastic Uncertainty**: Direct estimation of observation-dependent variance.
2. **Monte Carlo Dropout**: Approximating Bayesian inference using dropout during inference.
3. **Bayesian Neural Networks**: Full Bayesian treatment with distributions over weights.
4. **Ensemble Methods**: Combining multiple models for robust uncertainty estimates.

## Data Sources

The application can work with multiple data sources:

### Finance
- Sample datasets (included)
- Yahoo Finance API
- User-uploaded CSV/Excel files

### Climate
- Sample datasets (included)
- NOAA climate data
- Berkeley Earth temperature data
- User-uploaded CSV/Excel/NetCDF files

## System Architecture

The full system architecture is documented in `ARCHITECTURE.md`, which includes:

- Complete component diagram
- Directory structure
- Detailed component descriptions
- Implementation status

## License

[MIT License](LICENSE)

## Acknowledgments

- This project was created by Uzair Shaik as an MS Admissions Project
- Built with Streamlit, PyTorch, Plotly, and scikit-learn