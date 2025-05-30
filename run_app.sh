#!/bin/bash

# Script to run the Multi-Modal AI web application
# This automatically activates the virtual environment if it exists

# Set the application title
APP_TITLE="Multi-Modal AI for Finance and Climate Science"

# Directory where the virtual environment is located
VENV_DIR="venv"

# Function to check if the virtual environment exists and activate it
activate_venv() {
    if [ -d "$VENV_DIR" ]; then
        echo "Activating virtual environment..."
        source "$VENV_DIR/bin/activate"
    else
        echo "Virtual environment not found. Please create it first:"
        echo "python -m venv venv"
        echo "source venv/bin/activate"
        echo "pip install -r requirements.txt"
        exit 1
    fi
}

# Function to check if streamlit is installed
check_streamlit() {
    if ! command -v streamlit &> /dev/null; then
        echo "Streamlit not found. Please install it:"
        echo "pip install streamlit"
        exit 1
    fi
}

# Function to start the application
start_app() {
    echo "Starting $APP_TITLE..."
    echo "==============================================="
    echo "Access the application at: http://localhost:8501"
    echo "Press Ctrl+C to stop the application"
    echo "==============================================="
    
    # Run the Streamlit application
    streamlit run web_app/app.py
}

# Main execution
echo "==================================================="
echo "  $APP_TITLE"
echo "==================================================="

# Activate virtual environment if it exists
activate_venv

# Check if streamlit is installed
check_streamlit

# Start the application
start_app