"""
Session state utility functions for the multi-modal AI web application.

This module provides helper functions to manage Streamlit session state
for storing data, model results, and UI state across different pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime

def get_state(key, default=None):
    """
    Get a value from session state with a default if not found.
    
    Parameters:
    -----------
    key : str
        The key to retrieve from session state
    default : any
        The default value to return if the key is not found
    
    Returns:
    --------
    value : any
        The value from session state or the default
    """
    if key in st.session_state:
        return st.session_state[key]
    return default

def set_state(key, value):
    """
    Set a value in session state.
    
    Parameters:
    -----------
    key : str
        The key to set in session state
    value : any
        The value to store
    """
    st.session_state[key] = value

def reset_state(keys=None):
    """
    Reset specified keys in session state to their default values.
    
    Parameters:
    -----------
    keys : list or None
        List of keys to reset. If None, reset all keys.
    """
    if keys is None:
        # Reset all keys except for 'page' and 'domain'
        preserved_keys = {'page', 'domain'}
        keys_to_reset = [k for k in st.session_state.keys() if k not in preserved_keys]
    else:
        keys_to_reset = keys
    
    # Default values for common keys
    defaults = {
        'model_type': 'heteroscedastic',
        'data_source': 'sample',
        'results': None,
        'run_model': False,
        'finance_data': None,
        'climate_data': None,
        'finance_predictions': None,
        'climate_predictions': None
    }
    
    # Reset keys
    for key in keys_to_reset:
        if key in defaults:
            st.session_state[key] = defaults[key]
        else:
            # For keys not in defaults, remove them
            if key in st.session_state:
                del st.session_state[key]

def save_state(filename=None):
    """
    Save the current session state to a file.
    
    Parameters:
    -----------
    filename : str or None
        The filename to save to. If None, generate a timestamp-based filename.
    
    Returns:
    --------
    filename : str
        The filename that was used to save the state
    """
    if filename is None:
        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_state_{timestamp}.pickle"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save state
    with open(filename, 'wb') as f:
        # Filter out non-serializable objects
        serializable_state = {}
        for key, value in st.session_state.items():
            try:
                # Test if serializable
                pickle.dumps(value)
                serializable_state[key] = value
            except:
                pass
        
        pickle.dump(serializable_state, f)
    
    return filename

def load_state(filename):
    """
    Load session state from a file.
    
    Parameters:
    -----------
    filename : str
        The filename to load from
    
    Returns:
    --------
    success : bool
        True if load was successful, False otherwise
    """
    try:
        # Load state
        with open(filename, 'rb') as f:
            loaded_state = pickle.load(f)
        
        # Update session state
        for key, value in loaded_state.items():
            st.session_state[key] = value
        
        return True
    except Exception as e:
        st.error(f"Error loading session state: {e}")
        return False

def get_domain_data():
    """
    Get the current domain's data from session state.
    
    Returns:
    --------
    data : pandas.DataFrame or None
        The data for the current domain
    """
    domain = get_state('domain', 'finance')
    
    if domain == 'finance':
        return get_state('finance_data')
    else:  # climate
        return get_state('climate_data')

def get_domain_predictions():
    """
    Get the current domain's predictions from session state.
    
    Returns:
    --------
    predictions : dict or None
        The predictions for the current domain
    """
    domain = get_state('domain', 'finance')
    
    if domain == 'finance':
        return get_state('finance_predictions')
    else:  # climate
        return get_state('climate_predictions')

def store_finance_data(data, source='sample'):
    """
    Store finance data in session state.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The finance data to store
    source : str
        The source of the data (sample, yahoo_finance, upload, etc.)
    """
    set_state('finance_data', data)
    set_state('finance_data_source', source)
    set_state('finance_data_timestamp', datetime.now())

def store_climate_data(data, source='sample'):
    """
    Store climate data in session state.
    
    Parameters:
    -----------
    data : pandas.DataFrame or dict
        The climate data to store
    source : str
        The source of the data (sample, noaa, berkeley_earth, upload, etc.)
    """
    set_state('climate_data', data)
    set_state('climate_data_source', source)
    set_state('climate_data_timestamp', datetime.now())

def store_finance_predictions(predictions):
    """
    Store finance predictions in session state.
    
    Parameters:
    -----------
    predictions : dict
        The finance predictions to store
    """
    set_state('finance_predictions', predictions)
    set_state('finance_predictions_timestamp', datetime.now())

def store_climate_predictions(predictions):
    """
    Store climate predictions in session state.
    
    Parameters:
    -----------
    predictions : dict
        The climate predictions to store
    """
    set_state('climate_predictions', predictions)
    set_state('climate_predictions_timestamp', datetime.now())