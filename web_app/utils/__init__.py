"""
Utilities for the multi-modal AI web application.

This package contains utility functions and tools for the web application,
including session state management, download utilities, and other helpers.
"""

from .session_state import (
    get_state, set_state, reset_state, save_state, load_state,
    get_domain_data, get_domain_predictions,
    store_finance_data, store_climate_data,
    store_finance_predictions, store_climate_predictions
)

from .download import (
    download_button, create_download_section
)

__all__ = [
    'get_state', 'set_state', 'reset_state', 'save_state', 'load_state',
    'get_domain_data', 'get_domain_predictions',
    'store_finance_data', 'store_climate_data',
    'store_finance_predictions', 'store_climate_predictions',
    'download_button', 'create_download_section'
]