"""
Utility functions and classes for the Multi-Modal AI for Finance and Climate Science.
"""
from .config import config_manager, get_api_key, setup_api_key_prompt, get_config, save_config
from .metrics import (
    calculate_regression_metrics, calculate_calibration_metrics,
    calculate_finance_metrics, calculate_climate_metrics
)