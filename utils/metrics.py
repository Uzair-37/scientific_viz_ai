"""
Evaluation metrics for the multi-modal AI system.

This module provides metrics for evaluating model performance and uncertainty
calibration for both finance and climate domains.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, accuracy_score
)

def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate regression metrics for model evaluation.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
    sample_weight : np.ndarray, optional
        Sample weights
        
    Returns:
    --------
    metrics : dict
        Dictionary of metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Flatten arrays if they have more than 1 dimension
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    
    # Calculate R^2 if possible (not appropriate for some types of data)
    try:
        r2 = r2_score(y_true, y_pred, sample_weight=sample_weight)
    except:
        r2 = np.nan
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero by adding small epsilon
    with np.errstate(divide='ignore', invalid='ignore'):
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
        
    # Return metrics as dictionary
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape)
    }
    
    return metrics

def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    confidence_levels: Optional[List[float]] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculate uncertainty calibration metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
    y_std : np.ndarray
        Predicted standard deviations (uncertainty estimates)
    confidence_levels : list of float, optional
        Confidence levels to evaluate calibration at
        
    Returns:
    --------
    metrics : dict
        Dictionary of calibration metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_std = np.asarray(y_std)
    
    # Flatten arrays if they have more than 1 dimension
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_std.ndim > 1:
        y_std = y_std.flatten()
    
    # Default confidence levels if not provided
    if confidence_levels is None:
        confidence_levels = [0.5, 0.68, 0.9, 0.95, 0.99]
    
    # Calculate calibration error at each confidence level
    calibration_errors = {}
    expected_vs_actual = {}
    
    for p in confidence_levels:
        # Calculate z-score for this confidence level
        z = np.sqrt(2) * erfinv(p)
        
        # Calculate interval width
        half_width = np.abs(z * y_std)
        
        # Calculate coverage
        in_interval = np.abs(y_true - y_pred) <= half_width
        actual_coverage = np.mean(in_interval)
        
        # Calculate calibration error
        calibration_error = np.abs(p - actual_coverage)
        
        # Store results
        calibration_errors[f"{p:.2f}"] = float(calibration_error)
        expected_vs_actual[f"{p:.2f}"] = float(actual_coverage)
    
    # Calculate mean calibration error
    mean_calibration_error = np.mean(list(calibration_errors.values()))
    
    # Calculate sharpness (mean of predicted standard deviations)
    sharpness = float(np.mean(y_std))
    
    # Calculate Negative Log Likelihood (NLL) for Gaussian predictions
    # NLL = 0.5 * log(2*pi*sigma^2) + 0.5 * (y - mu)^2 / sigma^2
    epsilon = 1e-10  # Avoid division by zero
    nll = 0.5 * np.log(2 * np.pi * (y_std**2 + epsilon)) + \
          0.5 * (y_true - y_pred)**2 / (y_std**2 + epsilon)
    mean_nll = float(np.mean(nll))
    
    # Calculate standardized error (z-score)
    z_scores = (y_true - y_pred) / (y_std + epsilon)
    
    # Return metrics as dictionary
    metrics = {
        'mean_calibration_error': mean_calibration_error,
        'calibration_errors': calibration_errors,
        'expected_vs_actual': expected_vs_actual,
        'sharpness': sharpness,
        'mean_nll': mean_nll,
        'z_scores': z_scores.tolist() if len(z_scores) < 1000 else z_scores[:1000].tolist()
    }
    
    return metrics

def erfinv(x: float) -> float:
    """
    Inverse error function.
    
    Parameters:
    -----------
    x : float
        Input value (between -1 and 1)
        
    Returns:
    --------
    y : float
        Output value
    """
    # Approximation of the inverse error function
    # Source: https://stackoverflow.com/questions/5971830/need-code-for-inverse-error-function
    
    if abs(x) > 1:
        raise ValueError("Input to erfinv must be between -1 and 1")
    
    if x == 0:
        return 0
    
    # Handle edge cases
    if x >= 1.0:
        return np.inf
    if x <= -1.0:
        return -np.inf
    
    # Coefficients in rational approximation
    a = [0.886226899, -1.645349621, 0.914624893, -0.140543331]
    b = [-2.118377725, 1.442710462, -0.329097515, 0.012229801]
    c = [-1.970840454, -1.624906493, 3.429567803, 1.641345311]
    d = [3.543889200, 1.637067800]
    
    # Get sign of x
    sign_x = np.sign(x)
    x = abs(x)
    
    # Central range
    if x <= 0.7:
        z = x * x
        num = ((a[3] * z + a[2]) * z + a[1]) * z + a[0]
        den = ((b[3] * z + b[2]) * z + b[1]) * z + b[0]
        result = x * num / den
        return sign_x * result
    
    # Tail
    if x > 0.7:
        z = np.sqrt(-np.log((1 - x) / 2))
        num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0]
        den = ((d[1] * z + d[0]) * z + 1)
        result = z - num / den
        return sign_x * result

def calculate_finance_metrics(
    returns_true: np.ndarray,
    returns_pred: np.ndarray,
    returns_std: Optional[np.ndarray] = None,
    initial_investment: float = 1.0,
    risk_free_rate: float = 0.02,
    time_period: str = 'daily'
) -> Dict[str, Any]:
    """
    Calculate finance-specific metrics for model evaluation.
    
    Parameters:
    -----------
    returns_true : np.ndarray
        Ground truth returns
    returns_pred : np.ndarray
        Predicted returns
    returns_std : np.ndarray, optional
        Predicted standard deviations (uncertainty estimates)
    initial_investment : float
        Initial investment amount
    risk_free_rate : float
        Annual risk-free rate
    time_period : str
        Time period of returns ('daily', 'weekly', 'monthly', 'quarterly', 'annual')
        
    Returns:
    --------
    metrics : dict
        Dictionary of finance metrics
    """
    # Ensure inputs are numpy arrays
    returns_true = np.asarray(returns_true)
    returns_pred = np.asarray(returns_pred)
    
    # Flatten arrays if they have more than 1 dimension
    if returns_true.ndim > 1:
        returns_true = returns_true.flatten()
    if returns_pred.ndim > 1:
        returns_pred = returns_pred.flatten()
    
    # Calculate cumulative returns
    cum_returns_true = np.cumprod(1 + returns_true) * initial_investment
    cum_returns_pred = np.cumprod(1 + returns_pred) * initial_investment
    
    # Calculate final portfolio values
    final_value_true = cum_returns_true[-1]
    final_value_pred = cum_returns_pred[-1]
    
    # Calculate total return
    total_return_true = (final_value_true / initial_investment) - 1
    total_return_pred = (final_value_pred / initial_investment) - 1
    
    # Adjust risk-free rate based on time period
    if time_period == 'daily':
        periods_per_year = 252
    elif time_period == 'weekly':
        periods_per_year = 52
    elif time_period == 'monthly':
        periods_per_year = 12
    elif time_period == 'quarterly':
        periods_per_year = 4
    else:  # annual
        periods_per_year = 1
    
    risk_free_rate_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate annualized returns
    n_periods = len(returns_true)
    years = n_periods / periods_per_year
    
    if years > 0:
        annual_return_true = (1 + total_return_true) ** (1 / years) - 1
        annual_return_pred = (1 + total_return_pred) ** (1 / years) - 1
    else:
        annual_return_true = np.nan
        annual_return_pred = np.nan
    
    # Calculate volatility (standard deviation of returns)
    volatility_true = np.std(returns_true) * np.sqrt(periods_per_year)
    volatility_pred = np.std(returns_pred) * np.sqrt(periods_per_year)
    
    # Calculate Sharpe ratio
    excess_return_true = annual_return_true - risk_free_rate
    excess_return_pred = annual_return_pred - risk_free_rate
    
    if volatility_true > 0:
        sharpe_ratio_true = excess_return_true / volatility_true
    else:
        sharpe_ratio_true = np.nan
    
    if volatility_pred > 0:
        sharpe_ratio_pred = excess_return_pred / volatility_pred
    else:
        sharpe_ratio_pred = np.nan
    
    # Calculate maximum drawdown
    drawdown_true = 1 - cum_returns_true / np.maximum.accumulate(cum_returns_true)
    drawdown_pred = 1 - cum_returns_pred / np.maximum.accumulate(cum_returns_pred)
    max_drawdown_true = np.max(drawdown_true)
    max_drawdown_pred = np.max(drawdown_pred)
    
    # Calculate directional accuracy
    direction_true = np.sign(returns_true)
    direction_pred = np.sign(returns_pred)
    direction_accuracy = np.mean(direction_true == direction_pred)
    
    # Calculate information ratio
    tracking_error = np.std(returns_true - returns_pred) * np.sqrt(periods_per_year)
    if tracking_error > 0:
        information_ratio = (annual_return_true - annual_return_pred) / tracking_error
    else:
        information_ratio = np.nan
    
    # Return metrics as dictionary
    metrics = {
        'true': {
            'final_value': float(final_value_true),
            'total_return': float(total_return_true),
            'annual_return': float(annual_return_true),
            'volatility': float(volatility_true),
            'sharpe_ratio': float(sharpe_ratio_true),
            'max_drawdown': float(max_drawdown_true)
        },
        'pred': {
            'final_value': float(final_value_pred),
            'total_return': float(total_return_pred),
            'annual_return': float(annual_return_pred),
            'volatility': float(volatility_pred),
            'sharpe_ratio': float(sharpe_ratio_pred),
            'max_drawdown': float(max_drawdown_pred)
        },
        'comparison': {
            'direction_accuracy': float(direction_accuracy),
            'tracking_error': float(tracking_error),
            'information_ratio': float(information_ratio)
        }
    }
    
    # Add uncertainty metrics if standard deviations are provided
    if returns_std is not None:
        # Flatten array if it has more than 1 dimension
        if returns_std.ndim > 1:
            returns_std = returns_std.flatten()
        
        # Calculate uncertainty-adjusted metrics
        metrics['uncertainty'] = {
            'mean_uncertainty': float(np.mean(returns_std)),
            'uncertainty_ratio': float(np.mean(returns_std) / np.std(returns_true))
        }
        
        # Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)
        # using predicted uncertainty
        confidence_levels = [0.95, 0.99]
        for p in confidence_levels:
            z = np.sqrt(2) * erfinv(2*p - 1)
            var = -1 * (returns_pred - z * returns_std)
            cvar = -1 * (returns_pred - z * returns_std * np.sqrt(2*np.pi) * np.exp(z**2/2) / (1-p))
            
            metrics['uncertainty'][f'VaR_{int(p*100)}'] = float(np.mean(var))
            metrics['uncertainty'][f'CVaR_{int(p*100)}'] = float(np.mean(cvar))
    
    return metrics

def calculate_climate_metrics(
    temp_true: np.ndarray,
    temp_pred: np.ndarray,
    temp_std: Optional[np.ndarray] = None,
    spatial_coords: Optional[np.ndarray] = None,
    temporal_coords: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculate climate-specific metrics for model evaluation.
    
    Parameters:
    -----------
    temp_true : np.ndarray
        Ground truth temperature values
    temp_pred : np.ndarray
        Predicted temperature values
    temp_std : np.ndarray, optional
        Predicted standard deviations (uncertainty estimates)
    spatial_coords : np.ndarray, optional
        Spatial coordinates (lat, lon)
    temporal_coords : np.ndarray, optional
        Temporal coordinates (timestamps)
        
    Returns:
    --------
    metrics : dict
        Dictionary of climate metrics
    """
    # Ensure inputs are numpy arrays
    temp_true = np.asarray(temp_true)
    temp_pred = np.asarray(temp_pred)
    
    # Calculate basic regression metrics
    basic_metrics = calculate_regression_metrics(temp_true, temp_pred)
    
    # Calculate seasonal decomposition if temporal coordinates are provided
    seasonal_metrics = {}
    if temporal_coords is not None:
        # TODO: Implement seasonal decomposition
        pass
    
    # Calculate spatial error metrics if spatial coordinates are provided
    spatial_metrics = {}
    if spatial_coords is not None and temp_true.ndim > 1:
        # TODO: Implement spatial error metrics
        pass
    
    # Calculate climate-specific metrics
    
    # Mean bias error
    bias = np.mean(temp_pred - temp_true)
    
    # Normalized RMSE
    range_true = np.max(temp_true) - np.min(temp_true)
    if range_true > 0:
        nrmse = basic_metrics['rmse'] / range_true
    else:
        nrmse = np.nan
    
    # Temperature difference distribution
    temp_diff = temp_pred - temp_true
    
    # Return metrics as dictionary
    metrics = {
        **basic_metrics,
        'bias': float(bias),
        'nrmse': float(nrmse),
        'temp_diff_mean': float(np.mean(temp_diff)),
        'temp_diff_std': float(np.std(temp_diff)),
        'temp_diff_min': float(np.min(temp_diff)),
        'temp_diff_max': float(np.max(temp_diff)),
        'seasonal_metrics': seasonal_metrics,
        'spatial_metrics': spatial_metrics
    }
    
    # Add uncertainty metrics if standard deviations are provided
    if temp_std is not None:
        # Calculate calibration metrics
        calibration_metrics = calculate_calibration_metrics(temp_true, temp_pred, temp_std)
        metrics['uncertainty'] = calibration_metrics
    
    return metrics