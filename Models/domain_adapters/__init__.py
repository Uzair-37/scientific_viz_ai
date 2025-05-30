"""
Domain adapter implementations for the Multi-Modal AI system.

This package contains domain adapters for finance and climate domains,
which adapt the core multi-modal architecture to specific application domains.
"""

from .base_adapter import BaseAdapter
from .finance_adapter import FinanceAdapter
from .climate_adapter import ClimateAdapter

__all__ = ['BaseAdapter', 'FinanceAdapter', 'ClimateAdapter']