"""
Filtering components for motion data.

This package provides various filters for motion data, including:
- SpatialConsistencyFilter: Filters outlier matches based on spatial consistency
- TemporalFilter: Smooths motion over time using temporal filtering
- AdaptiveParamManager: Dynamically adjusts parameters based on tracking quality
"""

from .spatial_filter import SpatialConsistencyFilter
from .temporal_filter import TemporalFilter
from .adaptive_params import AdaptiveParamManager

__all__ = ['SpatialConsistencyFilter', 'TemporalFilter', 'AdaptiveParamManager'] 