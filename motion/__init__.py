"""
MonoSLAM Motion Estimation Package
==================================

This package provides components for visual motion estimation in the MonoSLAM system.
"""

# Import core components to make them available at the root level
from .core.motion_estimator import MotionEstimator
from .core.matcher import MatchingMethod, create_matcher
from .core.config import MotionConfig

# Import visualization components
from .viz.trajectory_visualizer import TrajectoryVisualizer
from .viz.feature_visualizer import FeatureVisualizer
from .viz.visualizer_factory import VisualizerFactory

__all__ = [
    # Core components
    'MotionEstimator', 
    'MatchingMethod', 
    'create_matcher', 
    'MotionConfig',
    
    # Visualization components
    'TrajectoryVisualizer',
    'FeatureVisualizer',
    'VisualizerFactory',
] 