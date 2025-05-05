"""
Core motion estimation components.

This package provides core components for motion estimation, including:
- MotionEstimator: Main component for estimating motion from visual features
- TrajectoryTracker: Tracks camera trajectory and maintains motion history
- KeyframeManager: Manages keyframe selection and tracking
- MatchingMethod: Enum for different feature matching methods
- MotionConfig: Configuration parameters for motion estimation
"""

from .motion_estimator import MotionEstimator
from .trajectory import TrajectoryTracker
from .keyframe import KeyframeManager, Keyframe
from .matcher import MatchingMethod, create_matcher
from .config import MotionConfig

__all__ = [
    'MotionEstimator', 
    'TrajectoryTracker', 
    'KeyframeManager',
    'Keyframe',
    'MatchingMethod',
    'create_matcher',
    'MotionConfig'
] 