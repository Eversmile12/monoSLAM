"""
Factory for creating motion visualization components.

This module provides a factory for creating different motion visualizers.
"""

from .base_visualizer import BaseVisualizer
from .trajectory_visualizer import TrajectoryVisualizer
from .feature_visualizer import FeatureVisualizer

class VisualizerFactory:
    """
    Factory class for creating motion visualizers.
    
    This class provides methods to create different types of visualizers
    with consistent configuration.
    """
    
    @staticmethod
    def create_trajectory_visualizer(window_name="Trajectory Visualization", size=(800, 600),
                                    background_color=(255, 255, 255), trajectory_color=(0, 0, 255),
                                    current_position_color=(0, 0, 255), start_position_color=(0, 255, 0),
                                    motion_vector_color=(255, 0, 0), grid_color=(200, 200, 200),
                                    text_color=(0, 0, 0), keyframe_color=(255, 255, 0)):
        """
        Create a trajectory visualizer.
        
        Args:
            window_name: Name of the visualization window
            size: Size of the visualization window (width, height)
            background_color: Background color (BGR)
            trajectory_color: Color for trajectory line (BGR)
            current_position_color: Color for current position marker (BGR)
            start_position_color: Color for start position marker (BGR)
            motion_vector_color: Color for motion vectors (BGR)
            grid_color: Color for grid lines (BGR)
            text_color: Color for text overlay (BGR)
            keyframe_color: Color for keyframe markers (BGR)
            
        Returns:
            TrajectoryVisualizer instance
        """
        return TrajectoryVisualizer(
            window_name=window_name,
            size=size,
            background_color=background_color,
            trajectory_color=trajectory_color,
            current_position_color=current_position_color,
            start_position_color=start_position_color,
            motion_vector_color=motion_vector_color,
            grid_color=grid_color,
            text_color=text_color,
            keyframe_color=keyframe_color
        )
    
    @staticmethod
    def create_feature_visualizer(window_name="Feature Visualization", size=(800, 600)):
        """
        Create a feature visualizer.
        
        Args:
            window_name: Name of the visualization window
            size: Size of the visualization window (width, height)
            
        Returns:
            FeatureVisualizer instance
        """
        return FeatureVisualizer(
            window_name=window_name,
            size=size
        )
    
    @staticmethod
    def create_composite_visualizer(motion_estimator, window_sizes=None, show_trajectory=True, 
                                  show_features=True):
        """
        Create multiple visualizers for a complete visualization setup.
        
        Args:
            motion_estimator: MotionEstimator instance to visualize
            window_sizes: Dictionary of window sizes for each visualizer
            show_trajectory: Whether to create trajectory visualizer
            show_features: Whether to create feature visualizer
            
        Returns:
            Dictionary of visualizer instances
        """
        if window_sizes is None:
            window_sizes = {
                'trajectory': (800, 600),
                'features': (800, 600)
            }
        
        visualizers = {}
        
        if show_trajectory:
            visualizers['trajectory'] = VisualizerFactory.create_trajectory_visualizer(
                window_name="Trajectory Visualization",
                size=window_sizes.get('trajectory', (800, 600))
            )
            
        if show_features:
            visualizers['features'] = VisualizerFactory.create_feature_visualizer(
                window_name="Feature Matching",
                size=window_sizes.get('features', (800, 600))
            )
            
        return visualizers 