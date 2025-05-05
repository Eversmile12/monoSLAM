"""
Trajectory tracking and management.
"""

import numpy as np
import time
from collections import deque

class TrajectoryTracker:
    """
    Tracks camera trajectory and maintains motion history.
    """
    
    def __init__(self, config):
        """
        Initialize the trajectory tracker.
        
        Args:
            config: MotionConfig object
        """
        self.config = config
        self.max_history_size = config.max_history_size
        
        # Trajectory history
        self.trajectory = []
        self.current_position = np.zeros(3)
        
        # Motion history
        self.rotations = deque(maxlen=self.max_history_size)
        self.translations = deque(maxlen=self.max_history_size)
        self.timestamps = deque(maxlen=self.max_history_size)
        
        # Feature scale tracking
        self.feature_scale = config.feature_scale
        self.scale_history = deque(maxlen=config.scale_estimation_window)
        
        # Statistics
        self.total_distance = 0.0
        self.start_time = time.time()
    
    def update(self, R, t, timestamp=None):
        """
        Update the trajectory with a new motion estimate.
        
        Args:
            R: Rotation matrix (3x3)
            t: Translation vector (3x1 or 3)
            timestamp: Current timestamp (optional)
            
        Returns:
            current_position: Updated position
        """
        # Add motion to history
        self.rotations.append(R.copy())
        self.translations.append(t.copy())
        
        if timestamp is not None:
            self.timestamps.append(timestamp)
        
        # Update position
        t_flat = t.ravel() if t is not None else np.zeros(3)
        
        # Apply translation in the rotated frame
        # R.T @ t gives translation in world frame
        prev_position = self.current_position.copy()
        self.current_position = self.current_position + R.T @ t_flat
        
        # Update trajectory
        self.trajectory.append(self.current_position.copy())
        
        # Update total distance traveled
        delta = np.linalg.norm(self.current_position - prev_position)
        self.total_distance += delta
        
        return self.current_position
    
    def get_position(self):
        """Get the current position"""
        return self.current_position.copy()
    
    def get_motion_history(self, window_size=None):
        """
        Get recent motion history.
        
        Args:
            window_size: Number of recent frames to include (None for all)
            
        Returns:
            rotations: List of rotation matrices
            translations: List of translation vectors
            timestamps: List of timestamps (if available)
        """
        if window_size is None or window_size >= len(self.rotations):
            return list(self.rotations), list(self.translations), list(self.timestamps)
        else:
            return (
                list(self.rotations)[-window_size:],
                list(self.translations)[-window_size:],
                list(self.timestamps)[-window_size:]
            )
    
    def get_trajectory(self, max_points=None):
        """
        Get the trajectory points.
        
        Args:
            max_points: Maximum number of points to return (None for all)
            
        Returns:
            trajectory: Array of position points (Nx3)
        """
        if not self.trajectory:
            return np.zeros((0, 3))
            
        trajectory = np.array(self.trajectory)
        
        if max_points is not None and len(trajectory) > max_points:
            # Return the most recent points
            return trajectory[-max_points:]
            
        return trajectory
    
    def update_scale(self, scale_factor):
        """
        Update the feature scale factor.
        
        Args:
            scale_factor: New scale factor from external measurement
        """
        # Add to history
        self.scale_history.append(scale_factor)
        
        # Update scale using a weighted average
        if len(self.scale_history) > 0:
            # More weight to recent measurements
            weights = np.linspace(0.5, 1.0, len(self.scale_history))
            weights = weights / np.sum(weights)
            
            self.feature_scale = np.sum(np.array(self.scale_history) * weights)
    
    def get_scale(self):
        """Get the current scale factor"""
        return self.feature_scale
    
    def get_stats(self):
        """Get trajectory statistics"""
        elapsed_time = time.time() - self.start_time
        return {
            'total_distance': self.total_distance,
            'elapsed_time': elapsed_time,
            'trajectory_length': len(self.trajectory),
            'current_position': self.current_position.tolist(),
            'scale_factor': self.feature_scale
        }
    
    def reset(self):
        """Reset the trajectory"""
        self.trajectory = []
        self.current_position = np.zeros(3)
        self.rotations.clear()
        self.translations.clear()
        self.timestamps.clear()
        self.total_distance = 0.0
        self.start_time = time.time() 