"""
Temporal filtering for motion smoothing.
"""

import numpy as np
from ..core.utils import (
    rotation_to_angle_axis, 
    angle_axis_to_rotation, 
    orthogonalize_rotation,
    slerp_rotation
)

class TemporalFilter:
    """
    Applies temporal filtering to motion estimates to reduce noise.
    """
    
    def __init__(self, config):
        """
        Initialize the temporal filter.
        
        Args:
            config: MotionConfig object with temporal filter parameters
        """
        self.config = config
        self.enable = config.enable_temporal_filter
        self.position_alpha = config.position_filter_alpha
        self.rotation_alpha = config.rotation_filter_alpha
        self.adaptive_smoothing = config.adaptive_smoothing
        self.velocity_weight = config.velocity_weight
        
        # State variables
        self.filtered_position = None
        self.filtered_rotation = None
        self.position_velocity = np.zeros(3)
        self.rotation_velocity = np.zeros(3)  # Axis-angle representation
        self.last_position = None
        self.last_rotation = None
        self.last_timestamp = None
    
    def filter(self, R, t, position, timestamp=None, tracking_quality=1.0):
        """
        Apply temporal filtering to motion.
        
        Args:
            R: Rotation matrix (3x3)
            t: Translation vector (3x1 or 3)
            position: Current position (3)
            timestamp: Current timestamp (optional)
            tracking_quality: Quality metric from 0 to 1
            
        Returns:
            filtered_R: Filtered rotation matrix
            filtered_t: Filtered translation vector
            filtered_position: Filtered position
        """
        if not self.enable:
            return R, t, position
        
        # Ensure proper array dimensions
        t_flat = t.ravel() if t is not None else np.zeros(3)
        
        # Initialize filtered state if not done yet
        if self.filtered_position is None:
            self.filtered_position = position.copy()
        if self.filtered_rotation is None:
            self.filtered_rotation = R.copy()
        if self.last_position is None:
            self.last_position = position.copy()
        if self.last_rotation is None:
            self.last_rotation = R.copy()
        
        # Calculate time delta
        dt = 1.0  # Default if no timestamps
        if timestamp is not None and self.last_timestamp is not None:
            dt = max(0.001, timestamp - self.last_timestamp)  # Avoid division by zero
        
        # Adapt filtering strength based on tracking quality if enabled
        position_alpha = self.position_alpha
        rotation_alpha = self.rotation_alpha
        
        if self.adaptive_smoothing:
            # Reduce smoothing when tracking quality is high
            # This allows faster response during reliable tracking
            quality_factor = min(1.0, max(0.3, tracking_quality))
            position_alpha *= quality_factor
            rotation_alpha *= quality_factor
            
            # Increase smoothing when tracking quality is low
            if tracking_quality < 0.5:
                position_alpha = max(0.05, position_alpha * 0.5)
                rotation_alpha = max(0.05, rotation_alpha * 0.5)
        
        # ----------------------------------
        # Position filtering
        # ----------------------------------
        
        # Calculate position delta
        position_delta = position - self.last_position
        
        # Update velocity estimate
        current_velocity = position_delta / dt
        self.position_velocity = (1 - position_alpha) * self.position_velocity + position_alpha * current_velocity
        
        # Predict position using velocity
        predicted_position = position
        if self.velocity_weight > 0:
            velocity_prediction = self.position_velocity * dt
            predicted_position = position + velocity_prediction * self.velocity_weight
        
        # Apply filtering to position
        self.filtered_position = (1 - position_alpha) * self.filtered_position + position_alpha * predicted_position
        
        # ----------------------------------
        # Rotation filtering
        # ----------------------------------
        
        # Calculate rotation delta
        R_delta = R @ self.last_rotation.T
        angle, axis = rotation_to_angle_axis(R_delta)
        
        # Limit maximum rotation angle (outlier rejection)
        max_angle = 0.5  # ~30 degrees per update
        if angle > max_angle:
            angle = max_angle
            # Recalculate R_delta with limited angle
            R_delta = angle_axis_to_rotation(angle, axis)
        
        # Convert to rotation vector
        rot_vector = axis * angle if not np.isclose(angle, 0) else np.zeros(3)
        
        # Update rotation velocity
        current_rot_velocity = rot_vector / dt
        self.rotation_velocity = (1 - rotation_alpha) * self.rotation_velocity + rotation_alpha * current_rot_velocity
        
        # Apply slerp-based filtering
        smoothing_factor = 1.0 - rotation_alpha  # Convert alpha to smoothing factor (1 = no smoothing)
        self.filtered_rotation = slerp_rotation(self.filtered_rotation, R, 1.0 - smoothing_factor)
        
        # Ensure rotation is valid
        self.filtered_rotation = orthogonalize_rotation(self.filtered_rotation)
        
        # Update last values
        self.last_position = position.copy()
        self.last_rotation = R.copy()
        self.last_timestamp = timestamp
        
        # Return filtered motion
        return self.filtered_rotation, t, self.filtered_position
    
    def reset(self):
        """Reset the filter state."""
        self.filtered_position = None
        self.filtered_rotation = None
        self.position_velocity = np.zeros(3)
        self.rotation_velocity = np.zeros(3)
        self.last_position = None
        self.last_rotation = None
        self.last_timestamp = None
    
    def get_stats(self):
        """Get filter statistics."""
        return {
            'position_alpha': self.position_alpha,
            'rotation_alpha': self.rotation_alpha,
            'position_velocity': np.linalg.norm(self.position_velocity),
            'rotation_velocity': np.linalg.norm(self.rotation_velocity)
        } 