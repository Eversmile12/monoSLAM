"""
Motion models for prediction and interpolation.
"""

import numpy as np
from ..core.utils import (
    rotation_to_angle_axis, 
    angle_axis_to_rotation, 
    orthogonalize_rotation,
    slerp_rotation
)

class MotionModel:
    """
    Predicts motion based on previous motion history.
    Supports multiple prediction strategies.
    """
    
    def __init__(self, config):
        """
        Initialize the motion model.
        
        Args:
            config: MotionConfig object
        """
        self.config = config
        self.use_motion_model = config.use_motion_model
        self.motion_window_size = config.motion_window_size
        self.use_momentum = config.use_momentum
        self.momentum_factor = config.momentum_factor
        self.use_speed_model = config.use_speed_model
    
    def predict(self, rotations, translations, timestamps=None, tracking_quality=1.0):
        """
        Predict the next motion based on history.
        
        Args:
            rotations: List of previous rotation matrices
            translations: List of previous translation vectors
            timestamps: List of previous timestamps (optional)
            tracking_quality: Current tracking quality (0-1)
            
        Returns:
            R_pred: Predicted rotation matrix
            t_pred: Predicted translation vector
        """
        if not self.use_motion_model or len(rotations) < 1:
            return None, None
        
        # Start with most recent values
        R_last = rotations[-1]
        t_last = translations[-1]
        
        # If we only have one previous motion, return it directly
        if len(rotations) < 2:
            return R_last, t_last
        
        # For recovery during low match frames, we need more robust motion prediction
        # Determine prediction confidence based on available history
        confidence_level = min(1.0, len(rotations) / 10.0)
        
        # Get a recent window of velocities for averaging
        window_size = min(len(rotations) - 1, self.motion_window_size)
        
        # Use momentum-based prediction if enabled
        if self.use_momentum and len(rotations) >= 2:
            # Get previous and current values
            R_prev = rotations[-2]
            t_prev = translations[-2]
            
            # Calculate relative rotation (delta rotation)
            delta_R = R_last @ R_prev.T
            
            # Extract rotation angle and axis
            angle, axis = rotation_to_angle_axis(delta_R)
            
            # Apply momentum to rotation - scaled by quality
            quality_factor = max(0.5, tracking_quality)  # Limit momentum during low quality
            angle_pred = angle * self.momentum_factor * quality_factor
            R_momentum = angle_axis_to_rotation(angle_pred, axis)
            
            # Apply momentum-based rotation
            R_pred = R_momentum @ R_last
            
            # Calculate predicted translation with momentum
            delta_t = t_last - t_prev
            t_momentum = delta_t * self.momentum_factor * quality_factor
            t_pred = t_last + t_momentum
            
            # Ensure R is orthogonal
            R_pred = orthogonalize_rotation(R_pred)
            
            # Adjust translation based on timestamp if available
            if self.use_speed_model and timestamps is not None and len(timestamps) >= 2:
                last_dt = timestamps[-1] - timestamps[-2]
                if last_dt > 0:
                    # Scale translation by time delta
                    nominal_dt = 0.033  # Assuming nominal frame time is 30fps
                    t_pred = t_pred * (nominal_dt / last_dt)
            
            # Ensure unit length for translation
            t_norm = np.linalg.norm(t_pred)
            if t_norm > 0:
                t_pred = t_pred / t_norm
                
                # Apply scale factor to match previous magnitudes
                recent_translations = translations[-min(5, len(translations)):]
                avg_t_magnitude = np.mean([np.linalg.norm(t) for t in recent_translations])
                t_pred = t_pred * avg_t_magnitude
            
            return R_pred, t_pred
        
        # If momentum is not enabled, use weighted average of past motions
        weights = np.linspace(0.5, 1.0, len(rotations))[-self.motion_window_size:]
        weights = weights / np.sum(weights)
        
        # Weighted average of translations
        t_pred = np.zeros_like(t_last)
        for i, (t, w) in enumerate(zip(list(translations)[-self.motion_window_size:], weights)):
            t_pred += t * w
            
        # Normalize translation
        t_norm = np.linalg.norm(t_pred)
        if t_norm > 0:
            t_pred = t_pred / t_norm
            
            # Scale to average magnitude of recent translations
            avg_t_magnitude = np.mean([np.linalg.norm(t) for t in translations[-3:]])
            t_pred = t_pred * avg_t_magnitude
        
        # For rotation, use SLERP interpolation between last few rotations
        R_pred = rotations[-1].copy()
        for i in range(min(3, len(rotations)-1)):
            R_prev = rotations[-(i+2)]
            R_pred = slerp_rotation(R_prev, R_pred, 0.7)  # Blend with 0.7 weight to current
        
        return R_pred, t_pred
    
    def interpolate(self, R1, t1, R2, t2, factor):
        """
        Interpolate between two motion states.
        
        Args:
            R1: First rotation matrix
            t1: First translation vector
            R2: Second rotation matrix
            t2: Second translation vector
            factor: Interpolation factor (0 = first state, 1 = second state)
            
        Returns:
            R: Interpolated rotation matrix
            t: Interpolated translation vector
        """
        # Ensure factor is within [0, 1]
        factor = max(0, min(1, factor))
        
        # Use SLERP for rotation interpolation
        R = slerp_rotation(R1, R2, factor)
        
        # Linear interpolation for translation
        t = (1 - factor) * t1 + factor * t2
        
        return R, t 