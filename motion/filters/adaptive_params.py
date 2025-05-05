"""
Adaptive parameter adjustment for motion estimation.
"""

import numpy as np
from collections import deque

class AdaptiveParamManager:
    """
    Dynamically adjusts filtering parameters based on tracking quality metrics.
    """
    
    def __init__(self, config):
        """
        Initialize the adaptive parameter manager.
        
        Args:
            config: MotionConfig object with parameters
        """
        self.config = config
        self.enabled = config.enable_adaptive_parameters
        self.window_size = config.quality_window_size
        self.min_smoothing = config.min_smoothing_factor
        self.max_smoothing = config.max_smoothing_factor
        
        # History windows for metrics
        self.match_counts = deque(maxlen=self.window_size)
        self.inlier_counts = deque(maxlen=self.window_size)
        self.tracking_qualities = deque(maxlen=self.window_size)
        
        # Current parameter values
        self.current_position_alpha = config.position_filter_alpha
        self.current_rotation_alpha = config.rotation_filter_alpha
        self.current_smoothing_factor = config.smoothing_factor
        self.current_ransac_threshold = config.ransac_threshold
    
    def update(self, match_count, inlier_count, tracking_quality):
        """
        Update the metrics history and adjust parameters.
        
        Args:
            match_count: Number of matches in current frame
            inlier_count: Number of inliers in current frame
            tracking_quality: Current tracking quality metric (0-1)
            
        Returns:
            params: Dictionary of updated parameters
        """
        if not self.enabled:
            return self._get_current_params()
        
        # Add new metrics to history
        self.match_counts.append(match_count)
        self.inlier_counts.append(inlier_count)
        self.tracking_qualities.append(tracking_quality)
        
        # Need enough history to make adjustments
        if len(self.match_counts) < 3:
            return self._get_current_params()
        
        # Calculate statistics
        match_variance = np.var(self.match_counts) if len(self.match_counts) > 1 else 0
        match_mean = np.mean(self.match_counts)
        tracking_mean = np.mean(self.tracking_qualities)
        tracking_trend = self._calculate_trend(self.tracking_qualities)
        match_ratio = match_variance / (match_mean + 1e-5)  # Normalized variance
        
        # Adjust smoothing factor based on match stability
        # - More smoothing when matches are unstable
        # - Less smoothing when matches are stable and plentiful
        smoothing_factor = self._adjust_smoothing_factor(match_ratio, tracking_mean)
        
        # Adjust alpha values for position and rotation
        # - Smaller alpha = more smoothing
        # - Larger alpha = less smoothing, more responsive
        position_alpha = self._adjust_position_alpha(tracking_mean, tracking_trend)
        rotation_alpha = self._adjust_rotation_alpha(tracking_mean, tracking_trend)
        
        # Adjust RANSAC threshold based on tracking quality
        # - Lower threshold when tracking is good (more strict)
        # - Higher threshold when tracking is poor (more lenient)
        ransac_threshold = self._adjust_ransac_threshold(tracking_mean)
        
        # Update current values
        self.current_smoothing_factor = smoothing_factor
        self.current_position_alpha = position_alpha
        self.current_rotation_alpha = rotation_alpha
        self.current_ransac_threshold = ransac_threshold
        
        return self._get_current_params()
    
    def _calculate_trend(self, values):
        """Calculate the trend in a series of values (-1 to 1)"""
        if len(values) < 3:
            return 0
            
        # Simple linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Normalize to [-1, 1]
        max_slope = 0.1  # Maximum expected slope
        return np.clip(slope / max_slope, -1, 1)
    
    def _adjust_smoothing_factor(self, match_ratio, tracking_quality):
        """Adjust the smoothing factor based on match stability and tracking quality"""
        # Base adjustment on match stability
        # Higher match_ratio = more variance = need more smoothing
        base_smoothing = np.clip(
            self.config.smoothing_factor + 0.1 * match_ratio, 
            self.min_smoothing, 
            self.max_smoothing
        )
        
        # Incorporate tracking quality
        # Lower quality = more smoothing
        quality_factor = 1.0 - (1.0 - tracking_quality) * 0.5
        
        return np.clip(
            base_smoothing / quality_factor,
            self.min_smoothing,
            self.max_smoothing
        )
    
    def _adjust_position_alpha(self, tracking_quality, tracking_trend):
        """Adjust position filter alpha based on tracking quality"""
        base_alpha = self.config.position_filter_alpha
        
        # Lower alpha (more smoothing) when tracking quality is poor
        quality_factor = 0.5 + 0.5 * tracking_quality
        
        # Adjust based on trend: if improving, be more responsive
        trend_factor = 1.0 + 0.2 * max(0, tracking_trend)
        
        return np.clip(
            base_alpha * quality_factor * trend_factor,
            0.05,  # Minimum alpha
            0.5    # Maximum alpha
        )
    
    def _adjust_rotation_alpha(self, tracking_quality, tracking_trend):
        """Adjust rotation filter alpha based on tracking quality"""
        base_alpha = self.config.rotation_filter_alpha
        
        # Similar to position, but rotation can be more sensitive
        quality_factor = 0.4 + 0.6 * tracking_quality
        trend_factor = 1.0 + 0.15 * max(0, tracking_trend)
        
        return np.clip(
            base_alpha * quality_factor * trend_factor,
            0.05,  # Minimum alpha
            0.4    # Maximum alpha
        )
    
    def _adjust_ransac_threshold(self, tracking_quality):
        """Adjust RANSAC threshold based on tracking quality"""
        base_threshold = self.config.ransac_threshold
        
        # Higher threshold (more lenient) when tracking is poor
        quality_factor = 2.0 - tracking_quality
        
        return np.clip(
            base_threshold * quality_factor,
            1.0,   # Minimum threshold
            3.5    # Maximum threshold
        )
    
    def _get_current_params(self):
        """Get the current parameter values"""
        return {
            'smoothing_factor': self.current_smoothing_factor,
            'position_filter_alpha': self.current_position_alpha,
            'rotation_filter_alpha': self.current_rotation_alpha,
            'ransac_threshold': self.current_ransac_threshold
        }
    
    def get_stats(self):
        """Get statistics about the adaptive parameters"""
        if len(self.match_counts) > 0:
            return {
                'match_count_mean': np.mean(self.match_counts),
                'match_count_var': np.var(self.match_counts),
                'tracking_quality_mean': np.mean(self.tracking_qualities),
                'tracking_trend': self._calculate_trend(self.tracking_qualities),
                'current_smoothing': self.current_smoothing_factor,
                'current_position_alpha': self.current_position_alpha,
                'current_rotation_alpha': self.current_rotation_alpha,
                'current_ransac_threshold': self.current_ransac_threshold
            }
        else:
            return {
                'status': 'Not enough data collected'
            } 