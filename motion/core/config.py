"""
Configuration classes for motion estimation.
"""

from enum import Enum

class MatchingMethod(Enum):
    """Enum for different feature matching methods"""
    BRUTE_FORCE = 1
    FLANN = 2
    OPTICAL_FLOW = 3

class MotionConfig:
    """
    Configuration parameters for motion estimation.
    Centralizes all parameter management in one place.
    """
    
    def __init__(self):
        # Feature matching parameters
        self.matching_method = MatchingMethod.BRUTE_FORCE
        self.max_distance_ratio = 0.75
        self.min_matches = 5
        self.use_ransac = True
        
        # Optical flow parameters
        self.lk_params = {
            'winSize': (41, 41),
            'maxLevel': 5,
            'criteria': (0x01 + 0x10, 30, 0.001),  # CV_TERMCRIT_EPS + CV_TERMCRIT_ITER
            'minEigThreshold': 0.0005,
            'flags': 8  # OPTFLOW_LK_GET_MIN_EIGENVALS
        }
        
        # Essential matrix parameters
        self.ransac_threshold = 2.0
        self.confidence = 0.999
        
        # Motion model parameters
        self.use_motion_model = True
        self.motion_window_size = 8
        self.max_history_size = 30
        self.use_momentum = True
        self.momentum_factor = 0.8
        self.use_speed_model = True
        
        # Motion smoothing parameters
        self.use_smoothing = True
        self.smoothing_factor = 0.7
        
        # Temporal filtering parameters
        self.enable_temporal_filter = True
        self.position_filter_alpha = 0.2
        self.rotation_filter_alpha = 0.3
        self.adaptive_smoothing = True
        self.velocity_weight = 0.5
        
        # Recovery parameters
        self.max_consecutive_failures = 3
        
        # Spatial consistency parameters
        self.max_spatial_distance = 100
        self.direction_cosine_threshold = 0.7
        
        # Debug parameters
        self.debug_mode = True
        
        # Scale parameters
        self.feature_scale = 1.0
        self.scale_estimation_window = 10
        
        # NEW: Keyframe parameters
        self.use_keyframes = False
        self.keyframe_min_translation = 0.2
        self.keyframe_min_rotation = 0.15  # ~8.6 degrees
        self.keyframe_min_features = 100
        self.keyframe_max_time_interval = 5.0  # seconds
        
        # NEW: Adaptive parameters
        self.enable_adaptive_parameters = False
        self.quality_window_size = 10
        self.min_smoothing_factor = 0.3
        self.max_smoothing_factor = 0.9

    def get_optical_flow_params(self):
        """Get OpenCV-compatible optical flow parameters"""
        import cv2
        # Convert the criteria to the format expected by OpenCV
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                  self.lk_params['criteria'][1], 
                  self.lk_params['criteria'][2])
        
        return {
            'winSize': self.lk_params['winSize'],
            'maxLevel': self.lk_params['maxLevel'],
            'criteria': criteria,
            'minEigThreshold': self.lk_params['minEigThreshold'],
            'flags': self.lk_params['flags']
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create a config from a dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def to_dict(self):
        """Convert config to a dictionary"""
        return {key: value for key, value in vars(self).items()
                if not key.startswith('_')} 