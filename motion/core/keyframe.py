"""
Keyframe selection and management.
"""

import numpy as np
import cv2
import time
from ..core.utils import rotation_to_angle_axis

class Keyframe:
    """
    Represents a keyframe with its features and pose information.
    """
    
    def __init__(self, frame_id, timestamp, image, keypoints, descriptors, 
                 position, rotation, scale_factor=1.0):
        """
        Initialize a keyframe.
        
        Args:
            frame_id: Unique identifier for the frame
            timestamp: Capture timestamp
            image: Grayscale image data
            keypoints: List of keypoints
            descriptors: Feature descriptors
            position: 3D position in world coordinates
            rotation: Rotation matrix
            scale_factor: Scale factor at time of capture
        """
        self.id = frame_id
        self.timestamp = timestamp
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.position = position
        self.rotation = rotation
        self.scale_factor = scale_factor
        
        # Calculate feature density and distribution metrics
        self.feature_count = len(keypoints) if keypoints is not None else 0
        self.feature_quality = self._calculate_feature_quality()
    
    def _calculate_feature_quality(self):
        """Calculate a quality metric for feature distribution"""
        if self.keypoints is None or len(self.keypoints) < 5:
            return 0.0
            
        # Get keypoint positions
        pts = np.array([kp.pt for kp in self.keypoints])
        
        if len(pts) < 5:
            return 0.0
        
        # Calculate coverage in different regions of the image
        h, w = self.image.shape if self.image is not None else (480, 640)
        
        # Divide image into a 3x3 grid and count points in each cell
        grid_cells = np.zeros((3, 3))
        for pt in pts:
            cell_x = min(2, int(pt[0] / (w / 3)))
            cell_y = min(2, int(pt[1] / (h / 3)))
            grid_cells[cell_y, cell_x] += 1
        
        # Calculate distribution metric (percentage of cells with features)
        cells_with_features = np.sum(grid_cells > 0)
        distribution_metric = cells_with_features / 9.0  # 9 cells total
        
        # Adjust by feature count (up to a reasonable maximum)
        count_metric = min(1.0, self.feature_count / 200.0)
        
        # Combine metrics with distribution weighted more
        return 0.7 * distribution_metric + 0.3 * count_metric
        

class KeyframeManager:
    """
    Manages keyframe selection and tracking.
    """
    
    def __init__(self, config):
        """
        Initialize the keyframe manager.
        
        Args:
            config: MotionConfig object
        """
        self.config = config
        self.use_keyframes = config.use_keyframes
        self.keyframe_min_translation = config.keyframe_min_translation
        self.keyframe_min_rotation = config.keyframe_min_rotation
        self.keyframe_min_features = config.keyframe_min_features
        self.keyframe_max_time_interval = config.keyframe_max_time_interval
        
        # Keyframe storage
        self.keyframes = []
        self.latest_keyframe = None
        self.frame_counter = 0
        self.last_keyframe_time = 0
    
    def should_create_keyframe(self, position, rotation, feature_count, timestamp=None):
        """
        Determine if a new keyframe should be created.
        
        Args:
            position: Current position
            rotation: Current rotation matrix
            feature_count: Number of features in current frame
            timestamp: Current timestamp (optional)
            
        Returns:
            bool: True if a new keyframe should be created
        """
        if not self.use_keyframes:
            return False
            
        # Always create the first keyframe
        if not self.keyframes:
            return True
        
        # Don't create keyframes too frequently
        if self.frame_counter < 5:
            self.frame_counter += 1
            return False
            
        # Reset frame counter
        self.frame_counter = 0
        
        # Check time interval
        if timestamp is not None and self.last_keyframe_time > 0:
            time_elapsed = timestamp - self.last_keyframe_time
            if time_elapsed > self.keyframe_max_time_interval:
                return True
        
        # Get the latest keyframe
        if not self.latest_keyframe:
            return True
            
        # Check translation distance
        distance = np.linalg.norm(position - self.latest_keyframe.position)
        if distance > self.keyframe_min_translation:
            return True
            
        # Check rotation difference
        if rotation is not None and self.latest_keyframe.rotation is not None:
            relative_rotation = rotation @ self.latest_keyframe.rotation.T
            angle, _ = rotation_to_angle_axis(relative_rotation)
            if angle > self.keyframe_min_rotation:
                return True
        
        # Check feature count (if significantly higher than current keyframe)
        if feature_count > self.keyframe_min_features and \
           feature_count > self.latest_keyframe.feature_count * 1.5:
            return True
            
        return False
    
    def add_keyframe(self, frame_id, timestamp, image, keypoints, descriptors, 
                    position, rotation, scale_factor=1.0):
        """
        Add a new keyframe.
        
        Args:
            frame_id: Unique identifier for the frame
            timestamp: Capture timestamp
            image: Grayscale image data
            keypoints: List of keypoints
            descriptors: Feature descriptors
            position: 3D position in world coordinates
            rotation: Rotation matrix
            scale_factor: Scale factor at time of capture
            
        Returns:
            keyframe: The newly created keyframe
        """
        # Create keyframe
        keyframe = Keyframe(
            frame_id, timestamp, image, keypoints, descriptors,
            position.copy(), rotation.copy(), scale_factor
        )
        
        # Add to collection
        self.keyframes.append(keyframe)
        self.latest_keyframe = keyframe
        
        # Update time
        if timestamp is not None:
            self.last_keyframe_time = timestamp
        
        return keyframe
    
    def get_best_matching_keyframe(self, descriptors, max_distance_ratio=0.75, min_matches=10):
        """
        Find the best matching keyframe for the given descriptors.
        
        Args:
            descriptors: Query descriptors
            max_distance_ratio: Ratio test threshold
            min_matches: Minimum number of matches required
            
        Returns:
            keyframe: Best matching keyframe
            match_count: Number of matches
        """
        if not self.keyframes or descriptors is None or len(descriptors) == 0:
            return None, 0
            
        best_keyframe = None
        best_match_count = 0
        
        # Create matcher
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Match against all keyframes
        for keyframe in self.keyframes:
            if keyframe.descriptors is None or len(keyframe.descriptors) == 0:
                continue
                
            # Match descriptors
            matches = matcher.knnMatch(descriptors, keyframe.descriptors, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < max_distance_ratio * n.distance:
                        good_matches.append(m)
            
            # Update best match
            if len(good_matches) > best_match_count:
                best_match_count = len(good_matches)
                best_keyframe = keyframe
        
        # Return the best match if it meets the threshold
        if best_match_count >= min_matches:
            return best_keyframe, best_match_count
            
        return None, 0
    
    def get_keyframes(self):
        """Get all keyframes"""
        return self.keyframes
    
    def get_latest_keyframe(self):
        """Get the most recent keyframe"""
        return self.latest_keyframe
    
    def clear(self):
        """Clear all keyframes"""
        self.keyframes = []
        self.latest_keyframe = None
        self.frame_counter = 0
        self.last_keyframe_time = 0
    
    def get_stats(self):
        """Get keyframe statistics"""
        return {
            'keyframe_count': len(self.keyframes),
            'latest_id': self.latest_keyframe.id if self.latest_keyframe else None,
            'avg_features': np.mean([kf.feature_count for kf in self.keyframes]) if self.keyframes else 0,
            'avg_quality': np.mean([kf.feature_quality for kf in self.keyframes]) if self.keyframes else 0
        } 