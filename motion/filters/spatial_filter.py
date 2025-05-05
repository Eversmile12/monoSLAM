"""
Spatial consistency filtering for feature matches.
"""

import numpy as np

class SpatialConsistencyFilter:
    """
    Filter matches based on spatial consistency to remove outliers.
    """
    
    def __init__(self, config):
        """
        Initialize the spatial filter.
        
        Args:
            config: MotionConfig object with spatial filter parameters
        """
        self.config = config
        self.max_distance = config.max_spatial_distance
        self.direction_threshold = config.direction_cosine_threshold
    
    def filter_matches(self, matched_kp1, matched_kp2):
        """
        Filter matches based on spatial consistency.
        
        Args:
            matched_kp1: Coordinates of matched keypoints in frame 1 (Nx2)
            matched_kp2: Coordinates of matched keypoints in frame 2 (Nx2)
            
        Returns:
            mask: Boolean mask of consistent matches
            matched_kp1_filtered: Filtered keypoints from frame 1
            matched_kp2_filtered: Filtered keypoints from frame 2
        """
        if len(matched_kp1) < 5:
            # Not enough matches to perform filtering
            return np.ones(len(matched_kp1), dtype=bool), matched_kp1, matched_kp2
        
        # Calculate displacements
        displacements = matched_kp2 - matched_kp1
        
        # Calculate distances (magnitudes of displacement vectors)
        distances = np.sqrt(np.sum(displacements**2, axis=1))
        
        # Filter by maximum allowed distance
        distance_mask = distances < self.max_distance
        
        # If we have enough points after distance filtering, do direction consistency check
        if np.sum(distance_mask) > 5:
            # Get median displacement for consistent matches
            filtered_displacements = displacements[distance_mask]
            median_displacement = np.median(filtered_displacements, axis=0)
            
            # Calculate angular difference between each displacement and the median
            dot_products = np.sum(displacements * median_displacement, axis=1)
            displacement_norms = np.sqrt(np.sum(displacements**2, axis=1))
            median_norm = np.linalg.norm(median_displacement)
            
            # Avoid division by zero
            safe_norms = np.maximum(displacement_norms * median_norm, 1e-6)
            cos_angles = dot_products / safe_norms
            
            # Allow for some deviation in direction based on threshold
            direction_mask = cos_angles > self.direction_threshold
            
            # Combine masks
            final_mask = distance_mask & direction_mask
        else:
            # Not enough points for direction filtering
            final_mask = distance_mask
        
        # Filter the keypoints
        matched_kp1_filtered = matched_kp1[final_mask]
        matched_kp2_filtered = matched_kp2[final_mask]
        
        return final_mask, matched_kp1_filtered, matched_kp2_filtered
    
    def get_stats(self):
        """Get filter statistics"""
        return {
            'max_distance': self.max_distance,
            'direction_threshold': self.direction_threshold
        } 