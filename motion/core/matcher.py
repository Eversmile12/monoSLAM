"""
Feature matching components for motion estimation.
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from .config import MatchingMethod

class FeatureMatcher(ABC):
    """
    Abstract base class for feature matchers.
    """
    
    def __init__(self, config):
        """
        Initialize the feature matcher.
        
        Args:
            config: MotionConfig object with matching parameters
        """
        self.config = config
        self.num_matches = 0
    
    @abstractmethod
    def match(self, kp1, desc1, kp2, desc2, prev_gray=None, gray=None):
        """
        Match features between two frames.
        
        Args:
            kp1: Keypoints from frame 1
            desc1: Descriptors from frame 1
            kp2: Keypoints from frame 2
            desc2: Descriptors from frame 2
            prev_gray: Previous frame in grayscale (for optical flow)
            gray: Current frame in grayscale (for optical flow)
            
        Returns:
            matches: List of matches
            matched_kp1: Coordinates of matched keypoints in frame 1
            matched_kp2: Coordinates of matched keypoints in frame 2
        """
        pass
    
    def get_stats(self):
        """Get matcher statistics"""
        return {'num_matches': self.num_matches}


class BruteForceMatcher(FeatureMatcher):
    """
    Brute force matcher using Hamming distance for binary descriptors.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def match(self, kp1, desc1, kp2, desc2, prev_gray=None, gray=None):
        """Match features using brute force matching."""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            self.num_matches = 0
            return [], np.array([]), np.array([])
        
        # Match descriptors using kNN
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:  # Some matches might return only one match
                m, n = match_pair
                if m.distance < self.config.max_distance_ratio * n.distance:
                    good_matches.append(m)
        
        # Get keypoint coordinates for good matches
        matched_kp1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        matched_kp2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        self.num_matches = len(good_matches)
        return good_matches, matched_kp1, matched_kp2


class FlannMatcher(FeatureMatcher):
    """
    FLANN-based matcher for faster matching with large datasets.
    """
    
    def __init__(self, config):
        super().__init__(config)
        # FLANN parameters for binary descriptors
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, 
                           table_number=6,
                           key_size=12,
                           multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    def match(self, kp1, desc1, kp2, desc2, prev_gray=None, gray=None):
        """Match features using FLANN matching."""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            self.num_matches = 0
            return [], np.array([]), np.array([])
        
        # Ensure descriptors are in the right format for FLANN
        desc1 = np.float32(desc1)
        desc2 = np.float32(desc2)
        
        # Match descriptors using kNN
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:  # Some matches might return only one match
                m, n = match_pair
                if m.distance < self.config.max_distance_ratio * n.distance:
                    good_matches.append(m)
        
        # Get keypoint coordinates for good matches
        matched_kp1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        matched_kp2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        self.num_matches = len(good_matches)
        return good_matches, matched_kp1, matched_kp2


class OpticalFlowMatcher(FeatureMatcher):
    """
    Matcher using Lucas-Kanade optical flow for tracking features.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.lk_params = config.get_optical_flow_params()
    
    def match(self, kp1, desc1, kp2, desc2, prev_gray=None, gray=None):
        """Match features using optical flow."""
        if prev_gray is None or kp1 is None or len(kp1) == 0 or gray is None:
            self.num_matches = 0
            return [], np.array([]), np.array([])
        
        # Extract keypoint coordinates
        p0 = np.float32([kp.pt for kp in kp1]).reshape(-1, 1, 2)
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **self.lk_params)
        
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        # Flatten arrays for easier handling
        matched_kp1 = good_old.reshape(-1, 2)
        matched_kp2 = good_new.reshape(-1, 2)
        
        self.num_matches = len(matched_kp1)
        return [], matched_kp1, matched_kp2


def create_matcher(method, config):
    """
    Factory function to create a matcher based on the method.
    
    Args:
        method: MatchingMethod enum value
        config: MotionConfig object with matching parameters
        
    Returns:
        FeatureMatcher: Instance of a feature matcher
    """
    if method == MatchingMethod.BRUTE_FORCE:
        return BruteForceMatcher(config)
    elif method == MatchingMethod.FLANN:
        return FlannMatcher(config)
    elif method == MatchingMethod.OPTICAL_FLOW:
        return OpticalFlowMatcher(config)
    else:
        raise ValueError(f"Unknown matching method: {method}") 