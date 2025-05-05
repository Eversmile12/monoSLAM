"""
Main motion estimation component for visual odometry.
"""

import cv2
import numpy as np
import time

from .config import MotionConfig, MatchingMethod
from .matcher import create_matcher
from .trajectory import TrajectoryTracker
from .keyframe import KeyframeManager
from .utils import orthogonalize_rotation

from ..filters.spatial_filter import SpatialConsistencyFilter
from ..filters.temporal_filter import TemporalFilter
from ..filters.adaptive_params import AdaptiveParamManager
from ..models.motion_model import MotionModel

class MotionEstimator:
    """
    Core component for estimating motion from visual features.
    
    Uses a set of specialized classes for handling different aspects
    of motion estimation:
    - Feature matching
    - Essential matrix computation
    - Trajectory tracking
    - Temporal filtering
    - Adaptive parameters
    - Motion prediction
    """
    
    def __init__(self, camera_matrix, dist_coeffs, config=None):
        """
        Initialize the motion estimator.
        
        Args:
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients
            config: Optional MotionConfig object
        """
        # Configuration
        self.config = config if config is not None else MotionConfig()
        
        # Camera parameters
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Feature matching
        self.matcher = create_matcher(self.config.matching_method, self.config)
        
        # Motion components
        self.trajectory = TrajectoryTracker(self.config)
        self.spatial_filter = SpatialConsistencyFilter(self.config)
        self.temporal_filter = TemporalFilter(self.config)
        self.motion_model = MotionModel(self.config)
        self.adaptive_params = AdaptiveParamManager(self.config)
        self.keyframe_manager = KeyframeManager(self.config)
        
        # Current frame data
        self.prev_frame = None
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.frame_id = 0
        
        # Results from last processing step
        self.current_R = np.eye(3)
        self.current_t = np.zeros((3, 1))
        self.current_position = np.zeros(3)
        self.tracking_quality = 1.0
        self.matched_kp1 = None
        self.matched_kp2 = None
        self.last_E = None
        self.last_timestamp = None
        
        # Statistics
        self.match_count = 0
        self.inlier_count = 0
        self.consecutive_failures = 0
        self.timing = {}
    
    def reset(self):
        """Reset the motion estimator state"""
        self.prev_frame = None
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.frame_id = 0
        self.current_R = np.eye(3)
        self.current_t = np.zeros((3, 1))
        self.current_position = np.zeros(3)
        self.last_E = None
        self.last_timestamp = None
        self.trajectory.reset()
        self.temporal_filter.reset()
        self.keyframe_manager.clear()
        self.consecutive_failures = 0
    
    def process_frame(self, frame, keypoints, descriptors, timestamp=None, gray=None):
        """
        Process a new frame to estimate motion.
        
        Args:
            frame: Input image
            keypoints: Detected keypoints
            descriptors: Feature descriptors
            timestamp: Capture timestamp (optional)
            gray: Grayscale image (optional)
            
        Returns:
            success: Whether motion was successfully estimated
            R: Rotation matrix
            t: Translation vector
            position: Current position
        """
        # Timing
        start_time = time.time()
        
        # Increment frame ID
        self.frame_id += 1
        
        # Create grayscale image if not provided
        if gray is None and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # First frame initialization
        if self.prev_frame is None:
            self.prev_frame = frame.copy() if frame is not None else None
            self.prev_gray = gray.copy() if gray is not None else None
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.last_timestamp = timestamp
            
            # Create initial keyframe
            if self.config.use_keyframes:
                self.keyframe_manager.add_keyframe(
                    self.frame_id, timestamp, gray.copy() if gray is not None else None,
                    keypoints, descriptors, self.current_position, self.current_R
                )
            
            return True, np.eye(3), np.zeros((3, 1)), np.zeros(3)
        
        # -------------------------
        # 1. Match features between frames
        # -------------------------
        match_start = time.time()
        
        matches, matched_kp1, matched_kp2 = self.matcher.match(
            self.prev_keypoints, self.prev_descriptors,
            keypoints, descriptors,
            self.prev_gray, gray
        )
        
        self.match_count = len(matched_kp1)
        self.timing['matching'] = time.time() - match_start
        
        # Check if we have enough matches
        if len(matched_kp1) < self.config.min_matches:
            # Not enough matches - try to recover using motion model
            self._handle_tracking_failure(timestamp)
            return False, self.current_R, self.current_t, self.current_position
        
        # Store matched keypoints
        self.matched_kp1 = matched_kp1
        self.matched_kp2 = matched_kp2
        
        # -------------------------
        # 2. Apply spatial consistency filtering
        # -------------------------
        filter_start = time.time()
        
        spatial_mask, filtered_kp1, filtered_kp2 = self.spatial_filter.filter_matches(
            matched_kp1, matched_kp2
        )
        
        if len(filtered_kp1) < 5:
            # Not enough consistent matches
            self._handle_tracking_failure(timestamp)
            return False, self.current_R, self.current_t, self.current_position
        
        self.timing['spatial_filter'] = time.time() - filter_start
        
        # -------------------------
        # 3. Compute essential matrix
        # -------------------------
        essential_start = time.time()
        
        E, mask, inlier_kp1, inlier_kp2 = self._compute_essential_matrix(
            filtered_kp1, filtered_kp2
        )
        
        if E is None or len(inlier_kp1) < 5:
            # Failed to compute essential matrix
            self._handle_tracking_failure(timestamp)
            return False, self.current_R, self.current_t, self.current_position
        
        self.last_E = E
        self.timing['essential'] = time.time() - essential_start
        
        # -------------------------
        # 4. Recover pose (rotation and translation)
        # -------------------------
        pose_start = time.time()
        
        R, t, success, pose_mask = self._estimate_motion(E, inlier_kp1, inlier_kp2)
        
        if not success:
            # Failed to recover pose
            self._handle_tracking_failure(timestamp)
            return False, self.current_R, self.current_t, self.current_position
        
        # Count inliers from pose recovery
        pose_inliers = np.sum(pose_mask) if pose_mask is not None else 0
        self.inlier_count = pose_inliers
        
        # Reset consecutive failures counter
        self.consecutive_failures = 0
        
        self.timing['pose'] = time.time() - pose_start
        
        # -------------------------
        # 5. Update trajectory
        # -------------------------
        traj_start = time.time()
        
        position = self.trajectory.update(R, t, timestamp)
        
        self.timing['trajectory'] = time.time() - traj_start
        
        # -------------------------
        # 6. Calculate tracking quality
        # -------------------------
        quality_start = time.time()
        
        # Calculate tracking quality metric
        self.tracking_quality = self._calculate_tracking_quality(
            matches, self.match_count, pose_inliers
        )
        
        # Update adaptive parameters
        adaptive_params = self.adaptive_params.update(
            self.match_count, 
            pose_inliers,
            self.tracking_quality
        )
        
        self.timing['quality'] = time.time() - quality_start
        
        # -------------------------
        # 7. Apply temporal filtering
        # -------------------------
        filter_start = time.time()
        
        filtered_R, filtered_t, filtered_position = self.temporal_filter.filter(
            R, t, position, timestamp, self.tracking_quality
        )
        
        self.timing['temporal_filter'] = time.time() - filter_start
        
        # -------------------------
        # 8. Keyframe management
        # -------------------------
        keyframe_start = time.time()
        
        if self.config.use_keyframes:
            if self.keyframe_manager.should_create_keyframe(
                filtered_position, filtered_R, 
                len(keypoints), timestamp
            ):
                self.keyframe_manager.add_keyframe(
                    self.frame_id, timestamp, gray.copy() if gray is not None else None,
                    keypoints, descriptors, filtered_position, filtered_R
                )
        
        self.timing['keyframe'] = time.time() - keyframe_start
        
        # Update current state
        self.current_R = filtered_R
        self.current_t = filtered_t
        self.current_position = filtered_position
        
        # Update previous frame data
        self.prev_frame = frame.copy() if frame is not None else None
        self.prev_gray = gray.copy() if gray is not None else None
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.last_timestamp = timestamp
        
        # Total timing
        self.timing['total'] = time.time() - start_time
        
        return True, filtered_R, filtered_t, filtered_position
    
    def _handle_tracking_failure(self, timestamp):
        """
        Handle tracking failure by using the motion model to predict motion.
        
        Args:
            timestamp: Current timestamp
        """
        self.consecutive_failures += 1
        
        # Use motion model to predict motion
        if self.consecutive_failures <= self.config.max_consecutive_failures:
            # Get motion history
            rotations, translations, timestamps = self.trajectory.get_motion_history()
            
            if rotations and translations:
                # Predict motion
                R_pred, t_pred = self.motion_model.predict(
                    rotations, translations, timestamps, 
                    tracking_quality=max(0.2, self.tracking_quality - 0.1 * self.consecutive_failures)
                )
                
                if R_pred is not None and t_pred is not None:
                    # Update trajectory with predicted motion
                    position = self.trajectory.update(R_pred, t_pred, timestamp)
                    
                    # Apply temporal filtering with low confidence
                    reduced_quality = max(0.1, self.tracking_quality / (self.consecutive_failures + 1))
                    filtered_R, filtered_t, filtered_position = self.temporal_filter.filter(
                        R_pred, t_pred, position, timestamp, reduced_quality
                    )
                    
                    # Update current state
                    self.current_R = filtered_R
                    self.current_t = filtered_t
                    self.current_position = filtered_position
                    
                    # Reset match counts for this frame
                    self.match_count = 0
                    self.inlier_count = 0
                    return
        
        # If we reach here, we're out of recovery options - maintain last pose
        pass
    
    def _compute_essential_matrix(self, pts1, pts2):
        """
        Compute essential matrix between matched points.
        
        Args:
            pts1: Matched points from first frame
            pts2: Matched points from second frame
            
        Returns:
            E: Essential matrix
            mask: Inlier mask
            inlier_pts1: Inlier points from first frame
            inlier_pts2: Inlier points from second frame
        """
        # Check if we have enough points
        if len(pts1) < 5 or len(pts2) < 5:
            return None, None, np.array([]), np.array([])
        
        # Get RANSAC threshold from adaptive params if available
        ransac_threshold = self.config.ransac_threshold
        if self.config.enable_adaptive_parameters:
            ransac_threshold = self.adaptive_params.current_ransac_threshold
        
        # Compute essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.camera_matrix,
            method=cv2.RANSAC,
            prob=self.config.confidence,
            threshold=ransac_threshold
        )
        
        # Check if E is valid
        if E is None or E.shape != (3, 3):
            return None, None, np.array([]), np.array([])
        
        # Get inliers based on mask
        if mask is not None:
            inlier_pts1 = pts1[mask.ravel() == 1]
            inlier_pts2 = pts2[mask.ravel() == 1]
            
            # Ensure we still have enough inliers
            if len(inlier_pts1) < 5:
                return None, None, np.array([]), np.array([])
                
            return E, mask, inlier_pts1, inlier_pts2
        
        return E, None, pts1, pts2
    
    def _estimate_motion(self, E, matched_kp1, matched_kp2):
        """
        Recover the rotation and translation from the essential matrix.
        
        Args:
            E: Essential matrix
            matched_kp1: Inlier keypoints from frame 1
            matched_kp2: Inlier keypoints from frame 2
            
        Returns:
            R: Rotation matrix
            t: Translation vector
            success: Whether the decomposition was successful
            mask: Inlier mask from pose recovery
        """
        # Check if essential matrix or matches are None or insufficient
        if E is None or len(matched_kp1) < 5 or len(matched_kp2) < 5:
            return np.eye(3), np.zeros((3, 1)), False, None
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(
            E, matched_kp1, matched_kp2, self.camera_matrix, mask=None
        )
        
        # Check if reconstruction is valid
        if R is None or t is None:
            return np.eye(3), np.zeros((3, 1)), False, None
        
        # Ensure R is a proper rotation matrix (det=1)
        R = orthogonalize_rotation(R)
        
        # Ensure t is a unit vector
        t_norm = np.linalg.norm(t)
        if t_norm > 0:
            t = t / t_norm
        
        return R, t, True, mask
    
    def _calculate_tracking_quality(self, matches, match_count, inlier_count):
        """
        Calculate tracking quality metric.
        
        Args:
            matches: List of matches
            match_count: Number of matches
            inlier_count: Number of inliers
            
        Returns:
            quality: Tracking quality metric (0-1)
        """
        # Avoid division by zero
        if match_count == 0:
            return 0.0
            
        # Inlier ratio component
        inlier_ratio = inlier_count / max(1, match_count)
        
        # Match count component
        match_quality = min(1.0, match_count / 100.0)
        
        # Combine components with higher weight on inlier ratio
        quality = 0.7 * inlier_ratio + 0.3 * match_quality
        
        return quality
    
    def predict_motion(self, timestamp=None):
        """
        Predict motion based on previous motion history.
        
        Args:
            timestamp: Current timestamp (optional)
            
        Returns:
            R_pred: Predicted rotation matrix
            t_pred: Predicted translation vector
        """
        # Get motion history
        rotations, translations, timestamps = self.trajectory.get_motion_history()
        
        # Use motion model to predict
        R_pred, t_pred = self.motion_model.predict(
            rotations, translations, timestamps, 
            tracking_quality=self.tracking_quality
        )
        
        return R_pred, t_pred
    
    def get_motion_stats(self):
        """
        Get motion estimation statistics.
        
        Returns:
            stats: Dictionary of statistics
        """
        # Get current rotation matrix as flattened list
        R_flat = self.current_R.flatten().tolist() if self.current_R is not None else [0] * 9
        
        # Get current translation vector as flattened list
        t_flat = self.current_t.flatten().tolist() if self.current_t is not None else [0] * 3
        
        stats = {
            'match_count': self.match_count,
            'inlier_count': self.inlier_count,
            'position': self.current_position.tolist(),
            'rotation': R_flat,
            'translation': t_flat,
            'tracking_quality': self.tracking_quality,
            'consecutive_failures': self.consecutive_failures,
            'timing': self.timing
        }
        
        return stats
    
    def get_trajectory(self, max_points=None):
        """
        Get the trajectory points.
        
        Args:
            max_points: Maximum number of points to return (None for all)
            
        Returns:
            trajectory: Array of position points (Nx3)
        """
        return self.trajectory.get_trajectory(max_points)
    
    def get_keyframes(self):
        """Get all keyframes"""
        return self.keyframe_manager.get_keyframes() 