import cv2
import numpy as np
import time
import os
import json
from datetime import datetime

class VisualProcessor:
    """
    Handles image processing and feature detection for monocular SLAM.
    Implements ORB feature detection with grid-based distribution to ensure 
    features are well-distributed across the image, which is critical for 
    robust visual odometry and SLAM.
    """
    
    def __init__(self, max_features=300, scale_factor=1.08, nlevels=3, 
                 fast_threshold=10, first_level=0, score_type=cv2.ORB_FAST_SCORE,
                 grid_size=4, features_per_grid=None, adaptive_threshold=True):
        """
        Initialize the visual processor with ORB feature detection.
        
        Args:
            max_features: Maximum number of features to detect
            scale_factor: Pyramid scale factor for multi-scale detection
            nlevels: Number of pyramid levels
            fast_threshold: FAST detector threshold
            first_level: First pyramid level
            score_type: Feature scoring type (Harris or FAST)
            grid_size: Size of grid for feature distribution (e.g., 4 means 4x4 grid)
            features_per_grid: Features per grid cell (None for automatic calculation)
            adaptive_threshold: Whether to adaptively adjust FAST threshold to maintain feature count
        """
        # Initialize ORB feature detector with tunable parameters
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=scale_factor,
            nlevels=nlevels,
            fastThreshold=fast_threshold,
            firstLevel=first_level,
            scoreType=score_type
        )
        
        # Grid-based feature extraction parameters
        self.grid_size = grid_size
        self.features_per_grid = features_per_grid
        if self.features_per_grid is None:
            # If not specified, calculate based on max_features
            self.features_per_grid = max(1, max_features // (grid_size * grid_size))
        
        # Adaptive threshold parameters
        self.adaptive_threshold = adaptive_threshold
        self.min_threshold = 3  # Lower minimum threshold for challenging scenes
        self.max_threshold = 40
        self.target_features_ratio = 0.85  # Increased target ratio for more consistent features
        
        # Store configuration for reference
        self.config = {
            'max_features': max_features,
            'scale_factor': scale_factor,
            'nlevels': nlevels,
            'fast_threshold': fast_threshold,
            'first_level': first_level,
            'score_type': score_type,
            'grid_size': grid_size,
            'features_per_grid': self.features_per_grid,
            'adaptive_threshold': adaptive_threshold
        }
        
        # Performance tracking
        self.processing_times = []
        self.max_times = 30  # Store last 30 processing times
        
        # Feature statistics
        self.feature_counts = []
        self.max_counts = 100  # Store last 100 feature counts
        
        # Data collection for tuning
        self.session_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'config': self.config,
            'frames': []
        }
        
        # Frame counter
        self.frame_counter = 0
        
        # Current FAST threshold (may be adjusted if adaptive_threshold is True)
        self.current_threshold = fast_threshold
        
        # Preprocessing options
        self.use_clahe = True
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Create crisis recovery mode flags
        self.crisis_mode = False
        self.consecutive_low_features = 0
        self.low_feature_threshold = 100
        
        print(f"VisualProcessor initialized with {max_features} max features, {grid_size}x{grid_size} grid")
    
    def detect_features(self, frame, collect_data=False):
        """
        Detect ORB features in the given frame using grid-based distribution.
        
        Args:
            frame: Input image frame
            collect_data: Whether to collect data for this frame
            
        Returns:
            keypoints: Detected keypoints
            descriptors: Feature descriptors
        """
        start_time = time.time()
        
        # Ensure frame is grayscale
        if len(frame.shape) > 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply preprocessing to improve feature detection in challenging scenes
        gray = self._preprocess_image(gray)
        
        # Get frame dimensions
        height, width = gray.shape
        
        # Create grid cells
        cell_h = height // self.grid_size
        cell_w = width // self.grid_size
        
        # Detect features in each grid cell and merge
        all_keypoints = []
        
        # Check if we're in crisis mode (very few features detected in previous frames)
        if self.crisis_mode:
            # Temporarily use more aggressive parameters to find any features
            temp_orb = cv2.ORB_create(
                nfeatures=self.config['max_features'],
                scaleFactor=1.05,  # Smaller scale factor
                nlevels=5,  # More levels
                fastThreshold=max(2, self.current_threshold - 5),  # Lower threshold
                firstLevel=0,
                scoreType=self.config['score_type']
            )
            # Detect across the entire image first to find any features
            crisis_keypoints = temp_orb.detect(gray, None)
            if crisis_keypoints and len(crisis_keypoints) > self.low_feature_threshold:
                # Successfully found more features, exit crisis mode
                self.crisis_mode = False
                self.consecutive_low_features = 0
                # Sort crisis keypoints by response and limit to max features
                crisis_keypoints = sorted(crisis_keypoints, key=lambda x: x.response, reverse=True)
                all_keypoints = crisis_keypoints[:self.config['max_features']]
            # Continue with normal grid detection as backup
        
        # If not in crisis mode or crisis detection didn't yield enough features
        if not self.crisis_mode or not all_keypoints:
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Define cell boundaries
                    x_start = j * cell_w
                    y_start = i * cell_h
                    x_end = min((j + 1) * cell_w, width)
                    y_end = min((i + 1) * cell_h, height)
                    
                    # Skip cells that are too small
                    if x_end - x_start < 10 or y_end - y_start < 10:
                        continue
                    
                    # Create mask for the current cell
                    cell_mask = np.zeros(gray.shape, dtype=np.uint8)
                    cell_mask[y_start:y_end, x_start:x_end] = 255
                    
                    # Detect keypoints in this cell
                    cell_keypoints = self.orb.detect(gray, cell_mask)
                    
                    # Sort by response and take the best ones
                    if cell_keypoints:
                        cell_keypoints = sorted(cell_keypoints, key=lambda x: x.response, reverse=True)
                        cell_keypoints = cell_keypoints[:self.features_per_grid]
                        all_keypoints.extend(cell_keypoints)
        
        # Compute descriptors for all keypoints
        all_keypoints, descriptors = self.orb.compute(gray, all_keypoints)
        
        # Check if we need to enter crisis mode
        if all_keypoints is None or len(all_keypoints) < self.low_feature_threshold:
            self.consecutive_low_features += 1
            if self.consecutive_low_features >= 3:
                self.crisis_mode = True
        else:
            self.consecutive_low_features = 0
            self.crisis_mode = False
        
        # Adaptive FAST threshold adjustment
        if self.adaptive_threshold and len(self.feature_counts) > 5:
            desired_features = int(self.config['max_features'] * self.target_features_ratio)
            current_features = len(all_keypoints) if all_keypoints is not None else 0
            
            # More aggressive threshold adjustment
            if current_features < desired_features * 0.7:
                # Too few features, decrease threshold more aggressively
                self.current_threshold = max(self.min_threshold, self.current_threshold - 2)
                # Recreate ORB detector with new threshold
                self._update_orb_detector()
            elif current_features < desired_features * 0.9:
                # Slightly too few features, decrease threshold slightly
                self.current_threshold = max(self.min_threshold, self.current_threshold - 1)
                # Recreate ORB detector with new threshold
                self._update_orb_detector()
            elif current_features > desired_features * 1.3:
                # Too many features, increase threshold more aggressively
                self.current_threshold = min(self.max_threshold, self.current_threshold + 2)
                # Recreate ORB detector with new threshold
                self._update_orb_detector()
            elif current_features > desired_features * 1.1:
                # Slightly too many features, increase threshold slightly
                self.current_threshold = min(self.max_threshold, self.current_threshold + 1)
                # Recreate ORB detector with new threshold
                self._update_orb_detector()
        
        # Track performance
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.max_times:
            self.processing_times.pop(0)
        
        # Track feature counts
        feature_count = len(all_keypoints) if all_keypoints is not None else 0
        self.feature_counts.append(feature_count)
        if len(self.feature_counts) > self.max_counts:
            self.feature_counts.pop(0)
        
        # Collect frame data if requested
        if collect_data:
            self._collect_frame_data(gray, all_keypoints, processing_time)
            
        self.frame_counter += 1
        
        return all_keypoints, descriptors
    
    def _preprocess_image(self, gray):
        """Apply preprocessing to improve feature detection in challenging scenes"""
        # Apply CLAHE for contrast enhancement if enabled
        if self.use_clahe:
            gray = self.clahe.apply(gray)
        
        # Additional preprocessing based on image statistics
        # Check if image has low contrast
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)
        if max_val - min_val < 50:  # Low contrast image
            # Enhance contrast
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Check if the image is too dark
        mean_val = np.mean(gray)
        if mean_val < 60:  # Dark image
            # Brighten the image
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        
        return gray
    
    def _update_orb_detector(self):
        """Update the ORB detector with the current threshold"""
        self.orb = cv2.ORB_create(
            nfeatures=self.config['max_features'],
            scaleFactor=self.config['scale_factor'],
            nlevels=self.config['nlevels'],
            fastThreshold=self.current_threshold,
            firstLevel=self.config['first_level'],
            scoreType=self.config['score_type']
        )
    
    def _collect_frame_data(self, frame, keypoints, processing_time):
        """Collect data about the processed frame for later analysis."""
        # Only collect every 10th frame to avoid excessive data
        if self.frame_counter % 10 != 0:
            return
            
        # Convert keypoints to serializable format
        kp_data = []
        if keypoints is not None:
            for kp in keypoints:
                kp_data.append({
                    'x': kp.pt[0],
                    'y': kp.pt[1],
                    'size': kp.size,
                    'angle': kp.angle,
                    'response': kp.response,
                    'octave': kp.octave
                })
        
        # Calculate feature distribution (divide image into 4x4 grid)
        height, width = frame.shape
        cell_width = width // 4
        cell_height = height // 4
        
        grid_distribution = np.zeros((4, 4), dtype=np.int32)
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            grid_x = min(x // cell_width, 3)
            grid_y = min(y // cell_height, 3)
            grid_distribution[grid_y, grid_x] += 1
        
        # Store frame data
        frame_data = {
            'frame_number': self.frame_counter,
            'processing_time': processing_time,
            'feature_count': len(keypoints) if keypoints is not None else 0,
            'grid_distribution': grid_distribution.tolist(),
            'keypoints_sample': kp_data[:10],  # Store just a sample of keypoints for analysis
            'current_threshold': self.current_threshold,
            'crisis_mode': self.crisis_mode
        }
        
        self.session_data['frames'].append(frame_data)
    
    def get_processing_fps(self):
        """Calculate current processing FPS based on recent frames."""
        if not self.processing_times:
            return 0.0
        avg_time = sum(self.processing_times) / len(self.processing_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_avg_feature_count(self):
        """Get the average number of features detected in recent frames."""
        if not self.feature_counts:
            return 0
        return sum(self.feature_counts) / len(self.feature_counts)
    
    def get_feature_stats(self):
        """Get statistics about feature detection."""
        if not self.feature_counts:
            return {'min': 0, 'max': 0, 'avg': 0, 'current': 0}
            
        return {
            'min': min(self.feature_counts),
            'max': max(self.feature_counts),
            'avg': sum(self.feature_counts) / len(self.feature_counts),
            'current': self.feature_counts[-1]
        }
    
    def draw_features(self, frame, keypoints):
        """
        Draw detected features on the frame for visualization.
        
        Args:
            frame: Input image frame
            keypoints: Detected keypoints
            
        Returns:
            Visualization frame with drawn keypoints
        """
        vis_frame = frame.copy()
        
        # If in crisis mode, draw a warning
        if self.crisis_mode:
            cv2.putText(vis_frame, "LOW FEATURES - RECOVERY MODE", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return cv2.drawKeypoints(vis_frame, keypoints, None, color=(0, 255, 0), 
                                flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    
    def draw_grid(self, frame):
        """
        Draw the grid used for feature distribution.
        
        Args:
            frame: Input image frame
            
        Returns:
            Frame with grid overlay
        """
        h, w = frame.shape[:2]
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size
        
        # Create a copy of the frame
        vis_frame = frame.copy()
        
        # Draw horizontal lines
        for i in range(1, self.grid_size):
            y = i * cell_h
            cv2.line(vis_frame, (0, y), (w, y), (0, 255, 0), 1)
        
        # Draw vertical lines
        for j in range(1, self.grid_size):
            x = j * cell_w
            cv2.line(vis_frame, (x, 0), (x, h), (0, 255, 0), 1)
        
        return vis_frame
    
    def draw_feature_stats(self, frame):
        """
        Draw feature statistics on the frame.
        
        Args:
            frame: Input visualization frame
            
        Returns:
            Frame with statistics drawn
        """
        stats = self.get_feature_stats()
        fps = self.get_processing_fps()
        
        # Add statistics text
        cv2.putText(frame, f"Features: {int(stats['current'])}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Avg: {int(stats['avg'])}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add threshold info if using adaptive threshold
        if self.adaptive_threshold:
            cv2.putText(frame, f"Threshold: {self.current_threshold}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add crisis mode info if active
            if self.crisis_mode:
                cv2.putText(frame, "RECOVERY MODE", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw feature count history as a small graph at the bottom
        if len(self.feature_counts) > 1:
            max_count = max(self.feature_counts)
            if max_count > 0:
                graph_width = 200
                graph_height = 40
                graph_x = frame.shape[1] - graph_width - 10
                graph_y = frame.shape[0] - graph_height - 10
                
                # Draw background
                cv2.rectangle(frame, (graph_x, graph_y), 
                             (graph_x + graph_width, graph_y + graph_height), 
                             (0, 0, 0), -1)
                
                # Draw graph
                points = []
                for i, count in enumerate(self.feature_counts[-graph_width:]):
                    x = graph_x + i * graph_width // min(len(self.feature_counts), graph_width)
                    y = graph_y + graph_height - int(count * graph_height / max_count)
                    points.append((x, y))
                
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (0, 255, 0), 1)
        
        return frame
    
    def save_data(self, filename="visual_processor_data.json"):
        """
        Save collected data to a JSON file for later analysis.
        
        Args:
            filename: Output JSON filename
            
        Returns:
            bool: True if save was successful
        """
        try:
            # Add summary statistics
            if self.feature_counts:
                self.session_data['summary'] = {
                    'avg_feature_count': sum(self.feature_counts) / len(self.feature_counts),
                    'min_feature_count': min(self.feature_counts),
                    'max_feature_count': max(self.feature_counts),
                    'avg_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
                    'estimated_fps': self.get_processing_fps(),
                    'total_frames': self.frame_counter,
                    'final_threshold': self.current_threshold
                }
            
            # Create directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Save to file
            filepath = os.path.join('data', filename)
            with open(filepath, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            
            print(f"Data saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def update_parameters(self, **kwargs):
        """
        Update ORB detector parameters dynamically.
        
        Args:
            **kwargs: Parameters to update (max_features, scale_factor, etc.)
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Update preprocessing options if provided
            if 'use_clahe' in kwargs:
                self.use_clahe = kwargs['use_clahe']
            
            # Update crisis mode parameters if provided
            if 'low_feature_threshold' in kwargs:
                self.low_feature_threshold = kwargs['low_feature_threshold']
            
            # Update grid parameters if provided
            if 'grid_size' in kwargs:
                self.grid_size = kwargs['grid_size']
            
            if 'features_per_grid' in kwargs:
                self.features_per_grid = kwargs['features_per_grid']
            
            # Create new ORB detector with updated parameters
            max_features = kwargs.get('max_features', self.config['max_features'])
            scale_factor = kwargs.get('scale_factor', self.config['scale_factor'])
            nlevels = kwargs.get('nlevels', self.config['nlevels'])
            fast_threshold = kwargs.get('fast_threshold', self.config['fast_threshold'])
            first_level = kwargs.get('first_level', self.config['first_level'])
            score_type = kwargs.get('score_type', self.config['score_type'])
            
            self.orb = cv2.ORB_create(
                nfeatures=max_features,
                scaleFactor=scale_factor,
                nlevels=nlevels,
                fastThreshold=fast_threshold,
                firstLevel=first_level,
                scoreType=score_type
            )
            
            # Update current threshold if fast_threshold is provided
            if 'fast_threshold' in kwargs:
                self.current_threshold = fast_threshold
            
            # Update configuration
            self.config.update(kwargs)
            
            # Save parameters to session data
            self.session_data['config_updates'] = self.session_data.get('config_updates', [])
            self.session_data['config_updates'].append({
                'frame': self.frame_counter,
                'new_config': self.config.copy()
            })
            
            print(f"Parameters updated: {kwargs}")
            return True
            
        except Exception as e:
            print(f"Error updating parameters: {e}")
            return False
