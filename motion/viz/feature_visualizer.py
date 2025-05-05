"""
Feature visualization for motion estimation.

This module provides visualization for feature detection and matching.
"""

import cv2
import numpy as np
from .base_visualizer import BaseVisualizer

class FeatureVisualizer(BaseVisualizer):
    """
    Visualizer for displaying feature detection and matching.
    
    Shows keypoints, matches, and inliers with detailed overlay information.
    """
    
    def __init__(self, window_name="Feature Visualization", size=(800, 600)):
        """
        Initialize the feature visualizer.
        
        Args:
            window_name: Name of the visualization window
            size: Size of the visualization window (width, height)
        """
        super().__init__(window_name, size)
        
        # Colors
        self.colors = {
            'keypoint': (0, 255, 0),     # Green
            'match': (0, 255, 255),      # Yellow
            'inlier': (0, 255, 0),       # Green
            'outlier': (0, 0, 255),      # Red
            'flow': (255, 0, 255),       # Magenta
            'text': (255, 255, 255)      # White
        }
        
        # Set text color
        self.text_color = self.colors['text']
        
        # Visualization settings
        self.show_keypoints = True
        self.show_matches = True
        self.show_flow = True
        self.show_stats = True
        
        # Extend key handlers
        self.key_handlers.update({
            ord('k'): 'toggle_keypoints',
            ord('m'): 'toggle_matches',
            ord('f'): 'toggle_flow',
            ord('s'): 'toggle_stats'
        })
    
    def draw_keypoints(self, image, keypoints, color=None, size=3, thickness=1):
        """
        Draw keypoints on the image.
        
        Args:
            image: Image to draw on
            keypoints: List of keypoints (either cv2.KeyPoint or (x,y) tuples)
            color: Color for the keypoints (BGR)
            size: Size of the keypoint circles
            thickness: Line thickness (-1 for filled)
            
        Returns:
            Image with keypoints drawn
        """
        if not self.show_keypoints or not keypoints:
            return image
            
        if color is None:
            color = self.colors['keypoint']
            
        result = image.copy()
        
        for kp in keypoints:
            if isinstance(kp, cv2.KeyPoint):
                x, y = map(int, kp.pt)
            else:
                x, y = map(int, kp)
                
            cv2.circle(result, (x, y), size, color, thickness)
            
        return result
    
    def draw_matches(self, frame1, kp1, frame2, kp2, matches=None, mask=None, 
                    horizontal=True, match_color=None, inlier_color=None, outlier_color=None):
        """
        Draw matches between two frames.
        
        Args:
            frame1: First frame
            kp1: Keypoints from first frame
            frame2: Second frame
            kp2: Keypoints from second frame
            matches: Optional list of match indices or DMatch objects
            mask: Optional mask for inliers (1) and outliers (0)
            horizontal: If True, frames are arranged horizontally, otherwise vertically
            match_color: Color for matches (if mask is None)
            inlier_color: Color for inlier matches
            outlier_color: Color for outlier matches
            
        Returns:
            Combined image with matches drawn
        """
        if not self.show_matches:
            return frame1.copy()
            
        # Set default colors
        if match_color is None:
            match_color = self.colors['match']
        if inlier_color is None:
            inlier_color = self.colors['inlier']
        if outlier_color is None:
            outlier_color = self.colors['outlier']
            
        # Make sure frames are color
        if len(frame1.shape) == 2:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
        if len(frame2.shape) == 2:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
            
        # Get frame sizes
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Create combined image
        if horizontal:
            width = w1 + w2
            height = max(h1, h2)
            result = np.zeros((height, width, 3), dtype=np.uint8)
            result[:h1, :w1] = frame1
            result[:h2, w1:w1+w2] = frame2
            offset = (w1, 0)
        else:
            width = max(w1, w2)
            height = h1 + h2
            result = np.zeros((height, width, 3), dtype=np.uint8)
            result[:h1, :w1] = frame1
            result[h1:h1+h2, :w2] = frame2
            offset = (0, h1)
            
        # Draw matches if we have them
        if matches is not None and kp1 is not None and kp2 is not None:
            for i, (pt1, pt2) in enumerate(zip(kp1, kp2)):
                # Check if this is a valid match to draw
                if matches is not None and i >= len(matches):
                    continue
                    
                # Convert keypoints to integer coordinates
                if isinstance(pt1, cv2.KeyPoint):
                    pt1 = (int(pt1.pt[0]), int(pt1.pt[1]))
                else:
                    pt1 = (int(pt1[0]), int(pt1[1]))
                    
                if isinstance(pt2, cv2.KeyPoint):
                    pt2 = (int(pt2.pt[0]), int(pt2.pt[1]))
                else:
                    pt2 = (int(pt2[0]), int(pt2[1]))
                
                # Adjust second point with offset
                pt2_offset = (pt2[0] + offset[0], pt2[1] + offset[1])
                
                # Determine line color based on mask
                if mask is not None and i < len(mask) and mask[i]:
                    color = inlier_color  # Inlier
                else:
                    color = outlier_color  # Outlier
                
                # Draw the match line
                cv2.line(result, pt1, pt2_offset, color, 1)
                
                # Draw the matched keypoints
                cv2.circle(result, pt1, 3, color, -1)
                cv2.circle(result, pt2_offset, 3, color, -1)
                
        return result
    
    def draw_optical_flow(self, frame, prev_pts, curr_pts, flow_color=None, line_thickness=1):
        """
        Draw optical flow vectors between matched points.
        
        Args:
            frame: Frame to draw on
            prev_pts: Previous points (Nx2 array or list of (x,y) tuples)
            curr_pts: Current points (Nx2 array or list of (x,y) tuples)
            flow_color: Color for flow lines/arrows
            line_thickness: Thickness of flow lines
            
        Returns:
            Frame with optical flow visualization
        """
        if not self.show_flow or prev_pts is None or curr_pts is None:
            return frame
            
        if flow_color is None:
            flow_color = self.colors['flow']
            
        result = frame.copy()
        
        # Draw flow lines
        for i, (prev, curr) in enumerate(zip(prev_pts, curr_pts)):
            # Convert to integer coordinates
            if isinstance(prev, np.ndarray) and prev.size >= 2:
                p1 = (int(prev[0]), int(prev[1]))
            else:
                p1 = (int(prev[0]), int(prev[1]))
                
            if isinstance(curr, np.ndarray) and curr.size >= 2:
                p2 = (int(curr[0]), int(curr[1]))
            else:
                p2 = (int(curr[0]), int(curr[1]))
            
            # Calculate flow vector length
            flow_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # Only draw if the flow is significant
            if flow_length > 1.0:
                # Draw arrow instead of line for better visualization
                cv2.arrowedLine(result, p1, p2, flow_color, line_thickness, tipLength=0.2)
        
        return result
    
    def add_stats_overlay(self, image, stats=None):
        """
        Add statistics overlay to the image.
        
        Args:
            image: Image to add overlay to
            stats: Dictionary with statistics (counts, ratios, etc.)
            
        Returns:
            Image with statistics overlay
        """
        if not self.show_stats or stats is None:
            return image
            
        result = image.copy()
        
        # Display common statistics
        y_pos = 20
        line_height = 20
        
        # Create translucent background for better text visibility
        overlay = result.copy()
        bg_height = len(stats) * line_height + 10
        cv2.rectangle(overlay, (5, 5), (300, 5 + bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)
        
        # Add stats as text
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    text = f"{key}: {value:.2f}"
                else:
                    text = f"{key}: {value}"
                    
                self.add_text(result, text, (10, y_pos))
                y_pos += line_height
                
        return result
    
    def update(self, current_frame, prev_frame=None, keypoints=None, prev_keypoints=None,
              matches=None, mask=None, stats=None):
        """
        Update the feature visualization.
        
        Args:
            current_frame: Current frame to visualize
            prev_frame: Previous frame (for matches visualization)
            keypoints: Keypoints in current frame
            prev_keypoints: Keypoints in previous frame
            matches: Matches between keypoints
            mask: Mask for inlier/outlier classification
            stats: Statistics to display
            
        Returns:
            Key command if a key was pressed
        """
        # Resize frames if needed
        if current_frame.shape[1] > self.window_size[0] or current_frame.shape[0] > self.window_size[1]:
            scale = min(self.window_size[0] / current_frame.shape[1],
                      self.window_size[1] / current_frame.shape[0])
            dim = (int(current_frame.shape[1] * scale), int(current_frame.shape[0] * scale))
            current_frame = cv2.resize(current_frame, dim)
            
            if prev_frame is not None:
                prev_frame = cv2.resize(prev_frame, dim)
                
        # Create visualization based on available data
        if prev_frame is not None and matches is not None:
            # Draw matches between frames
            view_image = self.draw_matches(
                prev_frame, prev_keypoints, 
                current_frame, keypoints,
                matches, mask
            )
        elif keypoints is not None:
            # Just draw keypoints on current frame
            view_image = self.draw_keypoints(current_frame, keypoints)
        else:
            # Just show the current frame
            view_image = current_frame.copy()
            
        # Add statistics overlay
        if stats:
            view_image = self.add_stats_overlay(view_image, stats)
            
        # Add recording indicator
        self.add_recording_indicator(view_image)
        
        # Show the visualization
        key_command = self.show(view_image)
        
        # Handle key commands
        if key_command:
            if key_command == 'toggle_keypoints':
                self.show_keypoints = not self.show_keypoints
            elif key_command == 'toggle_matches':
                self.show_matches = not self.show_matches
            elif key_command == 'toggle_flow':
                self.show_flow = not self.show_flow
            elif key_command == 'toggle_stats':
                self.show_stats = not self.show_stats
            else:
                return self.handle_key(key_command)
                
        return key_command
    
    def print_help(self):
        """Print help information to console."""
        print("\nFeature Visualizer Controls:")
        print("  'k' - Toggle keypoints")
        print("  'm' - Toggle matches")
        print("  'f' - Toggle optical flow")
        print("  's' - Toggle statistics overlay")
        print("  'r' - Start/stop recording")
        print("  'q' or ESC - Quit visualization") 