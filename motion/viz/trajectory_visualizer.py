"""
2D trajectory visualization for motion estimation results.

This module provides a 2D visualization of camera trajectory and motion vectors.
"""

import cv2
import numpy as np
import csv
import os
from datetime import datetime
import time

from .base_visualizer import BaseVisualizer
from ..core.config import MotionConfig

class TrajectoryVisualizer(BaseVisualizer):
    """
    2D real-time visualization of camera motion from MonoSLAM system using OpenCV.
    Shows trajectory and motion vectors with recording capabilities.
    """
    
    def __init__(self, window_name="Trajectory Visualization", size=(800, 600), 
                background_color=(255, 255, 255), trajectory_color=(0, 0, 255),
                current_position_color=(0, 0, 255), start_position_color=(0, 255, 0),
                motion_vector_color=(255, 0, 0), grid_color=(200, 200, 200),
                text_color=(0, 0, 0), keyframe_color=(255, 255, 0)):
        """
        Initialize the trajectory visualizer.
        
        Args:
            window_name: Name of the visualization window
            size: Size of the visualization window (width, height)
            background_color: Background color (BGR)
            trajectory_color: Color for trajectory line (BGR)
            current_position_color: Color for current position marker (BGR)
            start_position_color: Color for start position marker (BGR)
            motion_vector_color: Color for motion vectors (BGR)
            grid_color: Color for grid lines (BGR)
            text_color: Color for text overlay (BGR)
            keyframe_color: Color for keyframe markers (BGR)
        """
        super().__init__(window_name, size)
        
        # Colors in BGR format
        self.colors = {
            'background': background_color,   
            'trajectory': trajectory_color,
            'current_pos': current_position_color,
            'start_pos': start_position_color,
            'motion_vector': motion_vector_color,
            'grid': grid_color,
            'text': text_color,
            'keyframe': keyframe_color
        }
        
        # Extend key handlers
        self.key_handlers.update({
            ord('m'): 'toggle_motion_vectors',
            ord('a'): 'toggle_auto_scale',
            ord('+'): 'zoom_in',
            ord('-'): 'zoom_out',
            ord('g'): 'toggle_grid'
        })
        
        # Initialize display settings
        self.text_color = text_color
        self.scale_factor = 50.0  # Scale factor for converting world to pixel coordinates
        self.center_offset = [size[0] // 2, size[1] // 2]  # Center of the canvas
        self.motion_scale = 10.0  # Scale factor for motion vectors
        self.show_motion_vectors = True
        self.auto_scale = True
        self.show_grid = True
        
        # Trajectory data
        self.trajectory = []  # List of (x, y, z) points
        self.keyframe_positions = []  # List of keyframe positions
        self.latest_rotation = None
        self.latest_translation = None
        
        # Recording variables
        self.record_file = None
        self.csv_writer = None
        
        # Create mouse callback
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Create initial canvas
        self.image = self.create_image(size, self.colors['background'])
        self._draw_grid()
    
    def _world_to_pixel(self, world_point):
        """
        Convert world coordinates to pixel coordinates.
        
        Args:
            world_point: (x, y) or (x, y, z) in world coordinates
            
        Returns:
            (x, y) in pixel coordinates
        """
        x = int(world_point[0] * self.scale_factor) + self.center_offset[0]
        y = self.center_offset[1] - int(world_point[1] * self.scale_factor)  # Flip Y for screen coordinates
        return (x, y)
    
    def _draw_grid(self):
        """Draw a grid on the background for better spatial reference."""
        # Reset the canvas
        self.image = self.create_image(self.window_size, self.colors['background'])
        
        if not self.show_grid:
            return
            
        # Calculate grid spacing in pixels
        grid_world_spacing = 0.5  # How far apart grid lines are in world units
        grid_pixel_spacing = int(grid_world_spacing * self.scale_factor)
        
        # Draw vertical grid lines
        for x in range(self.center_offset[0] % grid_pixel_spacing, 
                      self.window_size[0], 
                      grid_pixel_spacing):
            cv2.line(self.image, (x, 0), (x, self.window_size[1]), 
                    self.colors['grid'], 1)
        
        # Draw horizontal grid lines
        for y in range(self.center_offset[1] % grid_pixel_spacing, 
                      self.window_size[1], 
                      grid_pixel_spacing):
            cv2.line(self.image, (0, y), (self.window_size[0], y), 
                    self.colors['grid'], 1)
        
        # Draw coordinate axes (thicker)
        cv2.line(self.image, (0, self.center_offset[1]), 
                (self.window_size[0], self.center_offset[1]), 
                (0, 0, 0), 2)  # X-axis
        cv2.line(self.image, (self.center_offset[0], 0), 
                (self.center_offset[0], self.window_size[1]), 
                (0, 0, 0), 2)  # Y-axis
    
    def _auto_adjust_scale(self):
        """Automatically adjust scale to fit the trajectory in the window."""
        if not self.trajectory:
            return
            
        # Find the bounding box of the trajectory
        x_coords = [p[0] for p in self.trajectory]
        y_coords = [p[1] for p in self.trajectory]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add margins
        margin = 0.1  # 10% margin
        x_range = max(0.1, (x_max - x_min) * (1 + margin))  # Avoid division by zero
        y_range = max(0.1, (y_max - y_min) * (1 + margin))
        
        # Calculate scale factors for both axes
        x_scale = (self.window_size[0] * 0.8) / x_range
        y_scale = (self.window_size[1] * 0.8) / y_range
        
        # Use the smaller scale factor
        self.scale_factor = min(x_scale, y_scale)
        
        # Adjust center offset to center the trajectory
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        self.center_offset[0] = int(self.window_size[0] / 2 - x_center * self.scale_factor)
        self.center_offset[1] = int(self.window_size[1] / 2 + y_center * self.scale_factor)
    
    def _draw_trajectory(self):
        """Draw the camera trajectory on the canvas."""
        # Make a copy of the gridded background
        view_image = self.image.copy()
        
        # Draw keyframes if available
        if self.keyframe_positions:
            for kf_pos in self.keyframe_positions:
                kf_pixel = self._world_to_pixel(kf_pos)
                cv2.drawMarker(view_image, kf_pixel, self.colors['keyframe'], 
                              markerType=cv2.MARKER_DIAMOND, markerSize=10, thickness=2)
        
        # Draw trajectory lines if we have at least 2 points
        if len(self.trajectory) >= 2:
            # Convert all trajectory points to pixel coordinates
            pixel_trajectory = [self._world_to_pixel(p) for p in self.trajectory]
            
            # Draw lines connecting the points
            for i in range(1, len(pixel_trajectory)):
                cv2.line(view_image, pixel_trajectory[i-1], pixel_trajectory[i], 
                        self.colors['trajectory'], 2)
        
        # Draw start position (green circle)
        if self.trajectory:
            start_pixel = self._world_to_pixel(self.trajectory[0])
            cv2.circle(view_image, start_pixel, 5, self.colors['start_pos'], -1)
        
        # Draw current position (red circle)
        if self.trajectory:
            current_pixel = self._world_to_pixel(self.trajectory[-1])
            cv2.circle(view_image, current_pixel, 5, self.colors['current_pos'], -1)
            
            # Draw a triangle to represent camera orientation
            if self.latest_rotation is not None:
                # Convert rotation matrix to angle (simplify to 2D rotation)
                angle = np.arctan2(self.latest_rotation[1, 0], self.latest_rotation[0, 0])
                
                # Create triangle points
                triangle_size = 10
                # Front point
                pt1 = (
                    int(current_pixel[0] + triangle_size * np.cos(angle)),
                    int(current_pixel[1] - triangle_size * np.sin(angle))
                )
                # Back-left point
                pt2 = (
                    int(current_pixel[0] + triangle_size * np.cos(angle + 2.5)),
                    int(current_pixel[1] - triangle_size * np.sin(angle + 2.5))
                )
                # Back-right point
                pt3 = (
                    int(current_pixel[0] + triangle_size * np.cos(angle - 2.5)),
                    int(current_pixel[1] - triangle_size * np.sin(angle - 2.5))
                )
                
                # Draw triangle
                camera_triangle = np.array([pt1, pt2, pt3], np.int32)
                cv2.polylines(view_image, [camera_triangle], True, (0, 0, 255), 2)
        
        # Draw motion vectors if enabled
        if self.show_motion_vectors and len(self.trajectory) >= 2 and self.latest_translation is not None:
            # Get the previous position in pixel coordinates
            prev_pixel = self._world_to_pixel(self.trajectory[-2])
            
            # Get the translation vector (scaled for visibility)
            tx, ty = self.latest_translation[0, 0], self.latest_translation[1, 0]
            motion_length = np.sqrt(tx*tx + ty*ty)
            
            # Skip very small movements
            if motion_length > 0.001:
                # Calculate arrow endpoint
                arrow_end = (
                    int(prev_pixel[0] + tx * self.scale_factor * self.motion_scale),
                    int(prev_pixel[1] - ty * self.scale_factor * self.motion_scale)
                )
                
                # Draw the arrow
                cv2.arrowedLine(view_image, prev_pixel, arrow_end, 
                               self.colors['motion_vector'], 2, 
                               tipLength=0.3)
        
        return view_image
    
    def _add_text_overlay(self, image, stats=None):
        """
        Add text overlay with position and statistics.
        
        Args:
            image: Image to add overlay to
            stats: Dictionary with statistics to display
        """
        # Draw position information if we have a trajectory
        if self.trajectory:
            current_pos = self.trajectory[-1]
            pos_text = f"Position: X={current_pos[0]:.2f}, Y={current_pos[1]:.2f}"
            if len(current_pos) > 2:
                pos_text += f", Z={current_pos[2]:.2f}"
                
            self.add_text(image, pos_text, (10, 20))
        else:
            self.add_text(image, "No position data yet", (10, 20))
        
        # Add statistics if provided
        if stats:
            y_offset = 40
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    stat_text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
                    self.add_text(image, stat_text, (10, y_offset))
                    y_offset += 20
        
        # Add scale information
        scale_text = f"Scale: {self.scale_factor:.1f} (Auto: {'ON' if self.auto_scale else 'OFF'})"
        self.add_text(image, scale_text, (10, self.window_size[1] - 30))
        
        # Add mode information
        mode_text = f"Motion vectors: {'ON' if self.show_motion_vectors else 'OFF'}"
        self.add_text(image, mode_text, (10, self.window_size[1] - 50))
        
        # Add recording indicator
        self.add_recording_indicator(image)
        
        # Add controls
        controls_text = "Controls: (r)record, (m)motion vectors, (a)auto-scale, (c)clear, (+/-)zoom, (g)grid, (q)quit"
        self.add_text(image, controls_text, (10, self.window_size[1] - 10))
    
    def start_recording(self, filepath=None):
        """
        Start recording trajectory data to CSV.
        
        Args:
            filepath: Path to save the CSV file, or None for auto-generated filename
        """
        super().start_recording()
        
        # Generate filename if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filepath = os.path.join(output_dir, f"trajectory_{timestamp}.csv")
        
        # Open CSV file for writing
        self.record_file = open(filepath, 'w', newline='')
        self.csv_writer = csv.writer(self.record_file)
        
        # Write header
        self.csv_writer.writerow([
            'Timestamp', 'X', 'Y', 'Z', 
            'RotX1', 'RotX2', 'RotX3',
            'RotY1', 'RotY2', 'RotY3',
            'RotZ1', 'RotZ2', 'RotZ3',
            'TransX', 'TransY', 'TransZ'
        ])
        
        print(f"Recording started: {filepath}")
    
    def stop_recording(self):
        """Stop recording and close the CSV file."""
        super().stop_recording()
        
        if self.record_file:
            self.record_file.close()
            self.record_file = None
            self.csv_writer = None
            print("Recording stopped")
    
    def record_frame(self, position, rotation=None, translation=None, timestamp=None):
        """
        Record a frame to CSV file if recording is active.
        
        Args:
            position: Current position (x, y, z)
            rotation: Current rotation matrix (3x3)
            translation: Current translation vector (3x1)
            timestamp: Current timestamp or None for automatic
        """
        if not self.recording or self.csv_writer is None:
            return
            
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = (datetime.now() - self.record_start_time).total_seconds()
            
        # Prepare row data
        row = [timestamp, position[0], position[1]]
        
        # Add Z if available
        if len(position) > 2:
            row.append(position[2])
        else:
            row.append(0.0)  # Default Z
        
        # Add rotation if available
        if rotation is not None:
            for i in range(3):
                for j in range(3):
                    row.append(rotation[i, j])
        else:
            row.extend([0.0] * 9)  # Default rotation
            
        # Add translation if available
        if translation is not None:
            row.append(translation[0, 0])
            row.append(translation[1, 0])
            row.append(translation[2, 0] if translation.shape[0] > 2 else 0.0)
        else:
            row.extend([0.0] * 3)  # Default translation
            
        # Write row
        self.csv_writer.writerow(row)
    
    def update(self, trajectory=None, current_position=None, R=None, t=None, 
              keyframe_positions=None, stats=None, frame=None):
        """
        Update visualization with new data.
        
        Args:
            trajectory: List of (x, y, z) trajectory points, or None to keep current
            current_position: Current position, or None to use last trajectory point
            R: Current rotation matrix (3x3)
            t: Current translation vector (3x1)
            keyframe_positions: List of keyframe positions
            stats: Dictionary with statistics to display
            frame: Optional camera frame to display (if None, only trajectory is shown)
            
        Returns:
            Key command if a key was pressed (e.g., 'quit', 'record', etc.)
        """
        # Update the trajectory if provided
        if trajectory is not None:
            self.trajectory = trajectory
            
            # Auto-adjust scale if enabled
            if self.auto_scale and trajectory:
                self._auto_adjust_scale()
                self._draw_grid()
                
        # Update current position if provided
        if current_position is not None and (not self.trajectory or 
                                           not np.array_equal(self.trajectory[-1], current_position)):
            if not self.trajectory:
                self.trajectory = [current_position]
            else:
                self.trajectory.append(current_position)
        
        # Update rotation and translation
        if R is not None:
            self.latest_rotation = R
        if t is not None:
            self.latest_translation = t
            
        # Update keyframe positions
        if keyframe_positions is not None:
            self.keyframe_positions = keyframe_positions
            
        # Record the current frame if recording is active
        if self.recording and current_position is not None:
            self.record_frame(current_position, R, t)
            
        # Draw the visualization
        view_image = self._draw_trajectory()
        
        # Add text overlay
        self._add_text_overlay(view_image, stats)
        
        # Show the visualization
        key_command = self.show(view_image)
        
        # Handle key commands
        if key_command:
            if key_command == 'toggle_motion_vectors':
                self.show_motion_vectors = not self.show_motion_vectors
            elif key_command == 'toggle_auto_scale':
                self.auto_scale = not self.auto_scale
            elif key_command == 'toggle_grid':
                self.show_grid = not self.show_grid
                self._draw_grid()
            elif key_command == 'zoom_in':
                self.scale_factor *= 1.2
                self._draw_grid()
            elif key_command == 'zoom_out':
                self.scale_factor /= 1.2
                self._draw_grid()
            elif key_command == 'clear':
                self.trajectory = []
                self.keyframe_positions = []
                self._draw_grid()
            else:
                return self.handle_key(key_command)
                
        return key_command
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:  # Scroll up - zoom in
                self.scale_factor *= 1.1
            else:  # Scroll down - zoom out
                self.scale_factor /= 1.1
            self._draw_grid()
    
    def print_help(self):
        """Print help information to console."""
        print("\nTrajectory Visualizer Controls:")
        print("  'r' - Start/stop recording data to CSV")
        print("  'm' - Toggle motion vectors")
        print("  'a' - Toggle auto-scaling")
        print("  'g' - Toggle grid")
        print("  'c' - Clear trajectory")
        print("  '+' - Zoom in (increase scale)")
        print("  '-' - Zoom out (decrease scale)")
        print("  'q' or ESC - Quit visualization")
        print("  Mouse wheel - Zoom in/out") 