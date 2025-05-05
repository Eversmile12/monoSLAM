"""
Base visualizer class for motion visualization.

This module provides a base class for all motion visualization components.
"""

import cv2
import numpy as np
from datetime import datetime

class BaseVisualizer:
    """
    Base class for motion visualization components.
    
    This class provides common functionality for all visualizers, such as:
    - Window creation and management
    - Key handling
    - Recording functionality
    - Text overlay capabilities
    """
    
    def __init__(self, window_name="Motion Visualization", size=(800, 600)):
        """
        Initialize the base visualizer.
        
        Args:
            window_name: Name of the visualization window
            size: Size of the visualization window (width, height)
        """
        self.window_name = window_name
        self.window_size = size
        self.image = None
        
        # Create the window
        cv2.namedWindow(self.window_name)
        
        # Recording state
        self.recording = False
        self.record_start_time = None
        
        # Font settings for text overlay
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.text_thickness = 1
        self.text_color = (255, 255, 255)  # White by default
        
        # Command mapping
        self.key_handlers = {
            ord('q'): 'quit',       # Quit
            27: 'quit',             # ESC key
            ord('r'): 'record',     # Start/stop recording
            ord('c'): 'clear',      # Clear display
            ord('h'): 'help'        # Show help
        }
    
    def create_image(self, size=None, color=(0, 0, 0)):
        """
        Create a new image for visualization.
        
        Args:
            size: Size of the image (width, height)
            color: Background color in BGR format
            
        Returns:
            New image
        """
        if size is None:
            size = self.window_size
        
        # Create a blank image with the specified color
        # Ensure the color is a tuple of integers to avoid any compatibility issues
        color_tuple = tuple([int(c) for c in color])
        
        # Create a properly formatted numpy array for OpenCV
        height, width = size[1], size[0]
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill with the specified color
        if isinstance(color, (list, tuple)) and len(color) == 3:
            img[:] = color_tuple
        else:
            # Default to black if color format is incorrect
            img[:] = (0, 0, 0)
            
        return img
    
    def add_text(self, image, text, position, color=None, scale=None, thickness=None):
        """
        Add text to the visualization image.
        
        Args:
            image: Image to add text to
            text: Text to add
            position: Position (x, y) to place the text
            color: Text color (defaults to self.text_color)
            scale: Font scale (defaults to self.font_scale)
            thickness: Text thickness (defaults to self.text_thickness)
        """
        if color is None:
            color = self.text_color
        if scale is None:
            scale = self.font_scale
        if thickness is None:
            thickness = self.text_thickness
            
        cv2.putText(image, text, position, self.font, scale, color, thickness)
    
    def show(self, image=None):
        """
        Show the visualization image.
        
        Args:
            image: Image to show (if None, uses self.image)
            
        Returns:
            Key command if a key was pressed
        """
        if image is not None:
            self.image = image
            
        if self.image is not None:
            cv2.imshow(self.window_name, self.image)
            
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # A key was pressed
            if key in self.key_handlers:
                return self.key_handlers[key]
        
        return None
    
    def get_image(self):
        """
        Get the current visualization image.
        
        Returns:
            Current image
        """
        return self.image
    
    def start_recording(self):
        """Start recording."""
        self.recording = True
        self.record_start_time = datetime.now()
    
    def stop_recording(self):
        """Stop recording."""
        self.recording = False
        self.record_start_time = None
    
    def toggle_recording(self):
        """Toggle recording state."""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def add_recording_indicator(self, image):
        """
        Add recording indicator to the image.
        
        Args:
            image: Image to add the indicator to
        """
        if self.recording:
            # Calculate recording duration
            elapsed = (datetime.now() - self.record_start_time).total_seconds()
            
            # Add red "REC" text
            rec_text = f"REC {elapsed:.1f}s"
            text_size = cv2.getTextSize(rec_text, self.font, self.font_scale, self.text_thickness)[0]
            rec_pos = (image.shape[1] - text_size[0] - 10, image.shape[0] - 10)
            cv2.putText(image, rec_text, rec_pos, self.font, self.font_scale, 
                       (0, 0, 255), self.text_thickness)
    
    def close(self):
        """Close the visualization window."""
        cv2.destroyWindow(self.window_name)
    
    def handle_key(self, key_command):
        """
        Handle key commands.
        
        Args:
            key_command: Command to handle
            
        Returns:
            True if visualization should exit, False otherwise
        """
        if key_command == 'quit':
            return True
        elif key_command == 'record':
            self.toggle_recording()
        elif key_command == 'help':
            self.print_help()
        
        return False
    
    def print_help(self):
        """Print help information."""
        print(f"\n{self.window_name} Controls:")
        print("  'q' or ESC - Quit visualization")
        print("  'r' - Start/stop recording")
        print("  'c' - Clear display")
        print("  'h' - Show this help message") 