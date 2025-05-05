#!/usr/bin/env python3
import time
import cv2
import numpy as np

# Try to import Picamera2
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Picamera2 not available")

def main():
    """Test camera capture and diagnose frame format"""
    if not PICAMERA2_AVAILABLE:
        print("Cannot run test: Picamera2 not available")
        return
    
    print("Initializing Picamera2...")
    picam2 = Picamera2()
    
    # Configure camera
    resolution = (640, 480)
    config = picam2.create_still_configuration(
        main={"size": resolution, "format": "RGB888"}
    )
    picam2.configure(config)
    
    # Start camera
    picam2.start()
    print("Camera started, waiting for warm-up...")
    time.sleep(2)
    
    try:
        # Capture a frame
        print("Capturing frame...")
        frame_data = picam2.capture_array()
        
        # Print type and shape
        print(f"Type of returned data: {type(frame_data)}")
        
        if isinstance(frame_data, tuple):
            print(f"Tuple length: {len(frame_data)}")
            for i, item in enumerate(frame_data):
                print(f"Item {i} type: {type(item)}")
                if hasattr(item, 'shape'):
                    print(f"Item {i} shape: {item.shape}")
        elif hasattr(frame_data, 'shape'):
            print(f"Shape: {frame_data.shape}")
        
        # If it's a tuple, extract the actual frame
        if isinstance(frame_data, tuple) and len(frame_data) > 0:
            print("Attempting to extract frame from tuple...")
            # Try first item
            for i, item in enumerate(frame_data):
                if isinstance(item, np.ndarray):
                    print(f"Item {i} appears to be a valid frame")
                    cv2.imwrite("test_frame.jpg", cv2.cvtColor(item, cv2.COLOR_RGB2BGR))
                    print("Saved test image as test_frame.jpg")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Cleaning up...")
        picam2.close()
        print("Camera stopped")

if __name__ == "__main__":
    main() 