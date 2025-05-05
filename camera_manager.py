import cv2
import numpy as np
import time
import os

# Try to import Picamera2 for Raspberry Pi
try:
    from picamera2 import Picamera2
    from picamera2.outputs import Output
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

class CameraManager:
    """
    A simple camera manager for monocular SLAM.
    Handles camera initialization, frame capture, and basic calibration.
    """
    
    def __init__(self, camera_id=0, resolution=(640, 480), fps=30, use_picamera2=None):
        """
        Initialize the camera manager.
        
        Args:
            camera_id: Camera identifier (0 for default camera, or device path)
            resolution: Desired resolution as (width, height)
            fps: Desired frames per second
            use_picamera2: Force use of Picamera2 if True, force OpenCV if False, auto-detect if None
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        
        # Determine if we should use Picamera2
        self.use_picamera2 = use_picamera2
        if use_picamera2 is None:
            self.use_picamera2 = PICAMERA2_AVAILABLE
        
        # Camera objects
        self.cap = None        # OpenCV capture
        self.picam2 = None     # Picamera2 object
        self.config = None     # Picamera2 configuration
        
        # Latest frame from Picamera2
        self.latest_frame = None
        self.latest_frame_time = 0
        
        # Camera calibration parameters (will be loaded from file if available)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_times = 30  # Keep track of last 30 frames for FPS calculation
        
        # Status flags
        self.is_running = False
        self.is_calibrated = False
        
        # Try to load calibration if it exists
        self._load_calibration()
    
    def start(self):
        """
        Start the camera capture.
        
        Returns:
            bool: True if camera started successfully, False otherwise
        """
        try:
            if self.use_picamera2 and PICAMERA2_AVAILABLE:
                # Initialize Picamera2
                self.picam2 = Picamera2()
                
                # Configure camera
                self.config = self.picam2.create_still_configuration(
                    main={"size": self.resolution, "format": "RGB888"},
                    controls={"FrameDurationLimits": (int(1/self.fps * 1000000), 1000000000)}
                )
                self.picam2.configure(self.config)
                
                # Start camera
                self.picam2.start()
                
                # Allow camera to warm up
                time.sleep(2)
                
                print(f"Picamera2 started with resolution {self.resolution} at {self.fps} FPS target")
            else:
                # Initialize OpenCV capture
                self.cap = cv2.VideoCapture(self.camera_id)
                
                # Set resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                
                # Set FPS
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                # Check if camera opened successfully
                if not self.cap.isOpened():
                    print("Error: Could not open camera.")
                    return False
                
                # Warm up camera (some cameras need this)
                for _ in range(5):
                    ret, _ = self.cap.read()
                    if not ret:
                        print("Warning: Failed to grab initial frames.")
                    time.sleep(0.1)
                
                print(f"OpenCV camera started with resolution {self.resolution} at {self.fps} FPS target")
            
            self.is_running = True
            return True
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop(self):
        """Stop camera capture and release resources."""
        if self.use_picamera2 and self.picam2:
            # Close the Picamera2
            self.picam2.close()
            self.picam2 = None
        elif self.cap:
            # Release OpenCV capture
            self.cap.release()
            self.cap = None
            
        self.is_running = False
        print("Camera stopped")
    
    def get_frame(self, grayscale=True):
        """
        Capture and return a frame from the camera.
        
        Args:
            grayscale: Whether to convert the frame to grayscale
        
        Returns:
            frame: The captured frame or None if capture failed
            timestamp: Time when frame was captured
        """
        if not self.is_running:
            print("Camera is not running. Call start() first.")
            return None, None
        
        start_time = time.time()
        frame = None
        
        try:
            # Capture frame
            if self.use_picamera2 and self.picam2:
                # Get a frame from Picamera2
                frame = self.picam2.capture_array()
                
                # Convert from RGB to BGR for OpenCV compatibility
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                # Capture from OpenCV
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    return None, None
            
            # Convert to grayscale if requested (more efficient for feature detection)
            if grayscale and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # If calibrated, undistort the frame
            if self.is_calibrated and self.camera_matrix is not None:
                # Only undistort if we have calibration parameters
                if not grayscale:  # Color frame
                    frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
                else:  # Grayscale frame
                    frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
            # Track frame capture time for FPS calculation
            end_time = time.time()
            self.frame_times.append(end_time - start_time)
            if len(self.frame_times) > self.max_frame_times:
                self.frame_times.pop(0)
            
            return frame, end_time
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None, None
    
    def get_fps(self):
        """
        Calculate current FPS based on recent frame times.
        
        Returns:
            float: Current FPS
        """
        if not self.frame_times:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def _load_calibration(self, filepath="camera_calibration.npz"):
        """
        Load camera calibration parameters from file.
        
        Args:
            filepath: Path to calibration file
            
        Returns:
            bool: True if calibration loaded successfully
        """
        try:
            if os.path.exists(filepath):
                data = np.load(filepath)
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
                self.is_calibrated = True
                print(f"Calibration loaded from {filepath}")
                return True
            else:
                print("No calibration file found. Camera will be used without calibration.")
                # Create identity camera matrix as default
                self.camera_matrix = np.array([
                    [self.resolution[0], 0, self.resolution[0]/2],
                    [0, self.resolution[0], self.resolution[1]/2],
                    [0, 0, 1]
                ], dtype=np.float32)
                self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
                return False
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def save_calibration(self, camera_matrix, dist_coeffs, filepath="camera_calibration.npz"):
        """
        Save camera calibration parameters to file.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            filepath: Path to save calibration file
            
        Returns:
            bool: True if calibration saved successfully
        """
        try:
            np.savez(filepath, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.is_calibrated = True
            print(f"Calibration saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
    
    def calibrate(self, checkerboard_size=(9, 6), num_samples=15):
        """
        Perform camera calibration using a checkerboard pattern.
        
        Args:
            checkerboard_size: Number of internal corners in the checkerboard (width, height)
            num_samples: Number of samples to take for calibration
            
        Returns:
            bool: True if calibration was successful
        """
        if not self.is_running:
            print("Camera is not running. Call start() first.")
            return False
        
        print(f"Starting calibration process. Looking for {checkerboard_size} checkerboard.")
        print("Hold a checkerboard in view of the camera.")
        print(f"Need {num_samples} good samples. Press 'q' to abort.")
        
        # Arrays to store object points and image points
        objpoints = [] # 3D points in real world space
        imgpoints = [] # 2D points in image plane
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        
        samples_collected = 0
        while samples_collected < num_samples:
            # Get a frame
            frame, _ = self.get_frame(grayscale=False)
            if frame is None:
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            
            # Display frame
            display_frame = frame.copy()
            if ret:
                # Refine corners for better accuracy
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Draw the corners
                cv2.drawChessboardCorners(display_frame, checkerboard_size, corners, ret)
                
                # Show info
                cv2.putText(display_frame, f"Sample {samples_collected+1}/{num_samples} - Hold still!", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imshow('Calibration', display_frame)
                key = cv2.waitKey(500)  # Longer pause to make sure board is still
                
                if key == ord('q'):
                    print("Calibration aborted by user")
                    cv2.destroyAllWindows()
                    return False
                
                # Add to our samples
                objpoints.append(objp)
                imgpoints.append(corners)
                samples_collected += 1
                print(f"Sample {samples_collected}/{num_samples} captured")
            else:
                # Show info
                cv2.putText(display_frame, "Looking for checkerboard...", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow('Calibration', display_frame)
            if cv2.waitKey(1) == ord('q'):
                print("Calibration aborted by user")
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        
        # Calculate calibration parameters
        print("Processing calibration data...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        if ret:
            # Save calibration
            self.save_calibration(camera_matrix, dist_coeffs)
            print("Calibration successful!")
            return True
        else:
            print("Calibration failed")
            return False


# Example usage (only runs if script is executed directly)
if __name__ == "__main__":
    # Create camera manager with Picamera2 (best option for Raspberry Pi)
    cam = CameraManager(resolution=(640, 480), fps=30, use_picamera2=True)
    
    # Alternatively, use OpenCV if preferred:
    # cam = CameraManager(use_picamera2=False)  # Force OpenCV
    
    # Start camera
    if cam.start():
        try:
            # Display some frames to test
            for _ in range(100):
                frame, timestamp = cam.get_frame(grayscale=False)
                if frame is not None:
                    # Display FPS
                    fps = cam.get_fps()
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Show the frame
                    cv2.imshow('Camera Test', frame)
                    
                    # Break on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("Failed to get frame")
                    break
                
            print("Basic test complete. Current FPS:", cam.get_fps())
            
            # Ask if user wants to calibrate
            response = input("Do you want to run camera calibration? (y/n): ")
            if response.lower() == 'y':
                cam.calibrate()
        
        finally:
            # Clean up
            cam.stop()
            cv2.destroyAllWindows() 