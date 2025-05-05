# MonoSLAM Motion Estimation Module

## Overview

The `motion` module provides a comprehensive visual motion estimation system for monocular cameras. It is designed to be modular, maintainable, and easy to extend with new features. The system uses feature tracking and essential matrix decomposition to estimate 6-DOF camera motion, with robust filtering and outlier rejection.

## Key Features

-   **Modular Architecture**: Separates concerns into specialized components
-   **Multiple Feature Matching Methods**: BruteForce, FLANN, and Optical Flow
-   **Robust Essential Matrix Calculation**: With adaptive RANSAC thresholds
-   **Trajectory Tracking**: Maintains and visualizes camera path
-   **Temporal Filtering**: Smooths motion using adaptive filtering
-   **Spatial Consistency Filtering**: Removes outlier feature matches
-   **Keyframe Management**: Reduces drift in long sequences
-   **Motion Prediction**: Recovers during tracking loss
-   **Adaptive Parameters**: Automatically adjusts parameters based on tracking quality

## Module Structure

The module is organized into specialized subpackages:

```
motion/
├── core/               # Core components
│   ├── config.py       # Configuration parameters
│   ├── matcher.py      # Feature matching
│   ├── motion_estimator.py  # Main motion estimation
│   ├── trajectory.py   # Trajectory tracking
│   ├── keyframe.py     # Keyframe management
│   └── utils.py        # Utility functions
├── filters/            # Filtering components
│   ├── spatial_filter.py    # Spatial consistency filtering
│   ├── temporal_filter.py   # Temporal smoothing
│   └── adaptive_params.py   # Adaptive parameters
├── models/             # Motion prediction models
│   └── motion_model.py      # Motion prediction
└── viz/                # Visualization components
    └── (future visualization tools)
```

## Usage

### Basic Usage

```python
from motion.core import MotionEstimator, MotionConfig
import cv2

# Create configuration
config = MotionConfig()
config.use_keyframes = True
config.enable_adaptive_parameters = True

# Initialize motion estimator with camera parameters
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.zeros(5)
motion_estimator = MotionEstimator(camera_matrix, dist_coeffs, config)

# Process frames
while True:
    # Get new frame and detect features
    frame = camera.get_frame()
    keypoints, descriptors = feature_detector.detectAndCompute(frame, None)

    # Process frame
    success, R, t, position = motion_estimator.process_frame(
        frame, keypoints, descriptors, timestamp
    )

    # Get trajectory for visualization
    trajectory = motion_estimator.get_trajectory()

    # Get statistics
    stats = motion_estimator.get_motion_stats()
```

### Command Line Tool

Use the test_refactored.py script to try the refactored motion estimator:

```bash
python test_refactored.py --keyframes --adaptive --method flow
```

Options:

-   `--keyframes`: Enable keyframe-based tracking
-   `--adaptive`: Enable adaptive parameters
-   `--method {bf|flann|flow}`: Choose matching method
-   `--save-data`: Save motion data to CSV/JSON
-   `--resolution WIDTHxHEIGHT`: Set camera resolution

## Configuration

The `MotionConfig` class centralizes all parameters that control the behavior of the motion estimator. Some important parameters:

-   `matching_method`: Method for feature matching (BRUTE_FORCE, FLANN, OPTICAL_FLOW)
-   `min_matches`: Minimum number of matches required (default: 5)
-   `ransac_threshold`: Threshold for RANSAC inliers (default: 2.0)
-   `use_keyframes`: Enable keyframe-based tracking (default: False)
-   `enable_adaptive_parameters`: Enable adaptive parameter adjustment (default: False)
-   `position_filter_alpha`: Position filter smoothing factor (default: 0.2)
-   `rotation_filter_alpha`: Rotation filter smoothing factor (default: 0.3)

## Dependencies

-   OpenCV (cv2)
-   NumPy
-   Python 3.6+

## Extending the Module

The modular design makes it easy to extend with new features:

-   Add new matching methods by extending the `FeatureMatcher` class
-   Create new filters by adding classes to the `filters` package
-   Add new motion models to the `models` package
-   Implement visualization tools in the `viz` package

## Future Improvements

-   Multi-frame feature tracking
-   Loop closure detection
-   Bundle adjustment
-   Scale recovery from external sensors
-   More advanced keyframe selection strategies
-   Visual odometry with depth information
