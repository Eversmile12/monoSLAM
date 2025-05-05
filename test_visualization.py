#!/usr/bin/env python3
"""
Test script for the refactored motion visualization components.
"""

import cv2
import numpy as np
import time
import argparse
import os
from datetime import datetime

# Import our custom modules
from camera_manager import CameraManager
from visual_processor import VisualProcessor
from motion.core.config import MotionConfig, MatchingMethod
from motion.core.motion_estimator import MotionEstimator
from motion.viz.visualizer_factory import VisualizerFactory
from motion.viz import viz_utils

def parse_args():
    parser = argparse.ArgumentParser(description='Test motion visualization components')
    parser.add_argument('--resolution', type=str, default='640x480',
                      help='Camera resolution in format WIDTHxHEIGHT')
    parser.add_argument('--fps', type=int, default=30,
                      help='Camera FPS')
    parser.add_argument('--use-opencv', action='store_true',
                      help='Use OpenCV camera instead of picamera2')
    parser.add_argument('--method', choices=['bf', 'flann', 'flow'], default='bf',
                      help='Feature matching method')
    parser.add_argument('--save-output', action='store_true',
                      help='Save output trajectory and images')
    return parser.parse_args()

def get_matching_method(method_str):
    if method_str == 'bf':
        return MatchingMethod.BRUTE_FORCE
    elif method_str == 'flann':
        return MatchingMethod.FLANN
    elif method_str == 'flow':
        return MatchingMethod.OPTICAL_FLOW
    else:
        return MatchingMethod.BRUTE_FORCE

def create_motion_config(args):
    """Create a motion configuration based on command line arguments"""
    config = MotionConfig()
    
    # Set matching method
    config.matching_method = get_matching_method(args.method)
    
    # Enable keyframes and adaptive parameters
    config.use_keyframes = True
    config.enable_adaptive_parameters = True
    
    # Other optimized parameters
    config.min_matches = 5
    config.ransac_threshold = 2.0
    config.max_consecutive_failures = 3
    config.position_filter_alpha = 0.2
    config.rotation_filter_alpha = 0.3
    config.smoothing_factor = 0.7
    config.adaptive_smoothing = True
    
    return config

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Create output directory if needed
    if args.save_output:
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Initialize components
    print("Initializing camera...")
    cam = CameraManager(resolution=(width, height), fps=args.fps, 
                       use_picamera2=not args.use_opencv)
    
    print("Initializing visual processor...")
    processor = VisualProcessor(
        max_features=500,
        scale_factor=1.05,
        nlevels=8,
        fast_threshold=10,
        first_level=0,
        grid_size=6,
        features_per_grid=20,
        adaptive_threshold=True
    )
    
    # Enable CLAHE for better feature detection after initialization
    processor.use_clahe = True
    
    print("Creating motion config...")
    config = create_motion_config(args)
    
    print("Initializing motion estimator...")
    motion_estimator = MotionEstimator(
        camera_matrix=cam.camera_matrix,
        dist_coeffs=cam.dist_coeffs,
        config=config
    )
    
    print("Creating visualizers...")
    # Use the factory to create visualizers
    visualizers = VisualizerFactory.create_composite_visualizer(
        motion_estimator,
        window_sizes={
            'trajectory': (800, 600),
            'features': (800, 600)
        }
    )
    
    # Extract individual visualizers
    trajectory_viz = visualizers['trajectory']
    feature_viz = visualizers['features']
    
    # Print controls
    trajectory_viz.print_help()
    feature_viz.print_help()
    
    # Variables for main loop
    frame_count = 0
    start_time = time.time()
    prev_frame = None
    prev_keypoints = None
    prev_descriptors = None
    
    # Start capturing
    cam.start()
    time.sleep(1)  # Let the camera warm up
    
    print("Processing frames...")
    try:
        while True:
            # Get frame from camera
            frame_data = cam.get_frame()
            if frame_data is None:
                continue
            
            # Unpack the frame tuple
            frame = frame_data[0]
            
            # Get current timestamp
            timestamp = time.time() - start_time
            
            # Process frame - convert to grayscale and detect features
            if len(frame.shape) > 2:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Detect features
            keypoints, descriptors = processor.detect_features(gray)
            
            # Detect and track motion
            success, R, t, position = motion_estimator.process_frame(
                frame, keypoints, descriptors, timestamp, gray
            )
            
            # Get motion statistics from motion estimator
            stats = motion_estimator.get_motion_stats()
            
            # Get tracking quality info for display
            tracking_info = {
                'quality': stats['tracking_quality'],
                'in_recovery_mode': stats['consecutive_failures'] > 0,
                'motion_model_active': stats['consecutive_failures'] > 0
            }
            
            # Print debug info occasionally
            if frame_count % 10 == 0:
                quality_str = f"Quality: {tracking_info['quality']:.2f}"
                if tracking_info['in_recovery_mode']:
                    quality_str += " (RECOVERY)"
                if tracking_info['motion_model_active']:
                    quality_str += " (MODEL)"
                
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                print(f"Frame {frame_count}: {fps:.1f} FPS | {quality_str} | "
                      f"Position=[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
            
            # Get data for visualization
            trajectory = motion_estimator.get_trajectory()
            keyframes = motion_estimator.get_keyframes() if config.use_keyframes else []
            keyframe_positions = np.array([kf.position for kf in keyframes]) if keyframes else np.zeros((0, 3))
            
            # Get matched points for visualization
            matched_kp1 = motion_estimator.matched_kp1
            matched_kp2 = motion_estimator.matched_kp2
            
            # Create statistics for visualization
            vis_stats = {
                'Matches': stats['match_count'],
                'Inliers': stats['inlier_count'],
                'Quality': stats['tracking_quality'],
                'Keyframes': len(keyframes) if config.use_keyframes else 0,
                'Recovery': 1 if tracking_info['in_recovery_mode'] else 0,
                'Motion Model': 1 if tracking_info['motion_model_active'] else 0
            }
            
            # Update trajectory visualization
            traj_cmd = trajectory_viz.update(
                trajectory=trajectory,
                current_position=position,
                R=R, 
                t=t,
                keyframe_positions=keyframe_positions,
                stats=vis_stats
            )
            
            # Update feature visualization
            feat_cmd = None
            if prev_frame is not None and matched_kp1 is not None and matched_kp2 is not None:
                # Create inlier mask (1 for inliers, 0 for outliers)
                # For simplicity, we'll consider all matches as inliers in this test
                mask = np.ones(len(matched_kp1), dtype=np.uint8)
                
                feat_cmd = feature_viz.update(
                    current_frame=frame,
                    prev_frame=prev_frame,
                    keypoints=keypoints,
                    prev_keypoints=prev_keypoints,
                    matches=list(range(len(matched_kp1))),
                    mask=mask,
                    stats=vis_stats
                )
            
            # Handle visualization commands
            if traj_cmd == 'quit' or feat_cmd == 'quit':
                break
            elif traj_cmd == 'record' or feat_cmd == 'record':
                # Toggle recording in both visualizers
                if not trajectory_viz.recording:
                    trajectory_viz.start_recording()
                else:
                    trajectory_viz.stop_recording()
                    
                # Save visualization images if requested
                if args.save_output and trajectory_viz.recording:
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Save trajectory visualization
                    viz_utils.save_visualization_image(
                        trajectory_viz.get_image(),
                        f"trajectory_{timestamp_str}.png",
                        'output'
                    )
                    # Save feature visualization
                    if feature_viz.get_image() is not None:
                        viz_utils.save_visualization_image(
                            feature_viz.get_image(),
                            f"features_{timestamp_str}.png",
                            'output'
                        )
            elif traj_cmd == 'clear':
                # Reset the motion estimator
                motion_estimator.reset()
                print("Trajectory cleared")
            
            # Save current frame and keypoints for next iteration
            prev_frame = gray.copy()
            prev_keypoints = keypoints
            prev_descriptors = descriptors
            
            # Increment frame counter
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("\nCleaning up...")
        cam.stop()
        cv2.destroyAllWindows()
        
        # Save trajectory data if requested
        if args.save_output and trajectory:
            # Save trajectory to CSV
            viz_utils.save_trajectory_to_csv(
                trajectory,
                filename=f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                output_dir='output',
                rotation_history=motion_estimator.trajectory.rotation_history,
                translation_history=motion_estimator.trajectory.translation_history
            )
            
            # Save trajectory to JSON with metadata
            viz_utils.save_trajectory_to_json(
                trajectory,
                filename=f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                output_dir='output',
                rotation_history=motion_estimator.trajectory.rotation_history,
                translation_history=motion_estimator.trajectory.translation_history,
                metadata={
                    'matching_method': config.matching_method.name,
                    'keyframes_enabled': config.use_keyframes,
                    'adaptive_params_enabled': config.enable_adaptive_parameters,
                    'total_frames': frame_count,
                    'camera_matrix': cam.camera_matrix.tolist() if cam.camera_matrix is not None else None,
                    'camera_resolution': [width, height]
                }
            )

if __name__ == "__main__":
    main() 