"""
Visualization utilities for motion visualization.

This module provides helper functions for motion visualization.
"""

import cv2
import numpy as np
import os
import csv
import json
from datetime import datetime

def create_output_dir(dirname='output'):
    """
    Create output directory if it doesn't exist.
    
    Args:
        dirname: Directory name to create
        
    Returns:
        Path to the created directory
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname

def save_trajectory_to_csv(trajectory, filename=None, output_dir='output', 
                          rotation_history=None, translation_history=None):
    """
    Save trajectory data to CSV file.
    
    Args:
        trajectory: List of position points (x, y, z)
        filename: Output filename or None for automatic timestamp-based name
        output_dir: Output directory
        rotation_history: Optional list of rotation matrices
        translation_history: Optional list of translation vectors
        
    Returns:
        Path to the saved file
    """
    # Create output directory
    create_output_dir(output_dir)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_{timestamp}.csv"
    
    # Create full path
    filepath = os.path.join(output_dir, filename)
    
    # Prepare headers
    headers = ['X', 'Y', 'Z']
    
    # Add rotation headers if provided
    if rotation_history is not None and len(rotation_history) > 0:
        for i in range(3):
            for j in range(3):
                headers.append(f'R{i+1}{j+1}')
    
    # Add translation headers if provided
    if translation_history is not None and len(translation_history) > 0:
        headers.extend(['TX', 'TY', 'TZ'])
    
    # Write data
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i, pos in enumerate(trajectory):
            row = list(pos)
            
            # Add rotation data if available
            if rotation_history is not None and i < len(rotation_history):
                R = rotation_history[i]
                for r_row in R:
                    row.extend(r_row)
            
            # Add translation data if available
            if translation_history is not None and i < len(translation_history):
                t = translation_history[i]
                row.extend([t[0, 0], t[1, 0], t[2, 0]])
                
            writer.writerow(row)
    
    print(f"Saved trajectory to {filepath}")
    return filepath

def save_trajectory_to_json(trajectory, filename=None, output_dir='output', 
                           rotation_history=None, translation_history=None, 
                           metadata=None):
    """
    Save trajectory data to JSON file.
    
    Args:
        trajectory: List of position points (x, y, z)
        filename: Output filename or None for automatic timestamp-based name
        output_dir: Output directory
        rotation_history: Optional list of rotation matrices
        translation_history: Optional list of translation vectors
        metadata: Optional dictionary with additional metadata
        
    Returns:
        Path to the saved file
    """
    # Create output directory
    create_output_dir(output_dir)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_{timestamp}.json"
    
    # Create full path
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data structure
    data = {
        'timestamp': datetime.now().isoformat(),
        'points_count': len(trajectory),
        'trajectory': []
    }
    
    # Add metadata if provided
    if metadata is not None:
        data['metadata'] = metadata
    
    # Convert trajectory points
    for i, pos in enumerate(trajectory):
        point_data = {
            'position': pos.tolist() if isinstance(pos, np.ndarray) else list(pos)
        }
        
        # Add rotation if available
        if rotation_history is not None and i < len(rotation_history):
            R = rotation_history[i]
            point_data['rotation'] = R.tolist() if isinstance(R, np.ndarray) else R
        
        # Add translation if available
        if translation_history is not None and i < len(translation_history):
            t = translation_history[i]
            point_data['translation'] = t.flatten().tolist() if isinstance(t, np.ndarray) else t
            
        data['trajectory'].append(point_data)
    
    # Write JSON file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Saved trajectory to {filepath}")
    return filepath

def save_visualization_image(image, filename=None, output_dir='output'):
    """
    Save visualization image to file.
    
    Args:
        image: Image to save
        filename: Output filename or None for automatic timestamp-based name
        output_dir: Output directory
        
    Returns:
        Path to the saved file
    """
    # Create output directory
    create_output_dir(output_dir)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visualization_{timestamp}.png"
    
    # Create full path
    filepath = os.path.join(output_dir, filename)
    
    # Save image
    cv2.imwrite(filepath, image)
    
    print(f"Saved visualization to {filepath}")
    return filepath

def colorize_depth(depth, min_depth=0.1, max_depth=10.0):
    """
    Colorize depth map for visualization.
    
    Args:
        depth: Depth map (float values representing distance)
        min_depth: Minimum depth value for normalization
        max_depth: Maximum depth value for normalization
        
    Returns:
        Colorized depth map as BGR image
    """
    # Normalize depth values
    depth_normalized = np.clip((depth - min_depth) / (max_depth - min_depth), 0, 1)
    
    # Convert to uint8
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Apply colormap
    depth_colorized = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    
    return depth_colorized

def draw_coordinate_system(image, camera_matrix, rvec, tvec, scale=0.1):
    """
    Draw 3D coordinate system axes on the image.
    
    Args:
        image: Image to draw on
        camera_matrix: Camera intrinsic matrix (3x3)
        rvec: Rotation vector
        tvec: Translation vector
        scale: Scale of the coordinate axes
        
    Returns:
        Image with coordinate system drawn
    """
    # Define the coordinate system points
    points = np.float32([
        [0, 0, 0],  # Origin
        [scale, 0, 0],  # X-axis
        [0, scale, 0],  # Y-axis
        [0, 0, scale]   # Z-axis
    ])
    
    # Project points to image plane
    dist_coeffs = np.zeros(4)  # Assume no distortion for visualization
    image_points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    
    # Convert to integer coordinates
    image_points = np.int32(image_points).reshape(-1, 2)
    
    # Draw the coordinate axes
    origin = tuple(image_points[0])
    x_axis = tuple(image_points[1])
    y_axis = tuple(image_points[2])
    z_axis = tuple(image_points[3])
    
    # Draw the axes with different colors
    image = cv2.line(image, origin, x_axis, (0, 0, 255), 2)  # X-axis: Red
    image = cv2.line(image, origin, y_axis, (0, 255, 0), 2)  # Y-axis: Green
    image = cv2.line(image, origin, z_axis, (255, 0, 0), 2)  # Z-axis: Blue
    
    return image

def create_trajectory_plot(trajectory, figsize=(10, 6), dpi=100, include_z=True):
    """
    Create a matplotlib figure with trajectory plot.
    
    Args:
        trajectory: List of position points (x, y, z)
        figsize: Figure size (width, height) in inches
        dpi: Dots per inch
        include_z: Whether to include Z component
        
    Returns:
        Figure as a numpy image array
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from io import BytesIO
        
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Extract coordinates
        x = [p[0] for p in trajectory]
        y = [p[1] for p in trajectory]
        
        if include_z and len(trajectory[0]) > 2:
            # 3D plot
            z = [p[2] for p in trajectory]
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(x, y, z, 'b-')
            ax.scatter(x[0], y[0], z[0], c='g', marker='o', s=50)  # Start
            ax.scatter(x[-1], y[-1], z[-1], c='r', marker='o', s=50)  # End
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Trajectory')
        else:
            # 2D plot
            ax = fig.add_subplot(111)
            ax.plot(x, y, 'b-')
            ax.scatter(x[0], y[0], c='g', marker='o', s=50)  # Start
            ax.scatter(x[-1], y[-1], c='r', marker='o', s=50)  # End
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('2D Trajectory')
            ax.grid(True)
        
        # Save figure to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        
        # Convert to numpy array
        buf.seek(0)
        img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        return img
    except ImportError:
        # If matplotlib is not available, return a message image
        height, width = int(figsize[1] * dpi), int(figsize[0] * dpi)
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Matplotlib not available for trajectory plotting", 
                  (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return img 