"""
Utility functions for motion estimation and transformation operations.
"""

import numpy as np

def rotation_to_angle_axis(R):
    """
    Convert a rotation matrix to angle-axis representation.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        angle: Rotation angle in radians
        axis: Rotation axis (unit vector)
    """
    angle = np.arccos((np.trace(R) - 1) / 2)
    if np.isclose(angle, 0):
        return 0, np.array([1, 0, 0])
        
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])
    axis = axis / (2 * np.sin(angle))
    return angle, axis

def angle_axis_to_rotation(angle, axis):
    """
    Convert angle-axis representation to rotation matrix.
    
    Args:
        angle: Rotation angle in radians
        axis: Rotation axis (unit vector)
        
    Returns:
        R: 3x3 rotation matrix
    """
    if np.isclose(angle, 0):
        return np.eye(3)
        
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle)
    b = np.sin(angle)
    c = 1 - a
    x, y, z = axis
    
    return np.array([
        [a + x*x*c, x*y*c - z*b, x*z*c + y*b],
        [y*x*c + z*b, a + y*y*c, y*z*c - x*b],
        [z*x*c - y*b, z*y*c + x*b, a + z*z*c]
    ])

def orthogonalize_rotation(R):
    """
    Enforce orthogonality constraint on rotation matrix.
    Uses SVD decomposition to find the closest orthogonal matrix.
    
    Args:
        R: Rotation matrix to orthogonalize
        
    Returns:
        R_ortho: Orthogonalized rotation matrix
    """
    if R is None:
        return np.eye(3)
        
    # Use SVD to find closest orthogonal matrix
    try:
        u, _, vh = np.linalg.svd(R, full_matrices=True)
        R_ortho = u @ vh
        
        # Ensure proper rotation (det=1)
        if np.linalg.det(R_ortho) < 0:
            u[:, -1] = -u[:, -1]
            R_ortho = u @ vh
            
        return R_ortho
    except:
        print("Error orthogonalizing rotation matrix, using identity")
        return np.eye(3)

def rotation_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        q: Quaternion [w, x, y, z]
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
        
    return np.array([w, x, y, z])

def quaternion_to_rotation(q):
    """
    Convert a quaternion to a rotation matrix.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        R: 3x3 rotation matrix
    """
    w, x, y, z = q
    
    # Normalize quaternion
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-10:
        return np.eye(3)
        
    q = q / n
    w, x, y, z = q
    
    # Build rotation matrix
    R = np.zeros((3, 3))
    
    R[0, 0] = 1 - 2*y*y - 2*z*z
    R[0, 1] = 2*x*y - 2*w*z
    R[0, 2] = 2*x*z + 2*w*y
    
    R[1, 0] = 2*x*y + 2*w*z
    R[1, 1] = 1 - 2*x*x - 2*z*z
    R[1, 2] = 2*y*z - 2*w*x
    
    R[2, 0] = 2*x*z - 2*w*y
    R[2, 1] = 2*y*z + 2*w*x
    R[2, 2] = 1 - 2*x*x - 2*y*y
    
    return R

def slerp_rotation(R1, R2, t):
    """
    Spherical linear interpolation between two rotation matrices.
    
    Args:
        R1: First rotation matrix
        R2: Second rotation matrix
        t: Interpolation parameter (0 = R1, 1 = R2)
        
    Returns:
        R: Interpolated rotation matrix
    """
    # Ensure t is in [0, 1]
    t = max(0, min(1, t))
    
    try:
        # Convert rotation matrices to quaternions
        q1 = rotation_to_quaternion(R1)
        q2 = rotation_to_quaternion(R2)
        
        # Calculate dot product
        dot = np.sum(q1 * q2)
        
        # If the dot product is negative, negate one quaternion
        # to ensure we take the shortest path
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        # Clamp dot to valid range for arccos
        dot = min(1.0, max(-1.0, dot))
        
        # Calculate the angle between quaternions
        theta = np.arccos(dot)
        
        # Handle special cases to avoid division by zero
        if theta < 1e-6:
            # Quaternions are very close, use linear interpolation
            result = q1 * (1 - t) + q2 * t
        else:
            # Use spherical linear interpolation
            sin_theta = np.sin(theta)
            result = (np.sin((1 - t) * theta) / sin_theta) * q1 + (np.sin(t * theta) / sin_theta) * q2
        
        # Normalize the result
        result = result / np.linalg.norm(result)
        
        # Convert back to rotation matrix
        return quaternion_to_rotation(result)
    except Exception as e:
        print(f"Error in SLERP: {e}")
        # Fall back to simple averaging
        return (1 - t) * R1 + t * R2 