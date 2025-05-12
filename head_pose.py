"""
Head Pose Detection Module
Tracks head position using facial landmarks and 3D model points
"""

import cv2
import dlib
import numpy as np
import math
from collections import deque

# Initialize face detection and landmark models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

# 3D model points for head pose estimation
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -50.0, -10.0),
    (-30.0, 40.0, -10.0),   # Left eye
    (30.0, 40.0, -10.0),    # Right eye
    (-25.0, -30.0, -10.0),  # Left mouth corner
    (25.0, -30.0, -10.0)    # Right mouth corner
], dtype=np.float64)

# Camera calibration (assuming 640x480 resolution)
CAMERA_MATRIX = np.array([
    [640, 0, 320],  # focal_length = 640, center_x = 320
    [0, 640, 240],  # focal_length = 640, center_y = 240
    [0, 0, 1]
], dtype=np.float64)

DIST_COEFFS = np.zeros((4, 1))  # No lens distortion assumed

# Smoothing parameters
ANGLE_HISTORY_SIZE = 10
yaw_history = deque(maxlen=ANGLE_HISTORY_SIZE)
pitch_history = deque(maxlen=ANGLE_HISTORY_SIZE)
roll_history = deque(maxlen=ANGLE_HISTORY_SIZE)

# State management
previous_state = "Looking at Screen"
calibrated_angles = None

def get_head_pose_angles(image_points):
    """Calculate head pose angles (pitch, yaw, roll) from facial landmarks
    
    Args:
        image_points: 2D facial landmark points from the image
    
    Returns:
        tuple: (pitch, yaw, roll) angles in degrees or None if detection fails
    """
    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return None

    # Convert rotation vector to matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Calculate Euler angles
    sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0

    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

def smooth_angle(angle_history, new_angle):
    """Apply moving average smoothing to angle measurements
    
    Args:
        angle_history: Deque containing angle history
        new_angle: Latest angle measurement
    
    Returns:
        float: Smoothed angle value
    """
    angle_history.append(new_angle)
    return np.mean(angle_history)

def process_head_pose(frame, calibrated_angles=None):
    """Process frame to detect head pose direction
    
    Args:
        frame: Input video frame
        calibrated_angles: Optional tuple of (pitch, yaw, roll) for calibration
    
    Returns:
        tuple: (processed_frame, head_direction) or (frame, angles) if calibrating
    """
    global previous_state

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    head_direction = "Looking at Screen"

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        
        # Get key facial landmarks for pose estimation
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
        ], dtype=np.float64)

        angles = get_head_pose_angles(image_points)
        if angles is None:
            continue

        # Apply smoothing to angles
        pitch = smooth_angle(pitch_history, angles[0])
        yaw = smooth_angle(yaw_history, angles[1])
        roll = smooth_angle(roll_history, angles[2])

        # Return raw angles if calibrating
        if calibrated_angles is None:
            return frame, (pitch, yaw, roll)

        # Use calibrated angles for direction detection
        pitch_offset, yaw_offset, roll_offset = calibrated_angles
        
        # Direction detection thresholds
        PITCH_THRESHOLD = 8
        YAW_THRESHOLD = 12 
        ROLL_THRESHOLD = 5

        # Determine head direction based on angle offsets
        if (abs(yaw - yaw_offset) <= YAW_THRESHOLD and 
            abs(pitch - pitch_offset) <= PITCH_THRESHOLD and
            abs(roll - roll_offset) <= ROLL_THRESHOLD):
            current_state = "Looking at Screen"
        elif yaw < yaw_offset - 15:
            current_state = "Looking Left"
        elif yaw > yaw_offset + 15:
            current_state = "Looking Right"
        elif pitch > pitch_offset + 10:
            current_state = "Looking Up"
        elif pitch < pitch_offset - 10:
            current_state = "Looking Down"
        elif abs(roll - roll_offset) > 7:
            current_state = "Tilted"
        else:
            current_state = previous_state

        previous_state = current_state
        head_direction = current_state

    return frame, head_direction
