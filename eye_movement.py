"""
Eye Movement Detection Module
Detects pupil position and determines gaze direction using facial landmarks
"""

import cv2
import dlib
import numpy as np

# Initialize face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

def detect_pupil(eye_region):
    """Detects pupil center and bounding box in an eye region
    
    Args:
        eye_region: Cropped image of the eye area
    
    Returns:
        tuple: (pupil_center, bounding_box) or (None, None) if not found
    """
    # Convert to grayscale and preprocess
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Threshold to isolate pupil (dark area)
    _, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours (potential pupils)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour (most likely pupil)
        pupil = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(pupil)
        center = (x + w//2, y + h//2)
        return center, (x, y, w, h)
    
    return None, None

def process_eye_movement(frame):
    """Processes frame to detect eye gaze direction
    
    Args:
        frame: Input camera frame
    
    Returns:
        tuple: (annotated_frame, gaze_direction)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    gaze = "Looking Center"  # Default gaze direction
    
    for face in faces:
        landmarks = landmark_predictor(gray, face)
        
        # Define eye landmark indices (dlib 68-point model)
        LEFT_EYE_POINTS = list(range(36, 42))
        RIGHT_EYE_POINTS = list(range(42, 48))
        
        # Get eye landmark coordinates
        left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE_POINTS])
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE_POINTS])
        
        # Get eye bounding boxes
        left_box = cv2.boundingRect(left_eye)
        right_box = cv2.boundingRect(right_eye)
        
        # Extract eye regions from frame
        left_region = frame[left_box[1]:left_box[1]+left_box[3], left_box[0]:left_box[0]+left_box[2]]
        right_region = frame[right_box[1]:right_box[1]+right_box[3], right_box[0]:right_box[0]+right_box[2]]
        
        # Detect pupils in both eyes
        left_pupil, _ = detect_pupil(left_region)
        right_pupil, _ = detect_pupil(right_region)
        
        # Draw eye bounding boxes
        cv2.rectangle(frame, (left_box[0], left_box[1]), 
                     (left_box[0]+left_box[2], left_box[1]+left_box[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_box[0], right_box[1]), 
                     (right_box[0]+right_box[2], right_box[1]+right_box[3]), (0, 255, 0), 2)
        
        # Draw pupils if detected
        if left_pupil:
            cv2.circle(frame, (left_box[0]+left_pupil[0], left_box[1]+left_pupil[1]), 5, (0, 0, 255), -1)
        if right_pupil:
            cv2.circle(frame, (right_box[0]+right_pupil[0], right_box[1]+right_pupil[1]), 5, (0, 0, 255), -1)
        
        # Determine gaze direction if both pupils detected
        if left_pupil and right_pupil:
            # Normalize pupil positions relative to eye size
            left_x, left_y = left_pupil
            right_x, right_y = right_pupil
            eye_width = left_box[2]
            eye_height = left_box[3]
            
            # Calculate normalized vertical position
            norm_left_y = left_y / eye_height
            norm_right_y = right_y / eye_height
            
            # Determine gaze direction based on pupil positions
            if left_x < eye_width//3 and right_x < eye_width//3:
                gaze = "Looking Left"
            elif left_x > 2*eye_width//3 and right_x > 2*eye_width//3:
                gaze = "Looking Right"
            elif norm_left_y < 0.3 and norm_right_y < 0.3:
                gaze = "Looking Up"
            elif norm_left_y > 0.5 and norm_right_y > 0.5:
                gaze = "Looking Down"
    
    return frame, gaze
