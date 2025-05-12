"""
Main script for Cheating Surveillance System
Detects eye movements, head pose, and mobile phone usage
"""

import cv2
import time
import os
from eye_movement import process_eye_movement
from head_pose import process_head_pose
from mobile_detection import process_mobile_detection

# Setup video capture from default camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera")
    exit()

# Create directory for saving violation screenshots
os.makedirs("violation_logs", exist_ok=True)

# Initialize variables
calibration_start = time.time()
calibration_done = False
calibration_angles = None

# Main detection loop
while True:
    # Get camera frame
    success, frame = camera.read()
    if not success:
        print("Warning: Couldn't get frame from camera")
        continue
    
    # 1. Eye tracking
    frame, eye_status = process_eye_movement(frame)
    cv2.putText(frame, f"Eyes: {eye_status}", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 2. Head position tracking
    if not calibration_done:
        # 5-second calibration period
        if time.time() - calibration_start <= 5:
            cv2.putText(frame, "Calibrating - look straight ahead", 
                       (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            if calibration_angles is None:
                try:
                    _, angles = process_head_pose(frame, None)
                    if angles and len(angles) == 3:
                        calibration_angles = tuple(angles)
                except Exception as e:
                    print(f"Calibration error: {e}")
        else:
            calibration_done = True
    
    if calibration_done:
        frame, head_status = process_head_pose(frame, calibration_angles)
        cv2.putText(frame, f"Head: {head_status}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 3. Phone detection
    frame, phone_detected = process_mobile_detection(frame)
    cv2.putText(frame, f"Phone: {phone_detected}", (20, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the combined results
    cv2.imshow("Exam Monitoring System", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()
