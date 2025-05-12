# Smart Exam Monitoring System Using Computer Vision

## Overview
Designed to monitor and detect potential cheating behaviors during examinations. It utilizes computer vision techniques to track eye movements, head poses, and mobile phone usage.

## File Descriptions
- **main.py**: The main script that initializes the video capture and runs the detection loop. It integrates the functionalities of eye movement detection, head pose estimation, and mobile phone detection.

- **eye_movement.py**: This module detects pupil positions and determines gaze direction using facial landmarks. It provides real-time feedback on whether the user is looking at the screen or not.

- **head_pose.py**: This module tracks head position using facial landmarks and a 3D model. It calculates pitch, yaw, and roll angles to determine the head's orientation.
    
- **mobile_detection.py**: This module uses a YOLO model to detect mobile phones in the camera frame. It highlights detected phones with bounding boxes and confidence scores.

## Technologies Used
- **OpenCV**: For image processing and computer vision tasks.
- **dlib**: For facial landmark detection.
- **PyTorch**: For implementing the YOLO model for mobile detection.
- **YOLO**: A state-of-the-art object detection model used for real-time detection.

## Setup Instructions
1. Clone the repository.
2. Install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```
3. Ensure you have the necessary model files in the `model/` directory.
4. Run the main script:
   ```
   python main.py
   ```

## Usage
- The system will start capturing video from the default camera.
- It will display real-time feedback on eye movements, head position, and mobile phone detection.
- Press 'q' to quit the application.
