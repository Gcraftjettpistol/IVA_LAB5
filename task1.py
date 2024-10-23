
import cv2
import numpy as np

def detect_motion_regions(frame1, frame2, threshold_value):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the current frame and the previous frame
    diff = cv2.absdiff(gray1, gray2)

    # Apply a binary threshold to the difference image
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    # Perform erosion and dilation to remove small noise and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Find contours in the processed image
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Parameters
motion_threshold = 50 # Motion detection sensitivity for frame differencing
min_contour_area = 100  # Minimum area to consider as significant motion

previous_frame = None

# Load the video file
video_path = "D:\\Assignments\\image and vdo\\LAB\\LAB 5\\samplevideo.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get the current timestamp in seconds
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert milliseconds to seconds

    # Create a copy of the current frame for motion detection
    frame_for_motion_detection = frame.copy()

    if previous_frame is not None:
        # Detect regions of motion using frame differencing
        contours = detect_motion_regions(previous_frame, frame_for_motion_detection, motion_threshold)

        motion_detected = False  # Flag to track if motion is detected

        # Filter and draw bounding boxes around significant motion regions
        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                continue  # Ignore small areas that are likely noise

            # Get the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(contour)
            # Draw bounding box on the original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion_detected = True  # Set the flag if motion is detected

        # If motion is detected, add timestamp to the frame
        if motion_detected:
            cv2.putText(frame, 'Motion Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'Timestamp: {timestamp:.2f}s', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Motion Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Update the previous frame
    previous_frame = frame_for_motion_detection

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
