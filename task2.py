import cv2
import numpy as np

# Load Video
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap

# Load Haar Cascades for face detection (frontal and profile)
face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Load Haar Cascade for eyes detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Skin color thresholding for hand detection
def detect_hands(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on size
    hands = []
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Minimum size threshold for hands
            x, y, w, h = cv2.boundingRect(contour)
            hands.append((x, y, w, h))
    return hands

# Extract Facial Features
def extract_facial_features(face):
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)

    # Determine if the mouth is open
    mouth_region = face[int(face.shape[0] * 0.65):, int(face.shape[1] * 0.4):int(face.shape[1] * 0.6)]
    mean_intensity = np.mean(mouth_region)
    mouth_open = mean_intensity < 50  # Threshold for detecting an open mouth

    # Determine if eyebrows are closed (if eyes detected or not)
    eyebrows_closed = len(eyes) == 0

    # Return feature indicators
    return {
        'mouth_open': mouth_open,
        'eyebrows_closed': eyebrows_closed
    }

# Classify Basic Emotions
def classify_emotion(features, hands_detected):
    if features['mouth_open']:
        return "surprised"
    elif features['eyebrows_closed']:
        return "sad"
    elif hands_detected:
        return "happy"
    else:
        return "neutral"

# Main Video Processing
def analyze_video(video_path):
    cap = load_video(video_path)
    overall_sentiment = {
        "happy": 0,
        "sad": 0,
        "neutral": 0,
        "surprised": 0,
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame using both frontal and profile cascades
        faces_frontal = face_cascade_frontal.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        faces_profile = face_cascade_profile.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # Combine detected faces
        faces = np.concatenate((faces_frontal, faces_profile))

        # Detect hands in the frame using skin color detection
        hands = detect_hands(frame)

        # Analyze each detected face
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            # Extract facial features to analyze the state of mouth and eyes
            features = extract_facial_features(face_region)

            # Check if hands are near the mouth
            hands_detected = any(
                (x < hand_x + hand_w and x + w > hand_x and y < hand_y + hand_h and y + h > hand_y) 
                for (hand_x, hand_y, hand_w, hand_h) in hands
            )

            # Classify emotion based on extracted features and hand positions
            emotion = classify_emotion(features, hands_detected)

            # Draw bounding box around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Put emotion label above the bounding box
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Increment overall sentiment count
            overall_sentiment[emotion] += 1

        cv2.imshow('Video Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return overall_sentiment

# Main Execution
video_path = "D:\\Assignments\\image and vdo\\LAB\\LAB 5\\Task2_sampleVideo.mp4"   # Replace with your video file path
overall_sentiment = analyze_video(video_path)

print(f"Overall Sentiment: {overall_sentiment}")
