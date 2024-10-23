import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Paths to male and female directories
male_dir = r'D:\Assignments\image and vdo\LAB\LAB 5\archive (5)\Train\Train\male'
female_dir = r'D:\Assignments\image and vdo\LAB\LAB 5\archive (5)\Train\Train\female'

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def extract_features(face_region):
    # Calculate aspect ratio as a simple geometric feature
    height, width = face_region.shape[:2]
    aspect_ratio = width / height

    # Calculate LBP features for texture
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    lbp = cv2.calcHist([gray_face], [0], None, [256], [0, 256]).flatten()
    
    # Combine features
    return np.hstack([aspect_ratio, lbp])

def create_feature_dataset(male_dir, female_dir):
    X = []  # Initialize X as a list
    y = []  # Initialize y as a list
    
    # Load male images and extract features
    male_images = load_images_from_folder(male_dir)
    for img in male_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_region = img[y:y+h, x:x+w]
            face_region = cv2.resize(face_region, (128, 128))
            features = extract_features(face_region)
            X.append(features)
            y.append(1)  # Male label

    # Load female images and extract features
    female_images = load_images_from_folder(female_dir)
    for img in female_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_region = img[y:y+h, x:x+w]
            face_region = cv2.resize(face_region, (128, 128))
            features = extract_features(face_region)
            X.append(features)
            y.append(0)  # Female label

    return np.array(X), np.array(y)

def save_features_to_csv(X, y, filename):
    df = pd.DataFrame(X)
    df['label'] = y  # Add labels as a new column
    df.to_csv(filename, index=False)

# Create feature dataset
X, y = create_feature_dataset(male_dir, female_dir)

# Save features to a CSV file
save_features_to_csv(X, y, 'D:\Assignments\image and vdo\LAB\LAB 5\gender_features.csv')

# Load the features from the CSV file for predictions
data = pd.read_csv('D:\Assignments\image and vdo\LAB\LAB 5\gender_features.csv')
X_loaded = data.drop(columns=['label']).values
y_loaded = data['label'].values

# Train a simple SVM classifier
classifier = make_pipeline(StandardScaler(), SVC(kernel='linear'))
classifier.fit(X_loaded, y_loaded)

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def crop_and_normalize(frame, face_coords):
    (x, y, w, h) = face_coords
    face_region = frame[y:y+h, x:x+w]
    face_region = cv2.resize(face_region, (128, 128))  # Resize to 128x128 for consistency
    return face_region

def process_frame_with_gender_prediction(frame):
    faces = detect_faces(frame)
    for (x, y, w, h) in faces:
        face_region = crop_and_normalize(frame, (x, y, w, h))
        features = extract_features(face_region)
        
        # Predict gender using the trained classifier
        gender_prediction = classifier.predict(features.reshape(1, -1))[0]
        gender_label = "Male" if gender_prediction == 1 else "Female"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, gender_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

def extract_frame_with_gender_predictions(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_with_predictions = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame_with_gender_prediction(frame)
        frame_with_predictions = processed_frame.copy()
        break

    cap.release()
    return frame_with_predictions

def display_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.title("Frame with Gender Predictions")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    video_path = r"D:\Assignments\image and vdo\LAB\LAB 5\Task2_sampleVideo.mp4"  # Replace with your video path
    frame_with_predictions = extract_frame_with_gender_predictions(video_path)

    if frame_with_predictions is not None:
        display_frame(frame_with_predictions)
    else:
        print("No frame could be processed.")
