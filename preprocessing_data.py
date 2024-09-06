import os
import cv2
import numpy as np
import logging
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import mediapipe as mp

# Paths and Constants
DATA_DIR = r'C:\Users\ASUS\Desktop\sign language translator\data'
X_TRAIN_PATH = 'X_train.npy'
X_VAL_PATH = 'X_val.npy'
Y_TRAIN_PATH = 'y_train.npy'
Y_VAL_PATH = 'y_val.npy'
LABEL_ENCODER_PATH = 'label_encoder_classes.npy'
USE_MEDIAPIPE = True

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_folders(paths):
    """Create directories if they don't exist."""
    for path in paths:
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Directory created: {dir_path}")

def extract_keypoints_with_mediapipe(image_path):
    """Extract keypoints using MediaPipe from an image."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        keypoints = [
            [lm.x, lm.y, lm.z] 
            for hand_landmarks in result.multi_hand_landmarks 
            for lm in hand_landmarks.landmark
        ]
        return np.array(keypoints).flatten()
    else:
        logging.warning(f"No hand landmarks detected in {image_path}")
        return None

def load_keypoints_from_xml(xml_path):
    """Load keypoints from an XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    keypoints_el = root.find('object/keypoints')
    if keypoints_el is None:
        logging.warning(f"No keypoints found in {xml_path}")
        return None
    
    keypoints = [
        [float(coord) for coord in kp_el.text.split(',')] 
        for kp_el in keypoints_el
    ]
    return np.array(keypoints).flatten()

def prepare_data(data_dir, use_mediapipe=True):
    """Prepare the dataset by extracting keypoints and labels."""
    X, y = [], []
    labels = os.listdir(data_dir)

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for filename in os.listdir(label_dir):
            if filename.endswith('.jpg'):
                image_path = os.path.join(label_dir, filename)
                keypoints = extract_keypoints_with_mediapipe(image_path) if use_mediapipe else load_keypoints_from_xml(image_path.replace('.jpg', '.xml'))

                if keypoints is not None:
                    X.append(keypoints)
                    y.append(label)

    logging.info(f"Data preparation completed: {len(X)} samples collected.")
    return np.array(X), np.array(y)

def save_data(X_train, X_val, y_train, y_val, label_encoder):
    """Save the preprocessed data and label encoder."""
    create_folders([X_TRAIN_PATH, X_VAL_PATH, Y_TRAIN_PATH, Y_VAL_PATH, LABEL_ENCODER_PATH])
    np.save(X_TRAIN_PATH, X_train)
    np.save(X_VAL_PATH, X_val)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_VAL_PATH, y_val)
    np.save(LABEL_ENCODER_PATH, label_encoder.classes_)
    logging.info("Preprocessed data and label encoder saved successfully.")

def main():
    logging.info("Starting data preprocessing...")

    # Prepare data
    X, y = prepare_data(DATA_DIR, use_mediapipe=USE_MEDIAPIPE)

    if X.size == 0 or y.size == 0:
        logging.error("No data to process. Exiting.")
        return

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logging.info(f"Labels encoded: {label_encoder.classes_}")

    # Scale the keypoints
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Keypoints scaled successfully.")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    logging.info(f"Data split: {len(X_train)} training samples, {len(X_val)} validation samples.")

    # Save preprocessed data
    save_data(X_train, X_val, y_train, y_val, label_encoder)
    logging.info("Data preprocessing complete.")

if __name__ == "__main__":
    main()
