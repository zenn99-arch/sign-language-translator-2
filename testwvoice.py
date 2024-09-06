import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import logging
import time
import pyttsx3


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Load the trained model
try:
    model = tf.keras.models.load_model('final_model.keras')
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

# Load label classes
try:
    label_classes = np.load('label_encoder_classes.npy')
except Exception as e:
    logging.error(f"Failed to load label classes: {e}")
    raise

num_classes = len(label_classes)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def preprocess_keypoints(keypoints, input_shape):
    """Preprocess keypoints for model input."""
    keypoints = np.array(keypoints).flatten()
    keypoints = keypoints / np.max(keypoints)  # Normalize keypoints
    if keypoints.shape[0] != np.prod(input_shape):
        raise ValueError(f"Keypoints shape mismatch: expected {input_shape}, got {keypoints.shape}")
    return keypoints

def predict_label(keypoints):
    """Predict label from keypoints using the trained model."""
    try:
        input_shape = model.input_shape[1:]  # Determine the input shape from the model
        keypoints = preprocess_keypoints(keypoints, input_shape)
        keypoints = np.expand_dims(keypoints, axis=0)  # Add batch dimension
        prediction = model.predict(keypoints, verbose=0)  # Suppress output for faster processing
        predicted_class = np.argmax(prediction, axis=1)
        return label_classes[predicted_class[0]], np.max(prediction)
    except ValueError as e:
        logging.warning(f"Preprocessing error: {e}")
        return "Unknown", 0.0
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return "Unknown", 0.0

def draw_landmarks_and_prediction(img, hand_landmarks, prediction, confidence):
    """Draw hand landmarks and prediction label on the image."""
    # Draw hand landmarks
    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw prediction
    cv2.putText(img, f'Prediction: {prediction} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Speak the prediction
    engine.say(prediction)
    engine.runAndWait()

def process_frame(img):
    """Process a single frame: extract keypoints and make predictions."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            label, confidence = predict_label(keypoints)
            draw_landmarks_and_prediction(img, hand_landmarks, label, confidence)
    return img

def main():
    """Main function to capture video and process frames."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open video capture.")
        return

    logging.info("Starting video capture...")
    prev_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            logging.error("Error reading frame.")
            break

        img = process_frame(img)
        cv2.imshow('Real-Time Sign Language Detection', img)

        # Measure FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Video capture ended.")

if __name__ == "__main__":
    main()
