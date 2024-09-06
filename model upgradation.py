import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import logging
import time
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    keypoints = keypoints / np.max(keypoints)
    if keypoints.shape[0] != np.prod(input_shape):
        raise ValueError(f"Keypoints shape mismatch: expected {input_shape}, got {keypoints.shape}")
    return keypoints

def predict_label(keypoints):
    """Predict label from keypoints using the trained model."""
    try:
        input_shape = model.input_shape[1:]
        keypoints = preprocess_keypoints(keypoints, input_shape)
        keypoints = np.expand_dims(keypoints, axis=0)
        prediction = model.predict(keypoints, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)
        return label_classes[predicted_class[0]], np.max(prediction)
    except ValueError as e:
        logging.warning(f"Preprocessing error: {e}")
        return "Unknown", 0.0
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return "Unknown", 0.0

def fine_tune_model(model, keypoints, correct_label):
    """Fine-tune the model with the new data point."""
    try:
        input_shape = model.input_shape[1:]
        keypoints = preprocess_keypoints(keypoints, input_shape)
        keypoints = np.expand_dims(keypoints, axis=0)

        correct_label_index = np.where(label_classes == correct_label)[0][0]
        correct_label_one_hot = np.zeros((1, num_classes))
        correct_label_one_hot[0, correct_label_index] = 1

        model.fit(keypoints, correct_label_one_hot, epochs=1, verbose=0)
    except Exception as e:
        logging.error(f"Fine-tuning error: {e}")

def draw_landmarks_and_prediction(img, hand_landmarks, prediction, confidence):
    """Draw hand landmarks and prediction label on the image."""
    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.putText(img, f'Prediction: {prediction} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def countdown_timer(img, countdown_time=3):
    """Display a countdown timer on the screen."""
    overlay = img.copy()
    height, width, _ = img.shape

    for i in range(countdown_time, 0, -1):
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        alpha = 0.5
        img_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        cv2.putText(img_new, f'Starting in {i}', (width // 2 - 150, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        cv2.imshow('Real-Time Sign Language Detection', img_new)
        cv2.waitKey(1000)

def prompt_for_feedback():
    """Prompt user for feedback using tkinter dialogs."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Prompt for correct label
    correct_label = simpledialog.askstring("Correct Label", "Select the correct label:", initialvalue=label_classes[0])
    if correct_label is None or correct_label not in label_classes:
        tk.messagebox.showwarning("Warning", "Invalid label selected. Please try again.")
        return None

    return correct_label

def process_frame(img, detect_mode):
    """Process a single frame: extract keypoints and make predictions if in detect_mode."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks and detect_mode:
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

    # Set full screen
    cv2.namedWindow('Real-Time Sign Language Detection', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Real-Time Sign Language Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    logging.info("Starting video capture...")
    prev_time = time.time()
    detect_mode = False

    while True:
        success, img = cap.read()
        if not success:
            logging.error("Error reading frame.")
            break

        if detect_mode:
            countdown_timer(img)

        img = process_frame(img, detect_mode)
        cv2.imshow('Real-Time Sign Language Detection', img)

        # Measure FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            detect_mode = not detect_mode
            logging.info(f"Detection mode {'activated' if detect_mode else 'deactivated'}")

            if detect_mode:
                # Wait for 3 seconds to give the trainer time to get ready
                countdown_timer(img)
                correct_label = prompt_for_feedback()
                if correct_label is not None:
                    keypoints = [[lm.x, lm.y, lm.z] for lm in hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).multi_hand_landmarks[0].landmark]
                    fine_tune_model(model, keypoints, correct_label)
                else:
                    logging.warning("No valid label provided. Skipping fine-tuning.")

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Video capture ended.")

    # Save the fine-tuned model with a timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    updated_model_filename = f'final_model_updated_{timestamp}.keras'
    model.save(updated_model_filename)
    logging.info(f"Updated model saved as {updated_model_filename}")

if __name__ == "__main__":
    main()
