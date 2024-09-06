from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import base64
import logging

app = Flask(__name__)

# Load the trained model and classes
model = tf.keras.models.load_model('final_model.keras')
label_classes = np.load('label_encoder_classes.npy')

# Initialize MediaPipe
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
    input_shape = model.input_shape[1:]  # Determine the input shape from the model
    keypoints = preprocess_keypoints(keypoints, input_shape)
    keypoints = np.expand_dims(keypoints, axis=0)  # Add batch dimension
    prediction = model.predict(keypoints, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)
    return label_classes[predicted_class[0]], np.max(prediction)

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Language Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background-color: #f4f4f4;
        }

        video {
            border: 2px solid #333;
            margin-bottom: 20px;
        }

        button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }

        #result {
            margin-top: 20px;
            font-size: 18px;
            color: #333;
        }
    </style>
</head>
<body>
    <h1>Sign Language Detection</h1>
    <video id="video" width="640" height="480" autoplay></video>
    <button id="startButton">Start Detection</button>
    <div id="result"></div>

    <script>
        document.getElementById('startButton').addEventListener('click', function() {
            startDetection();
        });

        function startDetection() {
            const video = document.getElementById('video');

            if (navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(function(stream) {
                        video.srcObject = stream;
                    })
                    .catch(function(error) {
                        console.log("Something went wrong!");
                    });
            }

            // Send video frames to backend every few seconds for processing
            setInterval(() => {
                captureFrameAndSend();
            }, 1000);
        }

        function captureFrameAndSend() {
            const video = document.getElementById('video');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Convert canvas to image data
            const data = canvas.toDataURL('image/jpeg');

            // Send image data to Flask server for processing
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: data })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').innerText = `Prediction: ${data.prediction}`;
            })
            .catch(error => {
                console.log("Error:", error);
            });
        }
    </script>
</body>
</html>
''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        # Decode base64 image data
        img_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Process image using MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        prediction = "Unknown"
        confidence = 0.0
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                keypoints = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                prediction, confidence = predict_label(keypoints)

        return jsonify({'prediction': prediction, 'confidence': confidence})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'prediction': 'Error', 'confidence': 0.0}), 500

if __name__ == '__main__':
    app.run(debug=True)
