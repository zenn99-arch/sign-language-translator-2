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
    keypoints = np.array(keypoints).flatten()
    keypoints = keypoints / np.max(keypoints)
    if keypoints.shape[0] != np.prod(input_shape):
        raise ValueError(f"Keypoints shape mismatch: expected {input_shape}, got {keypoints.shape}")
    return keypoints

def predict_label(keypoints):
    input_shape = model.input_shape[1:]
    keypoints = preprocess_keypoints(keypoints, input_shape)
    keypoints = np.expand_dims(keypoints, axis=0)
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
    <title>I-sravya - Sign Language Detection and Speech-to-Text</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f0f4f8;
            margin: 0;
            padding: 0;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            background-color: #4CAF50;
            color: white;
            padding: 20px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .content {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
        }
        .video-container {
            flex: 1;
            min-width: 300px;
            margin-right: 20px;
        }
        #video {
            width: 100%;
            max-width: 640px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }
        .controls {
            flex: 1;
            min-width: 300px;
        }
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 10px 5px;
            border: none;
            color: white;
        }
        .btn-primary {
            background-color: #4CAF50;
        }
        .btn-primary:hover {
            background-color: #45a049;
        }
        .btn-warning {
            background-color: #ff9800;
        }
        .btn-warning:hover {
            background-color: #e68a00;
        }
        .btn i {
            margin-right: 10px;
        }
        #result, #speechToTextResult {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            font-size: 18px;
        }
        .speech-to-text {
            margin-top: 40px;
        }
        .speech-to-text h2 {
            color: #4CAF50;
        }
        #micAnimation {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background-color: #4CAF50;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 20px auto;
            animation: pulse 1.5s infinite;
            display: none;
        }
        @keyframes pulse {
            0% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);
            }
            100% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1><i class="fas fa-sign-language"></i> I-sravya</h1>
    </div>
    <div class="container">
        <div class="content">
            <div class="video-container">
                <video id="video" autoplay></video>
            </div>
            <div class="controls">
                <button id="startButton" class="btn btn-primary">
                    <i class="fas fa-play"></i> Start Detection
                </button>
                <button id="toggleVoiceBtn" class="btn btn-warning">
                    <i class="fas fa-volume-up"></i> Enable Voice
                </button>
                <div id="result"></div>
                
                <div class="speech-to-text">
                    <h2><i class="fas fa-microphone"></i> Speech-to-Text</h2>
                    <button id="startSpeechToTextButton" class="btn btn-primary">
                        <i class="fas fa-microphone-alt"></i> Start Speaking
                    </button>
                    <div id="micAnimation">
                        <i class="fas fa-microphone" style="color: white; font-size: 24px;"></i>
                    </div>
                    <div id="speechToTextResult"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let isDetectionActive = false;
        let voiceEnabled = false;
        let detectionInterval;
        let lastPrediction = "";

        document.getElementById('startButton').addEventListener('click', function() {
            if (!isDetectionActive) {
                startDetection();
                this.innerHTML = '<i class="fas fa-stop"></i> Stop Detection';
                this.classList.remove('btn-primary');
                this.classList.add('btn-warning');
            } else {
                stopDetection();
                this.innerHTML = '<i class="fas fa-play"></i> Start Detection';
                this.classList.remove('btn-warning');
                this.classList.add('btn-primary');
            }
            isDetectionActive = !isDetectionActive;
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
            detectionInterval = setInterval(() => captureFrameAndSend(), 1000);
        }

        function stopDetection() {
            clearInterval(detectionInterval);
            const video = document.getElementById('video');
            const stream = video.srcObject;
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            video.srcObject = null;
        }

        document.getElementById('toggleVoiceBtn').addEventListener('click', function() {
            voiceEnabled = !voiceEnabled;
            this.innerHTML = voiceEnabled ? '<i class="fas fa-volume-mute"></i> Disable Voice' : '<i class="fas fa-volume-up"></i> Enable Voice';
        });

        function captureFrameAndSend() {
            const video = document.getElementById('video');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
            const data = canvas.toDataURL('image/jpeg');

            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: data })
            })
            .then(response => response.json())
            .then(data => {
                const resultElement = document.getElementById('result');
                resultElement.innerHTML = `<strong>Prediction:</strong> ${data.prediction}`;

                if (voiceEnabled && 'speechSynthesis' in window && data.prediction !== 'Unknown' && data.prediction !== 'Error') {
                    if (data.prediction !== lastPrediction) {
                        const utterance = new SpeechSynthesisUtterance(data.prediction);
                        window.speechSynthesis.speak(utterance);
                        lastPrediction = data.prediction;
                    }
                }
            })
            .catch(error => {
                console.log("Error:", error);
            });
        }

        document.getElementById('startSpeechToTextButton').addEventListener('click', function() {
            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = 'en-US';
            recognition.interimResults = false;

            const micAnimation = document.getElementById('micAnimation');
            micAnimation.style.display = 'flex';
            this.disabled = true;

            recognition.start();

            recognition.onresult = function(event) {
                const speechResult = event.results[0][0].transcript;
                document.getElementById('speechToTextResult').innerHTML = `<strong>You said:</strong> ${speechResult}`;
                micAnimation.style.display = 'none';
                document.getElementById('startSpeechToTextButton').disabled = false;
            };

            recognition.onerror = function(event) {
                document.getElementById('speechToTextResult').innerHTML = '<strong>Error occurred in recognition:</strong> ' + event.error;
                micAnimation.style.display = 'none';
                document.getElementById('startSpeechToTextButton').disabled = false;
            };

            recognition.onend = function() {
                micAnimation.style.display = 'none';
                document.getElementById('startSpeechToTextButton').disabled = false;
            };
        });
    </script>
</body>
</html>
''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        prediction = "Unknown"
        confidence = 0.0
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                keypoints = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                prediction, confidence = predict_label(keypoints)
        
        # Convert confidence to native Python float
        confidence = float(confidence)

        return jsonify({'prediction': prediction, 'confidence': confidence})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'prediction': 'Error', 'confidence': 0.0}), 500

if __name__ == '__main__':
    app.run(debug=True)
