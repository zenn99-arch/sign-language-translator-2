import cv2
import numpy as np
import os
import mediapipe as mp
import xml.etree.ElementTree as ET

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.75)
cap = cv2.VideoCapture(0)

# Specify the folder path for the specific sign/gesture class
folder = r"C:\Users\ASUS\Desktop\sign language translator\data\hello"

# Create the folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

# Set the label as a constant
LABEL = "Hello"
counter = 0

def create_xml_annotation(folder, filename, img_shape, keypoints, class_name):
    annotation = ET.Element("annotation")
    
    folder_el = ET.SubElement(annotation, "folder")
    folder_el.text = folder
    
    filename_el = ET.SubElement(annotation, "filename")
    filename_el.text = filename
    
    size_el = ET.SubElement(annotation, "size")
    width_el = ET.SubElement(size_el, "width")
    width_el.text = str(img_shape[1])
    height_el = ET.SubElement(size_el, "height")
    
    if len(img_shape) == 3:  # Check if the image has depth (channels)
        depth_el = ET.SubElement(size_el, "depth")
        depth_el.text = str(img_shape[2])

    object_el = ET.SubElement(annotation, "object")
    name_el = ET.SubElement(object_el, "name")
    name_el.text = class_name

    keypoints_el = ET.SubElement(object_el, "keypoints")
    for i, (x, y, z) in enumerate(keypoints):
        kp_el = ET.SubElement(keypoints_el, f"kp{i}")
        kp_el.text = f"{x},{y},{z}"

    tree = ET.ElementTree(annotation)
    xml_filename = os.path.join(folder, filename.replace(".jpg", ".xml"))
    tree.write(xml_filename)

    print(f'Annotation saved as: {xml_filename}')

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        print("Hand landmarks detected")
        keypoints_list = []
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Collect keypoints
            keypoints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            keypoints_list.append(keypoints)

        # Check if keypress is detected
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            print("Key 's' pressed, saving data...")

            # Save the image in the specified folder with a sequential filename
            img_filename = f'{folder}/Image_{counter}.jpg'
            print(f"Saving image to: {img_filename}")
            cv2.imwrite(img_filename, img)

            # Save annotation in XML format with keypoints and the constant label
            create_xml_annotation(folder, f'Image_{counter}.jpg', img.shape, keypoints, class_name=LABEL)
            print(f'Sample {counter} captured.')

    else:
        print("No hand landmarks detected")

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
TF_ENABLE_ONEDNN_OPTS=0