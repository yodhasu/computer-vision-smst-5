import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from live_detection_lib import load_features, live_detection

# Load pre-trained model and features
global_features, labels, local_features = load_features()
conference_model = load_model("Conference.keras")

# Streamlit App
st.title("Live Detection System")
st.write("Upload an image or use the webcam to perform live detection.")

# File Uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Webcam Input
use_webcam = st.camera_input("Take a photo with your webcam")

if uploaded_file:
    # Load the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    annotated_frame, detected_label = live_detection(conference_model, global_features, local_features, labels, frame)

    # Display results
    st.image(annotated_frame, caption=f"Detected: {detected_label}", channels="RGB")

elif use_webcam:
    # Read image from the webcam input
    img = use_webcam
    frame = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    annotated_frame, detected_label = live_detection(conference_model, global_features, local_features, labels, frame)

    # Display results
    st.image(annotated_frame, caption=f"Detected: {detected_label}", channels="RGB")
