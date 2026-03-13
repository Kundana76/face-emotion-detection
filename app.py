# app.py

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import os
import sys

# Add src folder
sys.path.append("src")

from face_detector import FaceDetector
from emotion_model import EmotionCNN
from data_logger import EmotionLogger
from analytics import EmotionAnalytics
from src.utils import Utils


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Face Emotion Detection",
    page_icon="😊",
    layout="wide"
)


# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False


# ---------------------------------------------------
# LOAD COMPONENTS
# ---------------------------------------------------

@st.cache_resource
def load_components():

    face_detector = FaceDetector(method="haar")

    emotion_model = EmotionCNN()

    logger = EmotionLogger()

    utils = Utils()

    return face_detector, emotion_model, logger, utils


face_detector, emotion_model, logger, utils = load_components()


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("😊 Advanced Face Emotion Detection System")
st.write("Real-time emotion recognition using AI")


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

mode = st.sidebar.radio(
    "Select Mode",
    ["Webcam Detection", "Image Upload", "Analytics Dashboard"]
)


# ---------------------------------------------------
# WEBCAM MODE
# ---------------------------------------------------

if mode == "Webcam Detection":

    st.header("Live Webcam Detection")

    start = st.sidebar.button("Start Webcam")
    stop = st.sidebar.button("Stop Webcam")

    if start:
        st.session_state.camera_active = True

    if stop:
        st.session_state.camera_active = False

    frame_placeholder = st.empty()

    if st.session_state.camera_active:

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not open webcam")
        else:

            for _ in range(200):  # limited frames to prevent freeze

                ret, frame = cap.read()

                if not ret:
                    st.error("Camera error")
                    break

                frame = cv2.flip(frame, 1)

                faces = face_detector.detect_faces(frame)

                for bbox in faces:

                    face_img = face_detector.extract_face_region(frame, bbox)

                    if face_img.size > 0:

                        emotion, confidence, probs = emotion_model.predict_emotion(face_img)

                        frame = utils.draw_bounding_box(
                            frame,
                            bbox,
                            emotion,
                            confidence
                        )

                        logger.log_emotion(emotion, confidence)

                        st.session_state.emotion_history.append({
                            "time": datetime.now(),
                            "emotion": emotion,
                            "confidence": confidence
                        })

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame_placeholder.image(frame, channels="RGB")

            cap.release()


    if st.session_state.emotion_history:

        df = pd.DataFrame(st.session_state.emotion_history)

        st.subheader("Recent Detections")

        st.dataframe(df.tail(10))

        st.subheader("Emotion Distribution")

        st.bar_chart(df["emotion"].value_counts())


# ---------------------------------------------------
# IMAGE UPLOAD MODE
# ---------------------------------------------------

elif mode == "Image Upload":

    st.header("Analyze Image Emotion")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image")

        image_np = np.array(image)

        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        if st.button("Analyze Emotion"):

            faces = face_detector.detect_faces(image_bgr)

            if len(faces) == 0:
                st.warning("No faces detected")

            else:

                for bbox in faces:

                    face_img = face_detector.extract_face_region(image_bgr, bbox)

                    emotion, confidence, probs = emotion_model.predict_emotion(face_img)

                    image_bgr = utils.draw_bounding_box(
                        image_bgr,
                        bbox,
                        emotion,
                        confidence
                    )

                    st.success(f"Emotion: {emotion} ({confidence:.2%})")

                    prob_df = pd.DataFrame({
                        "Emotion": list(probs.keys()),
                        "Probability": list(probs.values())
                    })

                    st.bar_chart(prob_df.set_index("Emotion"))

                result = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                st.image(result, caption="Result")


# ---------------------------------------------------
# ANALYTICS DASHBOARD
# ---------------------------------------------------

elif mode == "Analytics Dashboard":

    st.header("Emotion Analytics Dashboard")

    stats_df = logger.get_statistics("all")

    if stats_df.empty:

        st.info("No emotion data available yet")

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", len(stats_df))
        col2.metric("Unique Emotions", stats_df["emotion"].nunique())
        col3.metric("Avg Confidence", f"{stats_df['confidence'].mean():.2%}")

        st.subheader("Emotion Distribution")

        st.bar_chart(stats_df["emotion"].value_counts())

        st.subheader("Recent Logs")

        st.dataframe(stats_df.tail(50))

        if st.button("Export CSV"):

            path = logger.export_report("csv")

            st.success(f"Saved to {path}")

            with open(path, "rb") as f:
                st.download_button(
                    "Download CSV",
                    f,
                    file_name="emotion_report.csv"
                )


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")
st.caption("Built with Streamlit + TensorFlow + OpenCV")