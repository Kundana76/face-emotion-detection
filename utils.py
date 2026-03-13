import cv2
import numpy as np
import os
from datetime import datetime
from gtts import gTTS
import pygame
import tempfile
import logging


class Utils:
    """Utility functions for Emotion Detection System"""

    def __init__(self):

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize audio safely
        self.audio_enabled = False
        try:
            pygame.mixer.init()
            self.audio_enabled = True
        except Exception as e:
            self.logger.warning(f"Audio initialization failed: {e}")

    # -----------------------------
    # Draw bounding box
    # -----------------------------
    def draw_bounding_box(self, frame, bbox, emotion, confidence, color=(0, 255, 0)):

        x1, y1, x2, y2 = bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{emotion}: {confidence:.2f}"

        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        cv2.rectangle(
            frame,
            (x1, y1 - label_height - 10),
            (x1 + label_width, y1),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return frame

    # -----------------------------
    # Emotion probability bars
    # -----------------------------
    def draw_probability_bars(self, frame, probabilities, position=(10, 10)):

        x, y = position
        bar_height = 20
        bar_width = 200
        spacing = 25

        colors = {
            'Angry': (0, 0, 255),
            'Disgust': (0, 255, 255),
            'Fear': (128, 0, 128),
            'Happy': (0, 255, 0),
            'Sad': (255, 0, 0),
            'Surprise': (255, 255, 0),
            'Neutral': (128, 128, 128)
        }

        for i, (emotion, prob) in enumerate(probabilities.items()):

            y_pos = y + i * spacing

            # Background bar
            cv2.rectangle(
                frame,
                (x, y_pos),
                (x + bar_width, y_pos + bar_height),
                (200, 200, 200),
                -1
            )

            prob_width = int(bar_width * prob)

            # Emotion bar
            cv2.rectangle(
                frame,
                (x, y_pos),
                (x + prob_width, y_pos + bar_height),
                colors.get(emotion, (0, 255, 0)),
                -1
            )

            cv2.putText(
                frame,
                f"{emotion}: {prob:.2f}",
                (x + bar_width + 10, y_pos + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return frame

    # -----------------------------
    # Save detected face
    # -----------------------------
    def save_face_dataset(self, face_img, emotion, confidence, save_dir="data/captured_faces"):

        emotion_dir = os.path.join(save_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{emotion}_{confidence:.2f}_{timestamp}.jpg"
        filepath = os.path.join(emotion_dir, filename)

        cv2.imwrite(filepath, face_img)

        return filepath

    # -----------------------------
    # Screenshot capture
    # -----------------------------
    def capture_screenshot(self, frame, save_dir="data/screenshots"):

        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"screenshot_{timestamp}.jpg"

        filepath = os.path.join(save_dir, filename)

        cv2.imwrite(filepath, frame)

        return filepath

    # -----------------------------
    # Voice feedback
    # -----------------------------
    def voice_feedback(self, emotion):

        if not self.audio_enabled:
            return

        try:

            text = f"Detected {emotion} emotion"

            tts = gTTS(text=text, lang="en")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tmp_path = fp.name

            tts.save(tmp_path)

            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)

            pygame.mixer.music.stop()

            os.remove(tmp_path)

        except Exception as e:
            self.logger.error(f"Voice feedback error: {e}")

    # -----------------------------
    # Frame optimization
    # -----------------------------
    def optimize_frame(self, frame, target_size=None):

        if frame is None:
            return None

        if target_size:
            frame = cv2.resize(frame, target_size)

        return frame

    # -----------------------------
    # Emotion heatmap
    # -----------------------------
    def create_emotion_heatmap(self, frame, faces_data):

        if frame is None or not faces_data:
            return frame

        heatmap = np.zeros(frame.shape[:2], dtype=np.float32)

        for face in faces_data:

            bbox = face.get("bbox")
            confidence = face.get("confidence", 0)

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            radius = max(x2 - x1, y2 - y1) // 2

            Y, X = np.ogrid[:frame.shape[0], :frame.shape[1]]

            dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

            mask = dist <= radius

            heatmap[mask] += confidence * (1 - dist[mask] / radius)

        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        result = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)

        return result