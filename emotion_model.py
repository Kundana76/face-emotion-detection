# src/emotion_model.py

import os
import logging
import numpy as np
import cv2

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models


class EmotionCNN:
    """CNN model for emotion classification"""

    def __init__(self, model_path='models/emotion_model.h5'):
        """
        Initialize emotion model

        Args:
            model_path: Path to trained model
        """

        self.logger = logging.getLogger(__name__)

        self.emotions = [
            'Angry',
            'Disgust',
            'Fear',
            'Happy',
            'Sad',
            'Surprise',
            'Neutral'
        ]

        self.input_size = (48, 48)

        # Load trained model if available
        if os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                self.logger.info(f"Model loaded successfully from {model_path}")
            except Exception as e:
                self.logger.warning(f"Error loading model: {e}")
                self.model = self.build_model()
        else:
            self.logger.warning("Model file not found. Creating new model.")
            self.model = self.build_model()

    def build_model(self):
        """Build CNN architecture for emotion recognition"""

        model = models.Sequential([

            # First Convolution Block
            layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(48,48,1)),
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            # Second Block
            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            # Third Block
            layers.Conv2D(256, (3,3), activation='relu', padding='same'),
            layers.Conv2D(256, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            # Fourth Block
            layers.Conv2D(512, (3,3), activation='relu', padding='same'),
            layers.Conv2D(512, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            # Fully Connected Layers
            layers.Flatten(),

            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(len(self.emotions), activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.logger.info("New CNN model created")

        return model

    def preprocess_face(self, face_img):
        """
        Preprocess face image for model input
        """

        if face_img is None:
            return None

        # Convert to grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img

        # Resize
        resized = cv2.resize(gray, self.input_size)

        # Normalize
        normalized = resized.astype("float32") / 255.0

        # Add channel dimension
        expanded = np.expand_dims(normalized, axis=-1)

        # Add batch dimension
        batched = np.expand_dims(expanded, axis=0)

        return batched

    def predict_emotion(self, face_img):
        """
        Predict emotion from face image
        """

        processed = self.preprocess_face(face_img)

        if processed is None:
            return None, 0, {}

        predictions = self.model.predict(processed, verbose=0)[0]

        emotion_idx = np.argmax(predictions)

        confidence = float(predictions[emotion_idx])

        emotion = self.emotions[emotion_idx]

        all_probs = {
            self.emotions[i]: float(predictions[i])
            for i in range(len(self.emotions))
        }

        return emotion, confidence, all_probs

    def predict_multiple_faces(self, faces):
        """
        Predict emotions for multiple faces
        """

        if not faces:
            return []

        processed_faces = []

        for face in faces:
            p = self.preprocess_face(face)
            if p is not None:
                processed_faces.append(p)

        if not processed_faces:
            return []

        processed_faces = np.vstack(processed_faces)

        predictions = self.model.predict(processed_faces, verbose=0)

        results = []

        for pred in predictions:

            emotion_idx = np.argmax(pred)

            emotion = self.emotions[emotion_idx]

            confidence = float(pred[emotion_idx])

            all_probs = {
                self.emotions[i]: float(pred[i])
                for i in range(len(self.emotions))
            }

            results.append({
                "emotion": emotion,
                "confidence": confidence,
                "probabilities": all_probs
            })

        return results