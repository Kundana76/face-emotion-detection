# training/train_model.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class EmotionModelTrainer:
    """Train emotion recognition model on FER2013 dataset"""

    def __init__(self, data_path="fer2013.csv", img_size=(48, 48)):
        self.data_path = data_path
        self.img_size = img_size

        self.emotions = [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Sad",
            "Surprise",
            "Neutral",
        ]

        os.makedirs("models", exist_ok=True)
        os.makedirs("training/plots", exist_ok=True)
        os.makedirs("training", exist_ok=True)

    # --------------------------------------------------

    def load_fer2013(self):
        """Load FER2013 dataset"""

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}\nDownload FER2013 dataset first."
            )

        print("Loading FER2013 dataset...")

        df = pd.read_csv(self.data_path)

        pixels = df["pixels"].tolist()

        X = np.array(
            [np.fromstring(pixel, dtype="float32", sep=" ") for pixel in pixels]
        )

        X = X.reshape(-1, 48, 48, 1)

        X = X / 255.0

        y = df["emotion"].values

        print(f"Loaded {len(X)} images")

        return X, y

    # --------------------------------------------------

    def create_model(self):
        """Create CNN model"""

        model = models.Sequential(

            [
                layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                              input_shape=(48, 48, 1)),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Flatten(),

                layers.Dense(512, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),

                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),

                layers.Dense(len(self.emotions), activation="softmax"),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    # --------------------------------------------------

    def train(self, epochs=50, batch_size=64):

        X, y = self.load_fer2013()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        print("Training samples:", len(X_train))
        print("Validation samples:", len(X_val))
        print("Test samples:", len(X_test))

        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
        )

        model = self.create_model()

        model.summary()

        callbacks = [

            keras.callbacks.ModelCheckpoint(
                "models/best_model.h5",
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),

            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            ),

            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
            ),

            keras.callbacks.CSVLogger("training/training_log.csv"),
        ]

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
        )

        test_loss, test_acc = model.evaluate(X_test, y_test)

        print("\nTest Accuracy:", test_acc)

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred_classes,
                                    target_names=self.emotions))

        self.plot_training_history(history)

        self.plot_confusion_matrix(y_test, y_pred_classes)

        model.save("models/emotion_model.h5")

        print("\nFinal model saved: models/emotion_model.h5")

        return model, history

    # --------------------------------------------------

    def plot_training_history(self, history):

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"])

        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"])

        plt.tight_layout()
        plt.savefig("training/plots/training_history.png")
        plt.close()

    # --------------------------------------------------

    def plot_confusion_matrix(self, y_true, y_pred):

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.emotions,
            yticklabels=self.emotions,
        )

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.tight_layout()

        plt.savefig("training/plots/confusion_matrix.png")
        plt.close()


# ------------------------------------------------------

if __name__ == "__main__":

    trainer = EmotionModelTrainer(data_path="fer2013.csv")

    trainer.train(epochs=50, batch_size=64)