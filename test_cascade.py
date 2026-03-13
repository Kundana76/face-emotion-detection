import cv2

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error: Cascade file not loaded")
else:
    print("Cascade loaded successfully")