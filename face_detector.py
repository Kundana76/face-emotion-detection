# src/face_detector.py

import cv2
import numpy as np
import dlib
from imutils import face_utils
import logging

class FaceDetector:
    """Advanced face detection using multiple methods"""
    
    def __init__(self, method='haar', cascade_path='models/haarcascade_frontalface_default.xml'):
        """
        Initialize face detector
        
        Args:
            method: 'haar', 'dlib', or 'both'
            cascade_path: Path to Haar cascade XML
        """
        self.method = method
        self.logger = logging.getLogger(__name__)
        
        # Initialize Haar Cascade
        if method in ['haar', 'both']:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                self.logger.error("Failed to load Haar cascade")
                raise ValueError("Haar cascade file not found")
        
        # Initialize Dlib detector
        if method in ['dlib', 'both']:
            try:
                self.dlib_detector = dlib.get_frontal_face_detector()
                self.logger.info("Dlib detector initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Dlib: {e}")
                self.method = 'haar' if method == 'both' else method
    
    def detect_haar(self, frame):
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return [(x, y, x+w, y+h) for (x, y, w, h) in faces]
    
    def detect_dlib(self, frame):
        """Detect faces using Dlib"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with upsampling for better accuracy
        faces = self.dlib_detector(gray, 1)
        
        return [(f.left(), f.top(), f.right(), f.bottom()) for f in faces]
    
    def detect_faces(self, frame):
        """
        Detect faces using specified method
        
        Args:
            frame: Input image frame
            
        Returns:
            List of bounding boxes (x1, y1, x2, y2)
        """
        if self.method == 'haar':
            return self.detect_haar(frame)
        elif self.method == 'dlib':
            return self.detect_dlib(frame)
        elif self.method == 'both':
            # Combine both methods for better accuracy
            haar_faces = self.detect_haar(frame)
            dlib_faces = self.detect_dlib(frame)
            
            # Merge and remove duplicates
            all_faces = haar_faces + dlib_faces
            return self.merge_overlapping_faces(all_faces)
        
        return []
    
    def merge_overlapping_faces(self, faces, overlap_threshold=0.5):
        """Merge overlapping face detections"""
        if not faces:
            return []
        
        # Convert to numpy array for easier manipulation
        faces = np.array(faces)
        
        # Calculate areas
        areas = (faces[:, 2] - faces[:, 0]) * (faces[:, 3] - faces[:, 1])
        
        # Sort by area (largest first)
        indices = np.argsort(areas)[::-1]
        
        merged = []
        used = set()
        
        for i in indices:
            if i in used:
                continue
            
            current = faces[i]
            merged.append(current)
            
            # Mark overlapping faces as used
            for j in indices:
                if j in used or j == i:
                    continue
                
                other = faces[j]
                
                # Calculate IoU
                x1 = max(current[0], other[0])
                y1 = max(current[1], other[1])
                x2 = min(current[2], other[2])
                y2 = min(current[3], other[3])
                
                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    union = areas[i] + areas[j] - intersection
                    iou = intersection / union
                    
                    if iou > overlap_threshold:
                        used.add(j)
        
        return merged
    
    def extract_face_region(self, frame, bbox, margin=0.2):
        """
        Extract face region with margin
        
        Args:
            frame: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            margin: Margin percentage around face
            
        Returns:
            Cropped face image
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Calculate margin
        face_w = x2 - x1
        face_h = y2 - y1
        margin_x = int(face_w * margin)
        margin_y = int(face_h * margin)
        
        # Apply margin
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        # Extract face region
        face_region = frame[y1:y2, x1:x2]
        
        return face_region