import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

class FaceEmotionDetector:
    def __init__(self):
        # Load face detection cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Emotion labels
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Initialize video capture
        self.cap = None
        
    def preprocess_face(self, face_img):
        """Preprocess face image for emotion detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # Resize to model input size
        resized = cv2.resize(gray, (48, 48))
        # Normalize pixel values
        normalized = resized / 255.0
        # Reshape for model input
        preprocessed = np.expand_dims(np.expand_dims(normalized, -1), 0)
        return preprocessed
    
    def detect_emotion(self, image):
        """Detect emotions in an image"""
        # Convert PIL Image to cv2 format if necessary
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        # Process each detected face
        for (x, y, w, h) in faces:
            face_roi = image[y:y+h, x:x+w]
            # Preprocess face for emotion detection
            preprocessed_face = self.preprocess_face(face_roi)
            
            # TODO: Add emotion prediction when model is implemented
            # For now, return a placeholder result
            emotion = 'Neutral'  # Placeholder
            confidence = 1.0     # Placeholder
            
            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence
            })
            
            # Draw rectangle around face
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Add emotion label
            cv2.putText(image, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return image, results
    
    def start_video_capture(self):
        """Start video capture for real-time emotion detection"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open video capture device")
            
    def get_video_frame(self):
        """Get frame from video capture with emotion detection"""
        if self.cap is None:
            return None, None
            
        ret, frame = self.cap.read()
        if not ret:
            return None, None
            
        # Process frame for emotion detection
        processed_frame, results = self.detect_emotion(frame)
        return processed_frame, results
    
    def stop_video_capture(self):
        """Stop video capture"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __del__(self):
        """Cleanup resources"""
        self.stop_video_capture()