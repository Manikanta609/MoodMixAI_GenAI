import cv2
import numpy as np
import torch
from transformers import pipeline
from PIL import Image
import mediapipe as mp
import os

class CVEmotionInference:
    def __init__(self, model_path=None):
        """
        Initializes the CV emotion inference.
        Uses MediaPipe for face detection and a Hugging Face Transformer for emotion recognition.
        """
        self.device = 0 if torch.cuda.is_available() else -1
        # print(f"Loading CV Emotion model on device: {self.device}...")
        
        # Load Face Detector (MediaPipe)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        
        # Load Emotion Classifier (Hugging Face)
        # Using a robust ViT model fine-tuned for facial emotions
        # Construct absolute path to the flat model directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        model_path = os.path.join(project_root, "models", "cv_model_flat")
        
        # print(f"Loading CV Model from: {model_path}")

        self.classifier = pipeline(
            "image-classification", 
            model=model_path, 
            device=self.device,
            model_kwargs={"local_files_only": True}
        )
        
        # Mapping from model labels to our standard 7 emotions
        # Model labels: ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']
        self.target_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def predict(self, image):
        """
        Predicts emotion from an image.
        Args:
            image (numpy.ndarray): Input image (BGR from OpenCV).
        Returns:
            probs (dict): Dictionary of emotion probabilities.
            face_roi (numpy.ndarray): The cropped face image.
        """
        if image is None:
            return None, None

        # 1. Face Detection with MediaPipe
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.detections:
            return None, None
            
        # Get the first face
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        
        # Padding
        padding = 0
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(iw - x, w + 2*padding)
        h = min(ih - y, h + 2*padding)
        
        face_roi = image[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return None, None

        # 2. Emotion Recognition with Transformer
        # Convert to PIL Image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        
        try:
            results = self.classifier(pil_image)
            # results: [{'label': 'happy', 'score': 0.9}, ...]
            
            # Normalize to our dict format
            probs = {emo: 0.0 for emo in self.target_emotions}
            for res in results:
                label = res['label'].lower()
                score = res['score']
                
                # Map labels if necessary (this model uses standard names mostly)
                if label == 'sadness': label = 'sad'
                if label == 'happiness': label = 'happy'
                if label == 'anger': label = 'angry'
                
                if label in probs:
                    probs[label] = score
                    
            return probs, face_roi
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return None, face_roi
