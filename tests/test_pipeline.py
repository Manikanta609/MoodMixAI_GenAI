import sys
import os
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cv_emotion.infer_cv_emotion import CVEmotionInference
from src.nlp_emotion.infer_nlp_emotion import NLPEmotionInference
from src.fusion.fuse_mood import fuse_mood
from src.recommender.recommender import MusicRecommender

def test_pipeline():
    print("=== Testing MoodMix AI Pipeline ===")
    
    # 1. Test CV (with dummy image)
    print("\n[1] Testing CV Module...")
    cv_infer = CVEmotionInference("models/cv_emotion_model.pth")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a face-like rectangle so detection might work (or fail gracefully)
    cv2.rectangle(dummy_frame, (100, 100), (300, 300), (255, 255, 255), -1)
    
    cv_probs, _ = cv_infer.predict(dummy_frame)
    if cv_probs:
        print("CV Prediction:", cv_probs)
    else:
        print("CV: No face detected (Expected for dummy image if detector is strict)")
        # Mock probs for fusion test
        cv_probs = {"happy": 0.8, "neutral": 0.2}

    # 2. Test NLP
    print("\n[2] Testing NLP Module...")
    nlp_infer = NLPEmotionInference()
    text = "I am feeling absolutely wonderful today!"
    nlp_probs = nlp_infer.predict(text)
    print(f"Text: '{text}'")
    print("NLP Prediction:", nlp_probs)

    # 3. Test Fusion
    print("\n[3] Testing Fusion...")
    fusion_result = fuse_mood(cv_probs, nlp_probs)
    print("Fusion Result:", fusion_result)
    
    # 4. Test Recommender
    print("\n[4] Testing Recommender...")
    recommender = MusicRecommender("songs.csv")
    mood = fusion_result['final_label']
    songs = recommender.recommend(mood)
    print(f"Recommended songs for '{mood}':")
    for s in songs:
        print(f"- {s['title']} by {s['artist']}")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_pipeline()
