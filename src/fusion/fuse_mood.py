import numpy as np

def fuse_mood(cv_probs, nlp_probs=None):
    """
    Fuses CV and NLP emotion probabilities.
    
    Args:
        cv_probs (dict): {emotion: prob} from CV model.
                         Emotions: angry, disgust, fear, happy, sad, surprise, neutral
        nlp_probs (dict): {emotion: prob} from NLP model.
                          Emotions: joy, sadness, anger, fear, love, surprise
    
    Returns:
        result (dict): {
            "final_label": str,
            "final_vector": dict,
            "confidence": float
        }
    """
    # Target classes (CV classes are the master list)
    target_classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    
    # Initialize final vector
    final_vector = {cls: 0.0 for cls in target_classes}
    
    # If no CV probs (e.g. face not detected), rely on NLP or return neutral
    if not cv_probs:
        if nlp_probs:
            # Map NLP to Target
            mapped_nlp = _map_nlp_to_target(nlp_probs, target_classes)
            return _finalize(mapped_nlp)
        else:
            # No input at all
            return {"final_label": "neutral", "final_vector": final_vector, "confidence": 0.0}

    # If CV exists but no NLP
    if not nlp_probs:
        return _finalize(cv_probs)

    # Both exist: Fuse
    # Weight: 0.6 CV, 0.4 NLP (as per requirements)
    w_cv = 0.6
    w_nlp = 0.4
    
    mapped_nlp = _map_nlp_to_target(nlp_probs, target_classes)
    
    for cls in target_classes:
        val_cv = cv_probs.get(cls, 0.0)
        val_nlp = mapped_nlp.get(cls, 0.0)
        final_vector[cls] = w_cv * val_cv + w_nlp * val_nlp
        
    return _finalize(final_vector)

def _map_nlp_to_target(nlp_probs, target_classes):
    """
    Maps NLP emotions to Target emotions.
    NLP (tweet_emotions): empty, sadness, enthusiasm, neutral, worry, surprise, love, fun, hate, happiness, boredom, relief, anger
    Target (FER): angry, disgust, fear, happy, sad, surprise, neutral
    """
    mapped = {cls: 0.0 for cls in target_classes}
    
    for emo, prob in nlp_probs.items():
        # Happy-like
        if emo in ['happiness', 'enthusiasm', 'fun', 'love', 'relief', 'joy']:
            mapped['happy'] += prob
        # Sad-like
        elif emo in ['sadness', 'empty', 'boredom', 'sad']:
            mapped['sad'] += prob
        # Angry-like
        elif emo in ['anger', 'hate', 'angry']:
            mapped['angry'] += prob
        # Fear-like
        elif emo in ['worry', 'fear']:
            mapped['fear'] += prob
        # Surprise-like
        elif emo in ['surprise']:
            mapped['surprise'] += prob
        # Neutral-like
        elif emo in ['neutral']:
            mapped['neutral'] += prob
        else:
            mapped['neutral'] += prob # Fallback
            
    return mapped

def _finalize(vector):
    # Normalize (optional, but good practice)
    total = sum(vector.values())
    if total > 0:
        vector = {k: v / total for k, v in vector.items()}
        
    # Get max
    final_label = max(vector, key=vector.get)
    confidence = vector[final_label]
    
    return {
        "final_label": final_label,
        "final_vector": vector,
        "confidence": confidence
    }
