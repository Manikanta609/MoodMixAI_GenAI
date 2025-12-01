import os
import sys

# Disable progress bars GLOBALLY (Must be before other imports)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Patch tqdm to be silent (must be a class to allow inheritance)
try:
    from tqdm import tqdm
    
    class SilentTQDM:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, *args, **kwargs):
            pass
        @classmethod
        def write(cls, *args, **kwargs):
            pass
        @classmethod
        def get_lock(cls):
            # Return a dummy context manager or lock
            class DummyLock:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyLock()
        @classmethod
        def set_lock(cls, lock):
            pass
            
    tqdm.pandas = lambda *args, **kwargs: None
    import tqdm.auto as tqdm_auto
    tqdm_auto.tqdm = SilentTQDM
    import tqdm.std as tqdm_std
    tqdm_std.tqdm = SilentTQDM
except ImportError:
    pass

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from transformers import utils
import logging

# Disable transformers logging
utils.logging.set_verbosity_error()

# Add project root to path so we can import src modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# Set local cache directory for models
cache_dir = os.path.join(PROJECT_ROOT, "model_cache")
os.environ["HF_HOME"] = cache_dir

# Auto-download models if missing (for Streamlit Cloud)
from huggingface_hub import snapshot_download

def check_and_download_models():
    models_dir = os.path.join(PROJECT_ROOT, "models")
    cv_model_path = os.path.join(models_dir, "cv_model_flat")
    nlp_model_path = os.path.join(models_dir, "nlp_model_flat")
    
    try:
        if not os.path.exists(cv_model_path) or not os.listdir(cv_model_path):
            with st.spinner("Downloading CV Model (this may take a minute)..."):
                snapshot_download(repo_id="dima806/facial_emotions_image_detection", local_dir=cv_model_path)
                
        if not os.path.exists(nlp_model_path) or not os.listdir(nlp_model_path):
            with st.spinner("Downloading NLP Model (this may take a minute)..."):
                snapshot_download(repo_id="bhadresh-savani/distilbert-base-uncased-emotion", local_dir=nlp_model_path)
    except Exception as e:
        st.error(f"Failed to download models: {e}")
        # Don't stop, let it try to load and fail gracefully later if needed
        pass

check_and_download_models()

from src.cv_emotion.infer_cv_emotion import CVEmotionInference
from src.nlp_emotion.infer_nlp_emotion import NLPEmotionInference
from src.fusion.fuse_mood import fuse_mood
from src.recommender.recommender import MusicRecommender

# Page Config
st.set_page_config(page_title="MoodMix AI", page_icon="🎵", layout="wide")

# Load Custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = os.path.join(os.path.dirname(__file__), 'style.css')
if os.path.exists(css_path):
    load_css(css_path)

# Load Models (Cached)
@st.cache_resource
def load_cv_model_v2():
    return CVEmotionInference(os.path.join(PROJECT_ROOT, "models", "cv_emotion_model.pth"))

@st.cache_resource
def load_nlp_model_v2():
    return NLPEmotionInference(os.path.join(PROJECT_ROOT, "models", "text_emotion_model.pth"))

@st.cache_resource
def load_recommender_v2():
    return MusicRecommender(os.path.join(PROJECT_ROOT, "songs.csv"))

try:
    cv_model = load_cv_model_v2()
    nlp_model = load_nlp_model_v2()
    recommender = load_recommender_v2()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Helper: Radar Chart
def plot_radar_chart(mood_vector):
    categories = list(mood_vector.keys())
    values = list(mood_vector.values())
    
    # Close the loop
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Mood Profile',
        line_color='#00d2ff'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='rgba(255, 255, 255, 0.2)',
                tickfont=dict(color='white')
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# Sidebar
st.sidebar.title("MoodMix AI 🎵")
st.sidebar.markdown("---")
st.sidebar.info(
    "**How it works:**\n"
    "1. Capture your face via webcam.\n"
    "2. (Optional) Write how you feel.\n"
    "3. Get a personalized playlist!"
)

# Main Layout
st.markdown("<h1 style='text-align: center;'>MoodMix AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa;'>Multimodal Emotion-Aware Music Recommender</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("📸 Facial Emotion")
    
    tab1, tab2 = st.tabs(["Webcam", "Upload Image"])
    
    img_file_buffer = None
    
    with tab1:
        camera_buffer = st.camera_input("Take a picture")
        if camera_buffer:
            img_file_buffer = camera_buffer
            
    with tab2:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img_file_buffer = uploaded_file
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
    
    cv_probs = None
    if img_file_buffer is not None:
        # Convert to CV2 format
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Predict
        cv_probs, face_roi = cv_model.predict(cv2_img)
        
        if cv_probs:
            st.success("Face detected!")
        else:
            st.warning("No face detected.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("📝 Mood Diary")
    user_text = st.text_area("How are you feeling right now?", height=100, placeholder="I'm feeling amazing today...")
    
    nlp_probs = None
    if user_text:
        nlp_probs = nlp_model.predict(user_text)
    st.markdown('</div>', unsafe_allow_html=True)

# Fusion & Recommendation
if cv_probs or nlp_probs:
    st.markdown("---")
    
    # Fuse
    fusion_result = fuse_mood(cv_probs, nlp_probs)
    final_mood = fusion_result['final_label']
    confidence = fusion_result['confidence']
    
    # Results Section
    r_col1, r_col2 = st.columns([1, 2])
    
    with r_col1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("Mood Analysis")
        st.metric(label="Dominant Mood", value=final_mood.upper(), delta=f"{confidence*100:.1f}% Conf.")
        
        # Radar Chart
        fig = plot_radar_chart(fusion_result['final_vector'])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with r_col2:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader(f"🎵 Recommended for '{final_mood.title()}'")
        
        if st.button("Generate Playlist", type="primary", key="gen_btn"):
            songs = recommender.recommend(final_mood)
            
            if songs:
                for i, song in enumerate(songs):
                    with st.expander(f"{i+1}. {song['title']} - {song['artist']}", expanded=True):
                        st.write(f"**Mood:** {song['mood_tag'].title()}")
                        if "youtube.com" in song['url'] or "youtu.be" in song['url']:
                            st.video(song['url'])
                        else:
                            st.markdown(f"[Listen on External Player]({song['url']})")
            else:
                st.info("No songs found for this mood.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Please capture an image or enter text to get started.")
