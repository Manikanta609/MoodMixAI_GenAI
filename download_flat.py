from huggingface_hub import snapshot_download
import os

def download_flat():
    # Define flat paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, "models")
    
    cv_model_path = os.path.join(models_dir, "cv_model_flat")
    nlp_model_path = os.path.join(models_dir, "nlp_model_flat")
    
    print(f"Downloading CV Model to flat dir: {cv_model_path}")
    try:
        snapshot_download(
            repo_id="dima806/facial_emotions_image_detection", 
            local_dir=cv_model_path,
            local_dir_use_symlinks=False  # CRITICAL: No symlinks on Windows
        )
        print("CV Model downloaded successfully!")
    except Exception as e:
        print(f"FAILED to download CV Model: {e}")

    print(f"Downloading NLP Model to flat dir: {nlp_model_path}")
    try:
        snapshot_download(
            repo_id="bhadresh-savani/distilbert-base-uncased-emotion", 
            local_dir=nlp_model_path,
            local_dir_use_symlinks=False # CRITICAL: No symlinks on Windows
        )
        print("NLP Model downloaded successfully!")
    except Exception as e:
        print(f"FAILED to download NLP Model: {e}")

if __name__ == "__main__":
    download_flat()
