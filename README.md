# MoodMix AI - Multimodal Emotion-Aware Music Recommender

MoodMix AI is a project that recommends music based on your current mood, detected from both your facial expression (via webcam) and your text input (mood diary).

## Setup

1.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Setup**:
    -   Download the **FER2013** dataset (or similar) and place it in `data/fer2013/`.
    -   Download a text emotion dataset and place it in `data/text_emotion/`.

## How to Run

1.  **Run the Streamlit App**:
    ```bash
    streamlit run src/app/streamlit_app.py
    ```

## Modules

-   **CV Emotion**: Detects face and predicts emotion using a CNN.
-   **NLP Emotion**: Predicts emotion from text input.
-   **Fusion**: Combines CV and NLP predictions.
-   **Recommender**: Selects songs based on the fused mood.
