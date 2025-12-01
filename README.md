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
