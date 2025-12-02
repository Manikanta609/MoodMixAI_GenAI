import os

class NLPEmotionInference:
    def __init__(self, model_path="models/text_emotion_model.pth"):
        try:
            from transformers import pipeline
            import torch
            
            self.torch = torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.tokenizer = None
            self.classifier = None
            self.label_map = None
            self.idx_to_label = None
    
            if os.path.exists(model_path) and os.path.exists(model_path + ".labels"):
                # print(f"Loading custom NLP model from {model_path}...")
                try:
                    from .model import BERTForEmotion
                    from transformers import DistilBertTokenizer
                    
                    self.label_map = torch.load(model_path + ".labels")
                    self.idx_to_label = {i: label for label, i in self.label_map.items()}
                    
                    self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                    self.model = BERTForEmotion(num_classes=len(self.label_map))
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.to(self.device)
                    self.model.eval()
                    # print("Custom model loaded successfully.")
                except Exception as e:
                    # print(f"Error loading custom model: {e}. Falling back to pre-trained pipeline.")
                    self.model = None
            if self.model is None:
                # Pipeline inference
                # Construct absolute path to the flat model directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                model_path = os.path.join(project_root, "models", "nlp_model_flat")
    
                self.classifier = pipeline(
                    "text-classification", 
                    model=model_path, 
                    return_all_scores=True,
                    device=self.device,
                    model_kwargs={"local_files_only": True}
                )
            self.initialized = True
        except Exception as e:
            print(f"CRITICAL ERROR initializing NLPEmotionInference: {e}")
            self.initialized = False
            self.error = str(e)

    def predict(self, text):
        if not getattr(self, 'initialized', False):
            print(f"NLP Inference not initialized: {getattr(self, 'error', 'Unknown error')}")
            return None

        if not text or not text.strip():
            return None

        if self.model:
            # Custom model inference
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            with self.torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probs = self.torch.nn.functional.softmax(outputs, dim=1)[0]
                
            return {self.idx_to_label[i]: float(probs[i]) for i in range(len(probs))}
        else:
            # Pipeline inference
            results = self.classifier(text)
            scores = results[0]
            return {item['label']: item['score'] for item in scores}

if __name__ == "__main__":
    nlp = NLPEmotionInference()
    print(nlp.predict("I am feeling so happy today!"))
    print(nlp.predict("I am really mad at you."))
