import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from tqdm import tqdm
from .dataset import TextEmotionDataset
from .model import BERTForEmotion
import os

def train_nlp_model(csv_file, model_save_path, epochs=3, batch_size=16, learning_rate=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    dataset = TextEmotionDataset(csv_file, tokenizer=tokenizer)
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BERTForEmotion(num_classes=len(dataset.label_map)).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

    # Save model
    torch.save(model.state_dict(), model_save_path)
    # Also save the label map
    torch.save(dataset.label_map, model_save_path + ".labels")
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    # Example usage
    DATA_FILE = "data/text_emotion/data.csv"
    MODEL_PATH = "models/text_emotion_model.pth"
    os.makedirs("models", exist_ok=True)
    if os.path.exists(DATA_FILE):
        train_nlp_model(DATA_FILE, MODEL_PATH, epochs=1)
    else:
        print(f"Data file {DATA_FILE} not found.")
