import pandas as pd
from torch.utils.data import Dataset
import torch

class TextEmotionDataset(Dataset):
    """
    Dataset for text emotion classification.
    Expected CSV format: text, label
    """
    def __init__(self, csv_file, tokenizer=None, max_len=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Map labels to integers if they are strings
        self.label_map = {label: i for i, label in enumerate(self.data['sentiment'].unique())}
        self.idx_to_label = {i: label for label, i in self.label_map.items()}

    def __len__(self):
        # For quick testing/verification, uncomment the next line
        # return min(len(self.data), 100) 
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['content'])
        label = self.data.iloc[idx]['sentiment']
        
        label_id = self.label_map[label]

        if self.tokenizer:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )
            
            return {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label_id, dtype=torch.long)
            }
        else:
            return text, label_id
