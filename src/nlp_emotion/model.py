import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class BERTForEmotion(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_classes=6):
        super(BERTForEmotion, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits
