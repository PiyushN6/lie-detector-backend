import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextModel:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def predict_deception(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        return probs[1].item()
