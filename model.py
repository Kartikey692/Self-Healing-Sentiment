from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()
login(os.environ["HF_TOKEN"])

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = {0: "Negative", 1: "Positive"}
MODEL_PATH = "lora-distilbert-imdb"

def load_finetuned():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return model, tokenizer

def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
        conf, pred = torch.max(probs, dim=0)
    return LABELS[pred.item()], conf.item()
