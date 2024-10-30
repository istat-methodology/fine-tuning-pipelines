import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

def load_XLMRoBERTa(model_id: str, num_labels: int, hf_token: str, device: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer