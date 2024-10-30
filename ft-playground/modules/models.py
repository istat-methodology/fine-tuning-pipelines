import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

def load_model(model_id: str, task_type: str, num_labels: int, hf_token: str, device: str):
    if task_type == 'sequence-classification':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            token=hf_token
        ).to(device)
    else:
        model = None
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer