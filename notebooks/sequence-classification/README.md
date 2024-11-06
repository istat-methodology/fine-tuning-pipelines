<h1 align="center">
  Fine Tuning for Sequence Classification
</h1>

<div align="center">

  <a href="">![Static Badge](https://img.shields.io/badge/transformers-yellow)</a>
  <a href="">![Static Badge](https://img.shields.io/badge/AutoModelForSequenceClassification-blue)</a>
  
</div>

This folder contains code to fine-tune transformer models for sequence classification tasks, using the `transformers` library by Hugging Face. Sequence classification tasks include sentiment analysis, spam detection, topic classification, and more, where each sequence (e.g., a sentence or paragraph) is assigned a single label.

The code in this folders relies on the `AutoModelForSequenceClassification` class from `transformers`, which loads pre-trained transformer models and prepares them for sequence-level classification. This setup ensures that the model's final output dimension corresponds to the number of labels in the classification task, rather than the vocabulary size of the tokenizer.

---

Here is an example that loads a Llama 3.2 (1B) model for sequence classification with 3 labels.

```{python}
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    num_labels = 3
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B"
)
```

If we print the model's layers, we notice that the output dimension is equal to `num_labels`.