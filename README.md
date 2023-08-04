# esm2_loras

This is an attempt at training a Low Rank Adaptation (LoRA) for the protein language model ESM-2 for a token classification task. In particular, we attempt to train an RNA binding site predictor. There are still some issues to work out and any feedback or advice would be much appreciated. 

To set up the the conda environment, clone the repo and run:
```
conda env create -f environment.yml
```
Then run:
```
conda activate lora_esm_2
```
To train the model run:
```python
from lora_esm2_script import train_protein_model

train_protein_model()
```

It seems as though it is necessary to run the following to take care of any randomness in the model:
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

# Path to the saved model
model_path = "esm2_t6_8M-finetuned-lora_2023-08-03_18-32-25"

# Load the model
loaded_model = AutoModelForTokenClassification.from_pretrained(model_path)
loaded_model.eval()  # Set the model to evaluation mode

# Load the tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)

# New unseen protein sequence
new_protein_sequence = "MADSEQNSEWKEVKEQKANRGW"

# Tokenize the new sequence
inputs = loaded_tokenizer(new_protein_sequence, truncation=True, padding='max_length', max_length=512, return_tensors="pt")

# Make predictions
with torch.no_grad():
    outputs = loaded_model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

# Print logits for debugging
print("Logits:", logits)

# Convert predictions to a list
predicted_labels = predictions.squeeze().tolist()

# Get input IDs to identify padding and special tokens
input_ids = inputs['input_ids'].squeeze().tolist()

# Define a set of token IDs that correspond to special tokens
special_tokens_ids = {loaded_tokenizer.cls_token_id, loaded_tokenizer.pad_token_id, loaded_tokenizer.eos_token_id}

# Filter the predicted labels using the special_tokens_ids to remove predictions for special tokens
binding_sites = [label for label, token_id in zip(predicted_labels, input_ids) if token_id not in special_tokens_ids]

print("Predicted binding sites:", binding_sites)
```
