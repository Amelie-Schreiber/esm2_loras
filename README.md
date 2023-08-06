# esm2_loras

This is an attempt at training a Low Rank Adaptation (LoRA) for the protein language model ESM-2 for a token classification task. In particular, we attempt to train an RNA binding site predictor. There are still some issues to work out and any feedback or advice would be much appreciated. This code is for a small model so it should perform wandb sweeps for hyperparameter search in a reasonable amount of time on almost any GPU. You can easily swap out for [larger models](https://huggingface.co/facebook/esm2_t6_8M_UR50D) though if you want. 

## Model Weights and Config

The model itself 
```python
"AmelieSchreiber/esm2_t6_8M_UR50D_lora_rna_binding_sites"
```
can be found on [Hugging Face here](https://huggingface.co/AmelieSchreiber/esm2_t6_8M_UR50D_lora_rna_binding_sites).

## Setting up this repo

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

To use, try running:
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
from peft import PeftModel
import torch
import numpy as np
import random



# Path to the saved LoRA model
model_path = "esm2_t6_8M-finetuned-lora_2023-08-03_18-32-25"
# ESM2 base model
base_model_path = "facebook/esm2_t6_8M_UR50D"

# Load the model
base_model = AutoModelForTokenClassification.from_pretrained(base_model_path)
loaded_model = PeftModel.from_pretrained(base_model, model_path)

# Load the tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)

# New unseen protein sequence
new_protein_sequence = "FDLNDFLEQKVLVRMEAIINSMTMKERAKPEIIKGSRKRRIAAGSGMQVQDVNRLLKQFDDMQRMMKKM"

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
