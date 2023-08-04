import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    EsmForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import wandb
from peft import LoraConfig, get_peft_model, TaskType
import accelerate
from accelerate import Accelerator
from huggingface_hub import notebook_login

def train_protein_model():
    # Define the hyperparameter search space for wandb
    def wandb_hp_space(trial):
        return {
            "method": "random",
            "metric": {"name": "accuracy", "goal": "maximize"},
            "parameters": {
                "learning_rate": {"distribution": "uniform", "min": 1e-5, "max": 1e-3},
                "per_device_train_batch_size": {"values": [2, 4, 8, 16]},
            },
        }

    # Initialize accelerator
    accelerator = Accelerator()

    # Load XML file
    tree = ET.parse('binding_sites.xml')
    root = tree.getroot()

    # Define lists to hold sequences and binding site labels
    sequences = []
    binding_sites = []

    # Iterate through the XML and extract the sequences and binding site labels
    for partner in root.findall('partner'):
        for bind_partner in partner.findall('BindPartner'):
            sequence = bind_partner.find('proSeq').text
            pro_bnd = bind_partner.find('proBnd').text
            sites = [1 if char == '+' else 0 for char in pro_bnd]
            sequences.append(sequence)
            binding_sites.append(sites)

    unique_labels = set(label for site in binding_sites for label in site)
    print("Unique labels in binding sites:", unique_labels)

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    class ProteinDataset(Dataset):
        def __init__(self, sequences, binding_sites, tokenizer, max_length=512):
            self.sequences = sequences
            self.binding_sites = binding_sites
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            sequence = self.sequences[idx][:self.max_length]
            binding_site = self.binding_sites[idx][:self.max_length]
            encoding = self.tokenizer(sequence, truncation=True, padding='max_length', max_length=self.max_length)
            encoding['labels'] = binding_site + [-100] * (self.max_length - len(binding_site))
            return encoding

    dataset = ProteinDataset(sequences, binding_sites, tokenizer)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset, val_dataset = accelerator.prepare(train_dataset, val_dataset)

    def model_init(trial):
        base_model = EsmForTokenClassification.from_pretrained(
            "facebook/esm2_t6_8M_UR50D",
            num_labels=2 # For binary classification
        )
        config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=16,
            lora_alpha=16,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="all",
        )
        lora_model = get_peft_model(base_model, config)
        return accelerator.prepare(lora_model)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        labels = accelerator.gather(labels)
        mask = labels != -100
        accuracy = (predictions[mask] == labels[mask]).mean()
        return {'accuracy': accuracy}

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"esm2_t6_8M-finetuned-lora_{timestamp_str}"

    args = TrainingArguments(
        output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-3,
        per_device_train_batch_size=2,
        num_train_epochs=5,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_strategy="epoch",
        label_names=["labels"],
    )

    trainer = Trainer(
        model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        model_init=model_init,
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="wandb",
        hp_space=wandb_hp_space,
        n_trials=10,
    )

    print("Best Trial:", best_trial)

    def train_final_model(best_trial):
        best_hyperparameters = best_trial.hyperparameters
        model = model_init(None)
        args.learning_rate = best_hyperparameters["learning_rate"]
        args.per_device_train_batch_size = best_hyperparameters["per_device_train_batch_size"]
        final_trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        final_trainer.train()

        # Explicitly save the model's configuration
        model.config.save_pretrained(output_dir)

        # Save the model
        final_trainer.save_model(output_dir)
        
        # Log in to Hugging Face account
        # notebook_login()

        # Push the model to the Hub
        # repo_name = "AmelieSchreiber/esm2_t6_8M_UR50D_lora_rna_binding_sites" # Change this to your desired repository name
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        # model.push_to_hub(repo_name)

    train_final_model(best_trial)