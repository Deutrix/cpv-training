import os
import json
import argparse
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)

class CPVDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='Run a quick smoke test with a small dataset')
    args = parser.parse_args()

    # Paths
    data_file = 'training_data.csv'
    model_name = 'classla/bcms-bertic' # BERTic - specialized for BCMS languages (Bosnian, Croatian, Montenegrin, Serbian)
    output_dir = 'models/cpv-decoder'
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Could not find {data_file}. Please run prepare_training_data.py first.")

    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Rename columns to standard names if needed, based on prepare_training_data.py output ['input', 'output']
    # 'input' -> 'text', 'output' -> 'label'
    if 'input' in df.columns and 'output' in df.columns:
        df = df.rename(columns={'input': 'text', 'output': 'label'})
    
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(str)
    
    # Create Label Map
    unique_labels = sorted(df['label'].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    num_labels = len(unique_labels)
    
    print(f"Found {num_labels} unique labels.")
    
    # Save label map
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump({'label2id': label2id, 'id2label': id2label}, f, indent=2)
    
    # Split Data
    # Stratification disabled to avoid errors with classes having few samples
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=None)
    
    if args.smoke_test:
        print("Running in smoke test mode. Limiting dataset to 100 samples.")
        train_df = train_df.sample(min(100, len(train_df)))
        val_df = val_df.sample(min(20, len(val_df)))
        output_dir = 'models/cpv-decoder-test'
    
    print(f"Training on {len(train_df)} samples.")
    print(f"Validating on {len(val_df)} samples.")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(texts):
        return tokenizer(texts, padding=True, truncation=True, max_length=128)
    
    print("Tokenizing data...")
    train_encodings = tokenize_function(train_df['text'].tolist())
    val_encodings = tokenize_function(val_df['text'].tolist())
    
    train_labels = [label2id[l] for l in train_df['label']]
    val_labels = [label2id[l] for l in val_df['label']]
    
    train_dataset = CPVDataset(train_encodings, train_labels)
    val_dataset = CPVDataset(val_encodings, val_labels)
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=64, # Aggressive batch size for A100
        gradient_accumulation_steps=1,  # No need to accumulate with high VRAM
        per_device_eval_batch_size=64,
        dataloader_num_workers=16,      # utilize 94 vCPUs
        fp16=True,
        group_by_length=True,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Training complete.")

if __name__ == "__main__":
    main()
