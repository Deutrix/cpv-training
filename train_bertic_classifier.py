import json
import glob
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import joblib

DATA_DIR = Path("data/postupci")
MODEL_OUT_DIR = Path("models/bertic_cpv_classifier")

BASE_MODEL_NAME = "classla/bcms-bertic"  # BERTić


def load_procurements():
    texts = []
    labels = []

    json_files = glob.glob(str(DATA_DIR / "*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {DATA_DIR}")

    for fp in json_files:
        print(f"Loading {fp}")
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        for row in data:
            name = (row.get("Name") or "").strip()
            name_eng = (row.get("NameENG") or "").strip()
            type_contract = (row.get("TypeContract") or "").strip()
            procedure_type = (row.get("ProcedureType") or "").strip()

            text_parts = [name, name_eng, type_contract, procedure_type]
            text = " | ".join([t for t in text_parts if t])

            cpv_extended = row.get("CPVExtended") or ""
            if " - " in cpv_extended:
                cpv_code = cpv_extended.split(" - ", 1)[0].strip()
            else:
                continue

            if not text or not cpv_code:
                continue

            texts.append(text)
            labels.append(cpv_code)

    return texts, labels


def build_label_map(labels):
    unique_labels = sorted(set(labels))
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


class CPVDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label_str = self.labels[idx]
        label_id = self.label2id[label_str]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(label_id, dtype=torch.long)
        return item


def main():
    print("Loading data...")
    texts, labels = load_procurements()
    print(f"Loaded {len(texts)} samples, {len(set(labels))} unique CPV codes")

    # (opciono) filtriranje za evaluaciju: izbaciti klase s < 2 primera
    counts = Counter(labels)
    keep_mask = [counts[l] >= 2 for l in labels]
    texts_eval = [t for t, keep in zip(texts, keep_mask) if keep]
    labels_eval = [l for l, keep in zip(labels, keep_mask) if keep]

    # train/test split na filtriranim podacima za VALIDACIJU (ne za final train)
    X_train, X_val, y_train, y_val = train_test_split(
        texts_eval,
        labels_eval,
        test_size=0.1,
        random_state=42,
        stratify=labels_eval,
    )

    # ali za finalni model ćemo koristiti SVE podatke (za sada treniramo samo na train setu;
    # kasnije možeš lako da prebaciš na full-data-train ako si zadovoljan)
    label2id, id2label = build_label_map(labels)

    num_labels = len(label2id)
    print(f"num_labels = {num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    train_dataset = CPVDataset(X_train, y_train, tokenizer, label2id, max_length=64)
    val_dataset = CPVDataset(X_val, y_val, tokenizer, label2id, max_length=64)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # CPU only (use_cpu=True); ako imaš GPU, slobodno ukloni use_cpu parametar
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUT_DIR / "checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",  # Must match eval_strategy when using load_best_model_at_end
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        use_cpu=True,  # Changed from no_cuda (deprecated)
    )

    def compute_metrics(eval_pred):
        logits, labels_ids = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels_ids).mean()
        return {"accuracy": float(acc)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training BERTić classifier...")
    trainer.train()

    # Sačuvaj finalni model + tokenizer
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUT_DIR))

    # Sačuvaj i mapu labela (za API)
    joblib.dump(
        {"label2id": label2id, "id2label": id2label},
        MODEL_OUT_DIR / "label_map.joblib",
    )

    print(f"Saved fine-tuned model to {MODEL_OUT_DIR}")


if __name__ == "__main__":
    main()
