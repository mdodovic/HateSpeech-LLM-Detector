"""
Fine-tune classla/bcms-bertic (ELECTRA) for multi-label hate speech classification.

Data: single_sentence_hate_speech_no_offenses.xlsx
      Each row = one sentence with its own category label(s).
Labels: 14 subcategory-level classes (multi-label per sentence):
    0  - No hate speech
    1a - Race / skin color
    1b - Ethnic affiliation
    1c - Nationality / origin
    2  - Religious hate
    3a - Sex (sexism)
    3b - LGBTQ+ identities
    4a - Physical appearance
    4b - Illness / disability
    5  - Age (ageism)
    6a - Socioeconomic status / class
    6b - Occupation / profession
    6c - Political intolerance
    7  - Sports / fan-based hate
"""

import re
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


# ── Category labels ──────────────────────────────────────────────────────────

NUM_LABELS = 14  # all subcategory-level classes
LABEL_NAMES = [
    "0 (No hate speech)",
    "1a (Race/skin color)",
    "1b (Ethnic affiliation)",
    "1c (Nationality/origin)",
    "2 (Religious hate)",
    "3a (Sex/sexism)",
    "3b (LGBTQ+)",
    "4a (Physical appearance)",
    "4b (Illness/disability)",
    "5 (Age/ageism)",
    "6a (Socioeconomic status)",
    "6b (Occupation/profession)",
    "6c (Political intolerance)",
    "7 (Sports/fan hate)",
]
CODE_TO_IDX = {
    "0": 0,
    "1a": 1, "1b": 2, "1c": 3,
    "2": 4,
    "3a": 5, "3b": 6,
    "4a": 7, "4b": 8,
    "5": 9,
    "6a": 10, "6b": 11, "6c": 12,
    "7": 13,
}


# ── Parse sentence-level labels into multi-label vector ──────────────────────

def parse_sentence_labels(category_str: str) -> np.ndarray:
    """
    Parse a sentence-level category string into a binary vector of shape (14,).

    Examples:
        "0"      -> [1,0,0,0, 0, 0,0, 0,0, 0, 0,0,0, 0]
        "6c,0"   -> [1,0,0,0, 0, 0,0, 0,0, 0, 0,0,1, 0]
        "7,1b"   -> [0,0,1,0, 0, 0,0, 0,0, 0, 0,0,0, 1]
    """
    labels = np.zeros(NUM_LABELS, dtype=np.float32)
    parts = str(category_str).split(",")
    for part in parts:
        part = part.strip()
        # Handle compound labels like (1b;1c)
        if part.startswith("(") and part.endswith(")"):
            inner = part[1:-1].split(";")
            codes = [c.strip() for c in inner]
        else:
            codes = [part]
        for code in codes:
            # Handle colon-separated variants like "0:6b"
            for subcode in code.split(":"):
                subcode = subcode.strip().lower()
                if subcode in CODE_TO_IDX:
                    labels[CODE_TO_IDX[subcode]] = 1.0
    return labels


# ── Dataset ──────────────────────────────────────────────────────────────────

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    labels = labels.astype(int)

    # Per-label accuracy averaged across all labels
    per_label_acc = (preds == labels).mean()

    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
        "subset_accuracy": accuracy_score(labels, preds),
        "accuracy": per_label_acc,
    }


# ── Subcategory → Category → Binary mapping ─────────────────────────────────

# Which subcategory indices (1-13) belong to each main category (1-7)
CATEGORY_GROUP = {
    1: [1, 2, 3],     # 1a, 1b, 1c
    2: [4],            # 2
    3: [5, 6],         # 3a, 3b
    4: [7, 8],         # 4a, 4b
    5: [9],            # 5
    6: [10, 11, 12],   # 6a, 6b, 6c
    7: [13],           # 7
}

CATEGORY_NAMES = [
    "1 (Racial/ethnic)",
    "2 (Religious)",
    "3 (Sex/gender)",
    "4 (Physical/health)",
    "5 (Age)",
    "6 (Socioeconomic)",
    "7 (Sports/fan)",
]


def subcats_to_categories(arr):
    """Collapse 14-dim subcategory vectors into 7-dim category vectors."""
    out = np.zeros((arr.shape[0], 7), dtype=int)
    for cat_idx, (cat_num, sub_indices) in enumerate(CATEGORY_GROUP.items()):
        out[:, cat_idx] = arr[:, sub_indices].max(axis=1)
    return out


def subcats_to_binary(arr):
    """Collapse 14-dim subcategory vectors into binary hate/no-hate (1-dim)."""
    # Hate = any subcategory index 1-13 is active
    return (arr[:, 1:].max(axis=1) > 0).astype(int)


def evaluate_three_levels(preds, labels, split_name="TEST"):
    """Print accuracy & F1 at binary, category, and subcategory levels."""

    labels_int = labels.astype(int)
    preds_int = preds.astype(int)

    # ── 1) Binary: hate vs no hate ───────────────────────────────────────
    true_bin = subcats_to_binary(labels_int)
    pred_bin = subcats_to_binary(preds_int)
    bin_acc = accuracy_score(true_bin, pred_bin)
    bin_f1 = f1_score(true_bin, pred_bin, average="binary", zero_division=0)
    bin_prec = precision_score(true_bin, pred_bin, average="binary", zero_division=0)
    bin_rec = recall_score(true_bin, pred_bin, average="binary", zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"[{split_name}] 1) BINARY — hate vs no hate")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {bin_acc:.4f}")
    print(f"  F1:        {bin_f1:.4f}")
    print(f"  Precision: {bin_prec:.4f}")
    print(f"  Recall:    {bin_rec:.4f}")
    print(classification_report(
        true_bin, pred_bin, target_names=["No hate", "Hate"], zero_division=0,
    ))

    # ── 2) Category-level (7 main categories) ────────────────────────────
    true_cat = subcats_to_categories(labels_int)
    pred_cat = subcats_to_categories(preds_int)
    cat_acc = (pred_cat == true_cat).mean()
    cat_f1_micro = f1_score(true_cat, pred_cat, average="micro", zero_division=0)
    cat_f1_macro = f1_score(true_cat, pred_cat, average="macro", zero_division=0)
    cat_subset = accuracy_score(true_cat, pred_cat)

    print(f"\n{'=' * 60}")
    print(f"[{split_name}] 2) CATEGORY-LEVEL — 7 main categories")
    print(f"{'=' * 60}")
    print(f"  Per-label accuracy: {cat_acc:.4f}")
    print(f"  Subset accuracy:    {cat_subset:.4f}")
    print(f"  F1 micro:           {cat_f1_micro:.4f}")
    print(f"  F1 macro:           {cat_f1_macro:.4f}")
    print(classification_report(
        true_cat, pred_cat, target_names=CATEGORY_NAMES, zero_division=0,
    ))

    # ── 3) Subcategory-level (all 14 classes) ────────────────────────────
    sub_acc = (preds_int == labels_int).mean()
    sub_f1_micro = f1_score(labels_int, preds_int, average="micro", zero_division=0)
    sub_f1_macro = f1_score(labels_int, preds_int, average="macro", zero_division=0)
    sub_subset = accuracy_score(labels_int, preds_int)

    print(f"\n{'=' * 60}")
    print(f"[{split_name}] 3) SUBCATEGORY-LEVEL — all 14 classes")
    print(f"{'=' * 60}")
    print(f"  Per-label accuracy: {sub_acc:.4f}")
    print(f"  Subset accuracy:    {sub_subset:.4f}")
    print(f"  F1 micro:           {sub_f1_micro:.4f}")
    print(f"  F1 macro:           {sub_f1_macro:.4f}")
    print(classification_report(
        labels_int, preds_int, target_names=LABEL_NAMES, zero_division=0,
    ))


# ── Custom Trainer for multi-label BCE loss ──────────────────────────────────

class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERTić for hate speech detection")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="bertic_finetuned", help="Output directory")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/single_sentence_hate_speech_no_offenses.xlsx",
        help="Path to the annotated dataset",
    )
    args = parser.parse_args()

    # ── Load and prepare data ────────────────────────────────────────────
    print("Loading dataset...")
    df = pd.read_excel(args.data_path)

    # Test set: IDs 1-101, Train set: IDs >= 102  (sentence-level)
    df_test = df[df["ID"] < 102].reset_index(drop=True)
    df_train = df[df["ID"] >= 102].reset_index(drop=True)
    print(f"Train pool (ID >= 102): {len(df_train)} sentences")
    print(f"Test set   (ID 1-101):  {len(df_test)} sentences")

    train_texts_all = df_train["Text"].tolist()
    train_labels_all = np.array([parse_sentence_labels(cat) for cat in df_train["Category"]])

    test_texts = df_test["Text"].tolist()
    test_labels = np.array([parse_sentence_labels(cat) for cat in df_test["Category"]])

    # Show label distribution
    print("\nTrain label distribution:")
    has_hate_train = (train_labels_all[:, 1:].sum(axis=1) > 0).astype(int)
    print(f"  Sentences with hate speech: {has_hate_train.sum()} / {len(train_labels_all)}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name}: {int(train_labels_all[:, i].sum())}")

    print("\nTest label distribution:")
    has_hate_test = (test_labels[:, 1:].sum(axis=1) > 0).astype(int)
    print(f"  Sentences with hate speech: {has_hate_test.sum()} / {len(test_labels)}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name}: {int(test_labels[:, i].sum())}")

    # ── Train / Validation split (from train pool for early stopping) ────
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts_all, train_labels_all, test_size=args.val_split,
        random_state=args.seed, stratify=has_hate_train,
    )
    print(f"\nTrain: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # ── Tokenizer and model ──────────────────────────────────────────────
    model_name = "classla/bcms-bertic"
    print(f"\nLoading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    # ── Datasets ─────────────────────────────────────────────────────────
    train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer, args.max_length)
    test_dataset = HateSpeechDataset(test_texts, test_labels, tokenizer, args.max_length)

    # ── Training arguments ───────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.train()

    # ── Evaluate on validation set ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Validation set results (used for early stopping):")
    print("=" * 60)
    metrics = trainer.evaluate()
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # ── Evaluate on TEST set (IDs 1-101) ─────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST SET raw metrics (IDs 1-101):")
    print("=" * 60)
    test_output = trainer.predict(test_dataset)
    test_metrics = test_output.metrics
    for key, value in sorted(test_metrics.items()):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # ── Three-level evaluation on TEST set ───────────────────────────────
    probs = torch.sigmoid(torch.tensor(test_output.predictions)).numpy()
    preds = (probs >= 0.5).astype(int)
    evaluate_three_levels(preds, test_labels, split_name="TEST")

    # ── Save final model ─────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to: {final_dir}")


if __name__ == "__main__":
    main()
