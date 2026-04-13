"""
Fine-tune classla/bcms-bertic (ELECTRA) for 8-way category hate speech detection.

Input:  [CLS] full_paragraph [SEP] target_sentence [SEP]
Output: one category label in {0..7}

Data sources:
  - paragraph_hate_speech_no_offenses.xlsx      (paragraphs -> context)
  - single_sentence_hate_speech_no_offenses.xlsx (sentences  -> targets + labels)

Category mapping:
  0  - No hate speech
  1  - Racial/ethnic (1a, 1b, 1c)
  2  - Religious (2)
  3  - Sex/gender (3a, 3b)
  4  - Physical/health (4a, 4b)
  5  - Age (5)
  6  - Socioeconomic/political (6a, 6b, 6c)
  7  - Sports/fan (7)
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


VALID_CODES = {
    "0", "1a", "1b", "1c", "2", "3a", "3b",
    "4a", "4b", "5", "6a", "6b", "6c", "7",
}

SUBCAT_TO_CATEGORY = {
    "0": 0,
    "1a": 1, "1b": 1, "1c": 1,
    "2": 2,
    "3a": 3, "3b": 3,
    "4a": 4, "4b": 4,
    "5": 5,
    "6a": 6, "6b": 6, "6c": 6,
    "7": 7,
}

LABEL_NAMES = [
    "0 (No hate speech)",
    "1 (Racial/ethnic)",
    "2 (Religious)",
    "3 (Sex/gender)",
    "4 (Physical/health)",
    "5 (Age)",
    "6 (Socioeconomic/political)",
    "7 (Sports/fan)",
]


def parse_category_label(category_str: str):
    """
    Return (category_label, is_ambiguous).

    Ambiguous means more than one distinct category appears in annotation.
    Strategy for multi-category rows:
      - if both 0 and one hate category appear, choose the hate category
      - if multiple hate categories appear, choose the smallest id deterministically
    """
    codes = set()
    parts = str(category_str).split(",")
    for part in parts:
        part = part.strip()
        if part.startswith("(") and part.endswith(")"):
            for c in part[1:-1].split(";"):
                codes.add(c.strip().lower())
        else:
            for sub in part.split(":"):
                codes.add(sub.strip().lower())

    categories = sorted({SUBCAT_TO_CATEGORY[c] for c in codes if c in VALID_CODES})
    if not categories:
        return 0, False

    if len(categories) == 1:
        return categories[0], False

    non_zero = [c for c in categories if c != 0]
    if len(non_zero) == 1 and 0 in categories:
        return non_zero[0], True

    return (non_zero[0] if non_zero else 0), True


class CategoryDataset(Dataset):
    """Each sample = (paragraph_context, target_sentence) -> category label 0..7."""

    def __init__(self, paragraphs, sentences, labels, tokenizer, max_length=512):
        self.paragraphs = paragraphs
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.paragraphs[idx],
            self.sentences[idx],
            truncation="only_first",
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    labels = labels.astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERTic for 8-way category classification (paragraph + sentence)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="bertic_finetuned_categories")
    parser.add_argument(
        "--output", "-o", default="results/bertic/bertic_categories_results.xlsx",
        help="Output Excel path for results",
    )
    parser.add_argument(
        "--freeze", type=str, default="none",
        help="Freeze strategy: none, backbone, embeddings, embeddings+N, N (e.g. 3, 6)",
    )
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Classifier head dropout (default 0.1, try 0.3-0.5)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW weight decay (default 0.01, try 0.01-0.1)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor (e.g. 0.05-0.1)")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--sentence_path", type=str,
        default="data/single_sentence_hate_speech_no_offenses.xlsx",
    )
    parser.add_argument(
        "--paragraph_path", type=str,
        default="data/paragraph_hate_speech_no_offenses.xlsx",
    )
    args = parser.parse_args()

    # Load datasets
    print("Loading datasets...")
    df_sent = pd.read_excel(args.sentence_path)
    df_para = pd.read_excel(args.paragraph_path)

    para_lookup = dict(zip(df_para["ID"], df_para["Text"]))

    category_labels = []
    ambiguous_flags = []
    for cat in df_sent["Category"]:
        label, amb = parse_category_label(cat)
        category_labels.append(label)
        ambiguous_flags.append(amb)

    df_sent["category_label"] = category_labels
    df_sent["is_ambiguous"] = ambiguous_flags
    df_sent["paragraph_text"] = df_sent["ID"].map(para_lookup)

    missing_context = int(df_sent["paragraph_text"].isna().sum())
    if missing_context > 0:
        raise ValueError(f"Missing paragraph context for {missing_context} sentence rows.")

    # Split by paragraph ID
    df_test = df_sent[df_sent["ID"] < 102].reset_index(drop=True)
    df_train_pool = df_sent[df_sent["ID"] >= 102].reset_index(drop=True)

    print(f"Train pool (ID >= 102): {len(df_train_pool)} sentences")
    print(f"Test set   (ID 1-101):  {len(df_test)} sentences")
    print(f"Ambiguous labels in train pool: {int(df_train_pool['is_ambiguous'].sum())}")
    print(f"Ambiguous labels in test set:   {int(df_test['is_ambiguous'].sum())}")

    # Prepare arrays
    train_para_all = df_train_pool["paragraph_text"].tolist()
    train_sent_all = df_train_pool["Text"].tolist()
    train_labels_all = df_train_pool["category_label"].values

    test_para = df_test["paragraph_text"].tolist()
    test_sent = df_test["Text"].tolist()
    test_labels = df_test["category_label"].values

    # Train / Validation split
    indices = np.arange(len(train_labels_all))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=train_labels_all,
    )

    train_para = [train_para_all[i] for i in train_idx]
    train_sent = [train_sent_all[i] for i in train_idx]
    train_labels = train_labels_all[train_idx]

    val_para = [train_para_all[i] for i in val_idx]
    val_sent = [train_sent_all[i] for i in val_idx]
    val_labels = train_labels_all[val_idx]

    print(f"\nTrain: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")

    # Class weights
    classes = np.arange(len(LABEL_NAMES))
    cw = compute_class_weight("balanced", classes=classes, y=train_labels)
    class_weights = torch.tensor(cw, dtype=torch.float)
    print("Class weights:")
    for i, w in enumerate(cw):
        print(f"  {LABEL_NAMES[i]}: {w:.4f}")

    # Tokenizer and model
    model_name = "classla/bcms-bertic"
    print(f"\nLoading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_NAMES),
        classifier_dropout=args.dropout,
        hidden_dropout_prob=args.dropout,
    )
    print(f"Classifier dropout: {args.dropout}")

    # Layer freezing
    freeze = args.freeze.strip().lower()
    if freeze != "none":
        if freeze == "backbone":
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        elif freeze == "embeddings":
            for name, param in model.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = False
        elif freeze.startswith("embeddings+"):
            n_layers = int(freeze.split("+")[1])
            for name, param in model.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = False
                for i in range(n_layers):
                    if f"encoder.layer.{i}." in name:
                        param.requires_grad = False
        elif freeze.isdigit():
            n_layers = int(freeze)
            for name, param in model.named_parameters():
                for i in range(n_layers):
                    if f"encoder.layer.{i}." in name:
                        param.requires_grad = False
        else:
            raise ValueError(f"Unknown freeze strategy: {freeze}")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"\nFreeze strategy: {freeze}")
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Datasets
    train_dataset = CategoryDataset(train_para, train_sent, train_labels, tokenizer, args.max_length)
    val_dataset = CategoryDataset(val_para, val_sent, val_labels, tokenizer, args.max_length)
    test_dataset = CategoryDataset(test_para, test_sent, test_labels, tokenizer, args.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        warmup_ratio=0.15,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
    )

    # Trainer
    print("Using weighted cross-entropy loss")

    class WeightedTrainer(Trainer):
        def _save_optimizer_and_scheduler(self, output_dir):
            pass  # Skip optimizer state to avoid disk-space errors on checkpoints

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(labels.device),
                label_smoothing=args.label_smoothing,
            )(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Validation results
    print("\n" + "=" * 60)
    print("Validation set results (used for early stopping):")
    print("=" * 60)
    metrics = trainer.evaluate()
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Test set results
    print("\n" + "=" * 60)
    print("TEST SET results (IDs 1-101):")
    print("=" * 60)
    test_output = trainer.predict(test_dataset)
    for key, value in sorted(test_output.metrics.items()):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Save results to xlsx
    test_preds = np.argmax(test_output.predictions, axis=1)
    df_results = pd.DataFrame({
        "ID": df_test["ID"].values,
        "Text": df_test["Text"].values,
        "Category": df_test["Category"].values,
        "GT_label": [LABEL_NAMES[l] for l in test_labels],
        "Predicted_label": [LABEL_NAMES[p] for p in test_preds],
        "Is_ambiguous": df_test["is_ambiguous"].values,
        "Correct": (test_preds == test_labels),
    })

    metrics_summary = {
        "Accuracy": accuracy_score(test_labels, test_preds),
        "F1": f1_score(test_labels, test_preds, average="macro", zero_division=0),
    }
    df_metrics = pd.DataFrame({
        "Metric": list(metrics_summary.keys()),
        "Value": list(metrics_summary.values()),
    })

    xlsx_path = args.output
    output_folder = os.path.dirname(xlsx_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="Predictions", index=False)
        df_metrics.to_excel(writer, sheet_name="Metrics", index=False)
    print(f"\nResults saved to: {xlsx_path}")

    # Save model
    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to: {final_dir}")


if __name__ == "__main__":
    main()
