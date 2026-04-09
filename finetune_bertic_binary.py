"""
Fine-tune classla/bcms-bertic (ELECTRA) for binary hate speech detection.

Input:  [CLS] full_paragraph [SEP] target_sentence [SEP]
Output: binary — hate (1) or no hate (0)

Data sources:
  - paragraph_hate_speech_no_offenses.xlsx      (paragraphs -> context)
  - single_sentence_hate_speech_no_offenses.xlsx (sentences  -> targets + labels)

Evaluation modes:
  - Strict:    any non-0 category -> hate
  - Best-case: ambiguous sentences (both 0 and non-0 categories) are always correct
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
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


# ── Known category codes ─────────────────────────────────────────────────────

VALID_CODES = {
    "0", "1a", "1b", "1c", "2", "3a", "3b",
    "4a", "4b", "5", "6a", "6b", "6c", "7",
}


# ── Parse category string -> (binary_label, is_ambiguous) ────────────────────

def parse_binary_label(category_str: str):
    """
    Return (binary_label, is_ambiguous).

    binary_label : 1 if any hate category present, else 0
    is_ambiguous : True when BOTH '0' and a hate code appear
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

    has_hate = any(c != "0" and c in VALID_CODES for c in codes)
    has_no_hate = "0" in codes
    return (1 if has_hate else 0), (has_hate and has_no_hate)


# ── Dataset ──────────────────────────────────────────────────────────────────

class BinaryHateSpeechDataset(Dataset):
    """Each sample = (paragraph_context, target_sentence) -> binary label."""

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
            self.paragraphs[idx],       # text_a = context
            self.sentences[idx],        # text_b = target sentence
            truncation="only_first",    # keep target sentence intact
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    labels = labels.astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
    }


# ── Strict + Best-case evaluation ───────────────────────────────────────────

def evaluate_binary(preds, labels, ambiguous_mask, split_name="TEST"):
    labels_int = labels.astype(int)
    preds_int = preds.astype(int)

    # ── Strict ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"[{split_name}] STRICT evaluation (any hate category -> hate)")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {accuracy_score(labels_int, preds_int):.4f}")
    print(f"  F1:        {f1_score(labels_int, preds_int, average='binary', zero_division=0):.4f}")
    print(f"  Precision: {precision_score(labels_int, preds_int, average='binary', zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(labels_int, preds_int, average='binary', zero_division=0):.4f}")
    print(classification_report(
        labels_int, preds_int, target_names=["No hate", "Hate"], zero_division=0,
    ))

    # ── Best-case ────────────────────────────────────────────────────────
    n_amb = int(ambiguous_mask.sum())
    print(f"  Ambiguous sentences (both 0 and hate labels): {n_amb} / {len(labels_int)}")

    if n_amb > 0:
        # For ambiguous sentences, set ground truth = prediction (always correct)
        best_labels = labels_int.copy()
        best_labels[ambiguous_mask] = preds_int[ambiguous_mask]

        print(f"\n{'=' * 60}")
        print(f"[{split_name}] BEST-CASE evaluation (ambiguous -> always correct)")
        print(f"{'=' * 60}")
        print(f"  Accuracy:  {accuracy_score(best_labels, preds_int):.4f}")
        print(f"  F1:        {f1_score(best_labels, preds_int, average='binary', zero_division=0):.4f}")
        print(f"  Precision: {precision_score(best_labels, preds_int, average='binary', zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(best_labels, preds_int, average='binary', zero_division=0):.4f}")
        print(classification_report(
            best_labels, preds_int, target_names=["No hate", "Hate"], zero_division=0,
        ))
    else:
        print("  No ambiguous sentences — best-case = strict.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERTić for binary hate speech detection (paragraph context + sentence)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="bertic_finetuned_binary")
    parser.add_argument("--output", "-o", default="results/bertic/bertic_binary_results.xlsx",
                        help="Output Excel path for results")
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
        help="Number of updates steps to accumulate before performing a backward/update pass"
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

    # ── Load datasets ────────────────────────────────────────────────────
    print("Loading datasets...")
    df_sent = pd.read_excel(args.sentence_path)
    df_para = pd.read_excel(args.paragraph_path)

    # Paragraph lookup: ID -> full paragraph text
    para_lookup = dict(zip(df_para["ID"], df_para["Text"]))

    # Parse binary labels + ambiguity flag per sentence
    binary_labels = []
    ambiguous_flags = []
    for cat in df_sent["Category"]:
        bl, amb = parse_binary_label(cat)
        binary_labels.append(bl)
        ambiguous_flags.append(amb)

    df_sent["binary_label"] = binary_labels
    df_sent["is_ambiguous"] = ambiguous_flags
    df_sent["paragraph_text"] = df_sent["ID"].map(para_lookup)

    # ── Split by paragraph ID ────────────────────────────────────────────
    df_test = df_sent[df_sent["ID"] < 102].reset_index(drop=True)
    df_train_pool = df_sent[df_sent["ID"] >= 102].reset_index(drop=True)

    print(f"Train pool (ID >= 102): {len(df_train_pool)} sentences")
    print(f"Test set   (ID 1-101):  {len(df_test)} sentences")

    print(f"\nTrain distribution:")
    print(f"  Hate:      {df_train_pool['binary_label'].sum()} / {len(df_train_pool)}")
    print(f"  No hate:   {(df_train_pool['binary_label'] == 0).sum()} / {len(df_train_pool)}")
    print(f"  Ambiguous: {df_train_pool['is_ambiguous'].sum()}")

    print(f"\nTest distribution:")
    print(f"  Hate:      {df_test['binary_label'].sum()} / {len(df_test)}")
    print(f"  No hate:   {(df_test['binary_label'] == 0).sum()} / {len(df_test)}")
    print(f"  Ambiguous: {df_test['is_ambiguous'].sum()}")

    # ── Prepare arrays ───────────────────────────────────────────────────
    train_para_all = df_train_pool["paragraph_text"].tolist()
    train_sent_all = df_train_pool["Text"].tolist()
    train_labels_all = df_train_pool["binary_label"].values

    test_para = df_test["paragraph_text"].tolist()
    test_sent = df_test["Text"].tolist()
    test_labels = df_test["binary_label"].values
    test_ambiguous = df_test["is_ambiguous"].values

    # ── Train / Validation split ─────────────────────────────────────────
    indices = np.arange(len(train_labels_all))
    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_split,
        random_state=args.seed, stratify=train_labels_all,
    )

    train_para = [train_para_all[i] for i in train_idx]
    train_sent = [train_sent_all[i] for i in train_idx]
    train_labels = train_labels_all[train_idx]

    val_para = [train_para_all[i] for i in val_idx]
    val_sent = [train_sent_all[i] for i in val_idx]
    val_labels = train_labels_all[val_idx]

    print(f"\nTrain: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")

    # ── Class weights ────────────────────────────────────────────────────
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_labels)
    class_weights = torch.tensor(cw, dtype=torch.float)
    print(f"Class weights: No hate={cw[0]:.4f}, Hate={cw[1]:.4f}")

    # ── Tokenizer and model ──────────────────────────────────────────────
    model_name = "classla/bcms-bertic"
    print(f"\nLoading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
        classifier_dropout=args.dropout,
        hidden_dropout_prob=args.dropout,
    )
    print(f"Classifier dropout: {args.dropout}")

    # ── Layer freezing ───────────────────────────────────────────────────
    freeze = args.freeze.strip().lower()
    if freeze != "none":
        if freeze == "backbone":
            # Freeze everything except classifier head
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

    # ── Datasets ─────────────────────────────────────────────────────────
    train_dataset = BinaryHateSpeechDataset(
        train_para, train_sent, train_labels, tokenizer, args.max_length,
    )
    val_dataset = BinaryHateSpeechDataset(
        val_para, val_sent, val_labels, tokenizer, args.max_length,
    )
    test_dataset = BinaryHateSpeechDataset(
        test_para, test_sent, test_labels, tokenizer, args.max_length,
    )

    # ── Training arguments ───────────────────────────────────────────────
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

    # ── Trainer ──────────────────────────────────────────────────────────
    print("Using weighted cross-entropy loss")

    class WeightedTrainer(Trainer):
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

    # ── Train ────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.train()

    # ── Validation results ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Validation set results (used for early stopping):")
    print("=" * 60)
    metrics = trainer.evaluate()
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # ── Test set results ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST SET results (IDs 1-101):")
    print("=" * 60)
    test_output = trainer.predict(test_dataset)
    for key, value in sorted(test_output.metrics.items()):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # ── Strict + Best-case evaluation ────────────────────────────────
    test_preds = np.argmax(test_output.predictions, axis=1)
    evaluate_binary(test_preds, test_labels, test_ambiguous, split_name="TEST")

    # ── Save results to xlsx ──────────────────────────────────────────
    label_map = {0: "No hate", 1: "Hate"}
    best_labels = test_labels.copy()
    best_labels[test_ambiguous] = test_preds[test_ambiguous]

    df_results = pd.DataFrame({
        "ID": df_test["ID"].values,
        "Text": df_test["Text"].values,
        "Category": df_test["Category"].values,
        "GT_label": [label_map[l] for l in test_labels],
        "Predicted_label": [label_map[p] for p in test_preds],
        "Is_ambiguous": test_ambiguous,
        "Strict_correct": (test_preds == test_labels),
        "BestCase_correct": (test_preds == best_labels),
    })

    # Metrics summary
    strict_metrics = {
        "Accuracy": accuracy_score(test_labels, test_preds),
        "F1": f1_score(test_labels, test_preds, average="binary", zero_division=0),
        "Precision": precision_score(test_labels, test_preds, average="binary", zero_division=0),
        "Recall": recall_score(test_labels, test_preds, average="binary", zero_division=0),
    }
    best_metrics = {
        "Accuracy": accuracy_score(best_labels, test_preds),
        "F1": f1_score(best_labels, test_preds, average="binary", zero_division=0),
        "Precision": precision_score(best_labels, test_preds, average="binary", zero_division=0),
        "Recall": recall_score(best_labels, test_preds, average="binary", zero_division=0),
    }
    df_metrics = pd.DataFrame({
        "Metric": list(strict_metrics.keys()),
        "Strict": list(strict_metrics.values()),
        "Best-case": list(best_metrics.values()),
    })

    xlsx_path = args.output
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="Predictions", index=False)
        df_metrics.to_excel(writer, sheet_name="Metrics", index=False)
    print(f"\nResults saved to: {xlsx_path}")

    # ── Save model ───────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to: {final_dir}")


if __name__ == "__main__":
    main()
