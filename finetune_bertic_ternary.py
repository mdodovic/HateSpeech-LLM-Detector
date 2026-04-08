"""
Fine-tune classla/bcms-bertic (ELECTRA) for ternary hate speech detection.

Input:  [CLS] full_paragraph [SEP] target_sentence [SEP]
Output: ternary — 0 (no offense) / 1 (offense U) / 2 (hate speech)

Data sources:
  - paragraph_hate_speech_offenses.xlsx      (paragraphs -> context)
  - single_sentence_hate_speech_offenses.xlsx (sentences  -> targets + labels)

Evaluation modes:
  - Strict:    highest-priority label wins (hate > offense > no offense)
  - Best-case: ambiguous sentences (multiple class labels) are correct
               if the prediction matches ANY of the valid classes
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


# ── Constants ────────────────────────────────────────────────────────────────

LABEL_NAMES = ["No offense", "Offense (U)", "Hate speech"]

HATE_CODES = {
    "1a", "1b", "1c", "2", "3a", "3b",
    "4a", "4b", "5", "6a", "6b", "6c", "7",
    "1",  # bare '1' treated as hate
}


# ── Parse category string -> (ternary_label, valid_classes set) ───────────────

def parse_ternary_label(category_str: str):
    """
    Return (strict_label, valid_classes).

    strict_label : 2 if any hate code, else 1 if U, else 0
    valid_classes: set of class indices that this sentence legitimately belongs to
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

    valid = set()
    if "0" in codes:
        valid.add(0)
    if "u" in codes:
        valid.add(1)
    if any(c in HATE_CODES for c in codes):
        valid.add(2)

    # Strict: highest priority wins  (hate > offense > no offense)
    if 2 in valid:
        strict = 2
    elif 1 in valid:
        strict = 1
    else:
        strict = 0

    return strict, valid


# ── Dataset ──────────────────────────────────────────────────────────────────

class TernaryHateSpeechDataset(Dataset):
    """Each sample = (paragraph_context, target_sentence) -> ternary label."""

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
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


# ── Strict + Best-case evaluation ───────────────────────────────────────────

def evaluate_ternary(preds, labels, valid_classes_list, split_name="TEST"):
    labels_int = labels.astype(int)
    preds_int = preds.astype(int)

    # ── Strict ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"[{split_name}] STRICT evaluation (highest-priority label)")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {accuracy_score(labels_int, preds_int):.4f}")
    print(f"  F1 macro:  {f1_score(labels_int, preds_int, average='macro', zero_division=0):.4f}")
    print(f"  F1 weighted: {f1_score(labels_int, preds_int, average='weighted', zero_division=0):.4f}")
    print(classification_report(
        labels_int, preds_int, target_names=LABEL_NAMES, zero_division=0,
    ))

    # ── Best-case ────────────────────────────────────────────────────────
    ambiguous_mask = np.array([len(vc) > 1 for vc in valid_classes_list])
    n_amb = int(ambiguous_mask.sum())
    print(f"  Ambiguous sentences (multiple valid classes): {n_amb} / {len(labels_int)}")

    if n_amb > 0:
        # For ambiguous sentences, if the prediction is in valid_classes -> correct
        best_labels = labels_int.copy()
        for i in range(len(best_labels)):
            if ambiguous_mask[i] and preds_int[i] in valid_classes_list[i]:
                best_labels[i] = preds_int[i]

        print(f"\n{'=' * 60}")
        print(f"[{split_name}] BEST-CASE evaluation (ambiguous -> correct if pred in valid)")
        print(f"{'=' * 60}")
        print(f"  Accuracy:  {accuracy_score(best_labels, preds_int):.4f}")
        print(f"  F1 macro:  {f1_score(best_labels, preds_int, average='macro', zero_division=0):.4f}")
        print(f"  F1 weighted: {f1_score(best_labels, preds_int, average='weighted', zero_division=0):.4f}")
        print(classification_report(
            best_labels, preds_int, target_names=LABEL_NAMES, zero_division=0,
        ))
    else:
        print("  No ambiguous sentences — best-case = strict.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERTić for ternary classification (no offense / offense / hate speech)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="bertic_finetuned_ternary")
    parser.add_argument("--output", "-o", default="results/bertic_ternary_results.xlsx",
                        help="Output Excel path for results")
    parser.add_argument(
        "--sentence_path", type=str,
        default="data/single_sentence_hate_speech_offenses.xlsx",
    )
    parser.add_argument(
        "--paragraph_path", type=str,
        default="data/paragraph_hate_speech_offenses.xlsx",
    )
    args = parser.parse_args()

    # ── Load datasets ────────────────────────────────────────────────────
    print("Loading datasets...")
    df_sent = pd.read_excel(args.sentence_path)
    df_para = pd.read_excel(args.paragraph_path)

    # Paragraph lookup: ID -> full paragraph text
    para_lookup = dict(zip(df_para["ID"], df_para["Text"]))

    # Parse ternary labels + valid classes per sentence
    strict_labels = []
    valid_classes_all = []
    for cat in df_sent["Category"]:
        sl, vc = parse_ternary_label(cat)
        strict_labels.append(sl)
        valid_classes_all.append(vc)

    df_sent["ternary_label"] = strict_labels
    df_sent["valid_classes"] = valid_classes_all
    df_sent["paragraph_text"] = df_sent["ID"].map(para_lookup)

    # ── Split by paragraph ID ────────────────────────────────────────────
    df_test = df_sent[df_sent["ID"] < 102].reset_index(drop=True)
    df_train_pool = df_sent[df_sent["ID"] >= 102].reset_index(drop=True)

    print(f"Train pool (ID >= 102): {len(df_train_pool)} sentences")
    print(f"Test set   (ID 1-101):  {len(df_test)} sentences")

    for name, subset in [("Train", df_train_pool), ("Test", df_test)]:
        print(f"\n{name} distribution:")
        for cls_idx, cls_name in enumerate(LABEL_NAMES):
            n = (subset["ternary_label"] == cls_idx).sum()
            print(f"  {cls_name}: {n}")
        n_amb = sum(len(vc) > 1 for vc in subset["valid_classes"])
        print(f"  Ambiguous: {n_amb}")

    # ── Prepare arrays ───────────────────────────────────────────────────
    train_para_all = df_train_pool["paragraph_text"].tolist()
    train_sent_all = df_train_pool["Text"].tolist()
    train_labels_all = df_train_pool["ternary_label"].values
    train_vc_all = df_train_pool["valid_classes"].tolist()

    test_para = df_test["paragraph_text"].tolist()
    test_sent = df_test["Text"].tolist()
    test_labels = df_test["ternary_label"].values
    test_vc = df_test["valid_classes"].tolist()

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
    cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=train_labels)
    class_weights = torch.tensor(cw, dtype=torch.float)
    print(f"Class weights: No offense={cw[0]:.4f}, Offense={cw[1]:.4f}, Hate={cw[2]:.4f}")

    # ── Tokenizer and model ──────────────────────────────────────────────
    model_name = "classla/bcms-bertic"
    print(f"\nLoading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3,
    )

    # ── Datasets ─────────────────────────────────────────────────────────
    train_dataset = TernaryHateSpeechDataset(
        train_para, train_sent, train_labels, tokenizer, args.max_length,
    )
    val_dataset = TernaryHateSpeechDataset(
        val_para, val_sent, val_labels, tokenizer, args.max_length,
    )
    test_dataset = TernaryHateSpeechDataset(
        test_para, test_sent, test_labels, tokenizer, args.max_length,
    )

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
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(labels.device))(outputs.logits, labels)
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

    # ── Strict + Best-case evaluation ────────────────────────────────────
    test_preds = np.argmax(test_output.predictions, axis=1)
    evaluate_ternary(test_preds, test_labels, test_vc, split_name="TEST")

    # ── Save results to xlsx ──────────────────────────────────────────
    label_map = {0: "No offense", 1: "Offense (U)", 2: "Hate speech"}
    ambiguous_mask = np.array([len(vc) > 1 for vc in test_vc])
    best_labels = test_labels.copy()
    for i in range(len(best_labels)):
        if ambiguous_mask[i] and test_preds[i] in test_vc[i]:
            best_labels[i] = test_preds[i]

    df_results = pd.DataFrame({
        "ID": df_test["ID"].values,
        "Text": df_test["Text"].values,
        "Category": df_test["Category"].values,
        "GT_label": [label_map[l] for l in test_labels],
        "Predicted_label": [label_map[p] for p in test_preds],
        "Valid_classes": [";".join(label_map[c] for c in sorted(vc)) for vc in test_vc],
        "Is_ambiguous": ambiguous_mask,
        "Strict_correct": (test_preds == test_labels),
        "BestCase_correct": (test_preds == best_labels),
    })

    # Metrics summary
    strict_metrics = {
        "Accuracy": accuracy_score(test_labels, test_preds),
        "F1_macro": f1_score(test_labels, test_preds, average="macro", zero_division=0),
        "F1_weighted": f1_score(test_labels, test_preds, average="weighted", zero_division=0),
        "Precision_macro": precision_score(test_labels, test_preds, average="macro", zero_division=0),
        "Recall_macro": recall_score(test_labels, test_preds, average="macro", zero_division=0),
    }
    best_metrics = {
        "Accuracy": accuracy_score(best_labels, test_preds),
        "F1_macro": f1_score(best_labels, test_preds, average="macro", zero_division=0),
        "F1_weighted": f1_score(best_labels, test_preds, average="weighted", zero_division=0),
        "Precision_macro": precision_score(best_labels, test_preds, average="macro", zero_division=0),
        "Recall_macro": recall_score(best_labels, test_preds, average="macro", zero_division=0),
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
