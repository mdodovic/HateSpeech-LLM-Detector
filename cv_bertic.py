"""
Stratified k-fold cross-validation for BERTić hate speech detection.

Grouping unit: paragraph (sentences from the same paragraph stay together,
preventing data leakage between train and test folds).

Stratification label: sentence-level label, with groups ensuring all sentences
from a paragraph land in the same fold.

For each fold:
  1. Train BERTić on sentences from k-1 folds (fresh model from classla/bcms-bertic)
  2. Reserve 15 % of training sentences for validation / early stopping
  3. Evaluate on the held-out fold
  4. Collect per-fold and per-sample metrics

Aggregate statistics (mean ± std) are reported at the end and saved to Excel.

Supported tasks
  --task binary       2-class   (no hate / hate)
  --task category     8-class   (0..7)
  --task subcategory  14-class  (0, 1a, 1b, 1c, 2, 3a, 3b, 4a, 4b, 5, 6a, 6b, 6c, 7)

Usage
  python cv_bertic.py --task binary --k 3
  python cv_bertic.py --task category --k 5 --epochs 10 --lr 3e-5
"""

import argparse
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "classla/bcms-bertic"

VALID_CODES = {
    "0", "1a", "1b", "1c", "2", "3a", "3b",
    "4a", "4b", "5", "6a", "6b", "6c", "7",
}

SUBCAT_TO_CAT = {
    "0": 0,
    "1a": 1, "1b": 1, "1c": 1,
    "2": 2,
    "3a": 3, "3b": 3,
    "4a": 4, "4b": 4,
    "5": 5,
    "6a": 6, "6b": 6, "6c": 6,
    "7": 7,
}

SUBCAT_TO_IDX = {
    "0": 0, "1a": 1, "1b": 2, "1c": 3,
    "2": 4,
    "3a": 5, "3b": 6,
    "4a": 7, "4b": 8,
    "5": 9,
    "6a": 10, "6b": 11, "6c": 12,
    "7": 13,
}
IDX_TO_SUBCAT = {v: k for k, v in SUBCAT_TO_IDX.items()}

CATEGORY_NAMES = [
    "0 No hate",
    "1 Racial/ethnic",
    "2 Religious",
    "3 Sex/gender",
    "4 Physical/health",
    "5 Age",
    "6 Socioeconomic/political",
    "7 Sports/fan",
]

SUBCATEGORY_NAMES = [
    "0 No hate",
    "1a Race/skin",
    "1b Ethnic",
    "1c Nationality",
    "2 Religious",
    "3a Sex/sexism",
    "3b LGBTQ+",
    "4a Appearance",
    "4b Illness/disability",
    "5 Age",
    "6a Socioeconomic",
    "6b Political",
    "6c Regional",
    "7 Sports/fan",
]


# ─────────────────────────────────────────────────────────────────────────────
# Label parsing
# ─────────────────────────────────────────────────────────────────────────────

def _extract_codes(category_str: str):
    """Return list of valid codes from a category cell string."""
    codes = []
    for part in str(category_str).split(","):
        part = part.strip()
        if part.startswith("(") and part.endswith(")"):
            for c in part[1:-1].split(";"):
                codes.append(c.strip().lower())
        else:
            for sub in part.split(":"):
                codes.append(sub.strip().lower())
    return [c for c in codes if c in VALID_CODES]


def parse_binary(category_str: str):
    """Return (label: int, is_ambiguous: bool)."""
    codes = set(_extract_codes(category_str))
    has_hate = any(c != "0" for c in codes)
    return int(has_hate), (has_hate and "0" in codes)


def parse_category(category_str: str):
    """Return (primary category 0-7: int, is_ambiguous: bool)."""
    codes = _extract_codes(category_str)
    has_hate = any(c != "0" for c in codes)
    primary = next(
        (SUBCAT_TO_CAT[c] for c in codes if c != "0"),
        0,
    )
    return primary, (has_hate and "0" in set(codes))


def parse_subcategory(category_str: str):
    """Return (primary subcategory idx 0-13: int, is_ambiguous: bool)."""
    codes = _extract_codes(category_str)
    has_hate = any(c != "0" for c in codes)
    primary = next(
        (SUBCAT_TO_IDX[c] for c in codes if c != "0"),
        0,
    )
    return primary, (has_hate and "0" in set(codes))


PARSE_FN = {
    "binary": parse_binary,
    "category": parse_category,
    "subcategory": parse_subcategory,
}

N_LABELS = {"binary": 2, "category": 8, "subcategory": 14}

LABEL_NAMES_MAP = {
    "binary": ["No hate", "Hate"],
    "category": CATEGORY_NAMES,
    "subcategory": SUBCATEGORY_NAMES,
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HateSpeechDataset(Dataset):
    """[CLS] paragraph [SEP] sentence [SEP] → label"""

    def __init__(self, paragraphs, sentences, labels, tokenizer, max_length=512):
        self.paragraphs = paragraphs
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.paragraphs[idx],
            self.sentences[idx],
            truncation="only_first",
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Trainer with weighted cross-entropy
# ─────────────────────────────────────────────────────────────────────────────

def _make_weighted_trainer(class_weights: torch.Tensor):
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(outputs.logits.device)
            )(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss
    return WeightedTrainer


def _compute_metrics(task: str):
    avg = "binary" if task == "binary" else "macro"
    def fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy":  accuracy_score(labels, preds),
            "f1":        f1_score(labels, preds, average=avg, zero_division=0),
            "precision": precision_score(labels, preds, average=avg, zero_division=0),
            "recall":    recall_score(labels, preds, average=avg, zero_division=0),
        }
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Single fold
# ─────────────────────────────────────────────────────────────────────────────

def run_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    para_lookup: dict,
    task: str,
    args,
):
    """Train on train_df, evaluate on test_df, return (metrics_dict, per_sample_df)."""
    parse_fn = PARSE_FN[task]
    n_labels = N_LABELS[task]

    def _arrays(df):
        parsed = [parse_fn(cat) for cat in df["Category"]]
        labels = np.array([p[0] for p in parsed], dtype=int)
        ambiguous = np.array([p[1] for p in parsed], dtype=bool)
        para = df["ID"].map(para_lookup).tolist()
        sent = df["Text"].tolist()
        return labels, ambiguous, para, sent

    tr_labels, tr_amb, tr_para, tr_sent = _arrays(train_df)
    te_labels, te_amb, te_para, te_sent = _arrays(test_df)

    print(f"\n{'=' * 70}")
    print(f"  FOLD {fold_idx + 1}  |  task={task}")
    print(f"  Train: {len(tr_labels)} sentences  |  Test: {len(te_labels)} sentences")
    label_dist = Counter(te_labels)
    print(f"  Test label distribution: {dict(sorted(label_dist.items()))}")
    print(f"{'=' * 70}")

    # Inner train / val split (stratified, 15 %)
    n_unique = len(np.unique(tr_labels))
    tr_idx, val_idx = train_test_split(
        np.arange(len(tr_labels)),
        test_size=0.15,
        stratify=tr_labels if n_unique > 1 else None,
        random_state=args.seed,
    )
    tr_p2  = [tr_para[i] for i in tr_idx];  tr_s2  = [tr_sent[i] for i in tr_idx]
    val_p  = [tr_para[i] for i in val_idx]; val_s  = [tr_sent[i] for i in val_idx]
    tr_l2  = tr_labels[tr_idx];             val_l  = tr_labels[val_idx]

    # Class weights from the inner training set
    classes = np.unique(tr_l2)
    cw = compute_class_weight("balanced", classes=classes, y=tr_l2)
    cw_full = np.ones(n_labels, dtype=float)
    for cls, w in zip(classes, cw):
        cw_full[cls] = w
    class_weights = torch.tensor(cw_full, dtype=torch.float)

    # Fresh model for each fold
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_labels,
        classifier_dropout=args.dropout,
        hidden_dropout_prob=args.dropout,
    )

    fold_dir = f"{args.output_dir}/fold_{fold_idx + 1}"

    tr_ds   = HateSpeechDataset(tr_p2, tr_s2, tr_l2,    tokenizer, args.max_length)
    val_ds  = HateSpeechDataset(val_p, val_s, val_l,    tokenizer, args.max_length)
    test_ds = HateSpeechDataset(te_para, te_sent, te_labels, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=fold_dir,
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
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
    )

    WeightedTrainer = _make_weighted_trainer(class_weights)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tr_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics(task),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Predict on test fold
    pred_output = trainer.predict(test_ds)
    preds = np.argmax(pred_output.predictions, axis=1)

    avg = "binary" if task == "binary" else "macro"
    metrics = {
        "fold":      fold_idx + 1,
        "n_train":   int(len(tr_l2)),
        "n_val":     int(len(val_l)),
        "n_test":    int(len(te_labels)),
        "accuracy":  round(float(accuracy_score(te_labels, preds)), 4),
        "f1":        round(float(f1_score(te_labels, preds, average=avg, zero_division=0)), 4),
        "precision": round(float(precision_score(te_labels, preds, average=avg, zero_division=0)), 4),
        "recall":    round(float(recall_score(te_labels, preds, average=avg, zero_division=0)), 4),
    }

    # Best-case for binary
    if task == "binary":
        bc_labels = te_labels.copy()
        bc_labels[te_amb] = preds[te_amb]
        metrics["f1_bestcase"]       = round(float(f1_score(bc_labels, preds, average="binary", zero_division=0)), 4)
        metrics["accuracy_bestcase"] = round(float(accuracy_score(bc_labels, preds)), 4)

    print(f"\n  Fold {fold_idx + 1} results:")
    for k, v in metrics.items():
        if k not in ("fold", "n_train", "n_val", "n_test"):
            print(f"    {k:25s}: {v:.4f}")

    label_names = LABEL_NAMES_MAP[task]
    present_labels = sorted(np.unique(np.concatenate([te_labels, preds])))
    print(classification_report(
        te_labels, preds,
        labels=present_labels,
        target_names=[label_names[i] for i in present_labels],
        zero_division=0,
    ))

    # Per-sample DataFrame for this fold
    per_sample = pd.DataFrame({
        "fold":       fold_idx + 1,
        "paragraph_id": test_df["ID"].values,
        "text":         test_df["Text"].values,
        "y_true":       te_labels,
        "y_pred":       preds,
        "is_ambiguous": te_amb,
        "correct":      (te_labels == preds),
    })

    # Clean up fold checkpoint directory unless asked to keep
    if not args.keep_fold_models and Path(fold_dir).exists():
        shutil.rmtree(fold_dir)

    return metrics, per_sample


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stratified k-fold cross-validation for BERTić hate speech detection.",
    )
    parser.add_argument("--task",     type=str, default="binary",
                        choices=["binary", "category", "subcategory"])
    parser.add_argument("--k",        type=int, default=3,
                        help="Number of folds (default 3)")
    parser.add_argument("--epochs",   type=int,   default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr",       type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--dropout",  type=float, default=0.1)
    parser.add_argument("--weight_decay",   type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="cv_checkpoints",
                        help="Base directory for fold checkpoint files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output Excel path (default: results/cv_{task}_k{k}.xlsx)")
    parser.add_argument(
        "--sentence_path", type=str,
        default="data/single_sentence_hate_speech_no_offenses.xlsx",
    )
    parser.add_argument(
        "--paragraph_path", type=str,
        default="data/paragraph_hate_speech_no_offenses.xlsx",
    )
    parser.add_argument("--keep_fold_models", action="store_true",
                        help="Keep checkpoint directories after each fold (default: delete to save space)")
    args = parser.parse_args()

    output_path = Path(
        args.output or f"results/cv_{args.task}_k{args.k}.xlsx"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"\nLoading datasets...")
    df_sent = pd.read_excel(args.sentence_path)
    df_para = pd.read_excel(args.paragraph_path)

    para_lookup = dict(zip(df_para["ID"], df_para["Text"]))

    # Sentence-level labels and paragraph IDs
    parse_fn = PARSE_FN[args.task]
    parsed = [parse_fn(cat) for cat in df_sent["Category"]]
    y_all = np.array([p[0] for p in parsed], dtype=int)
    groups = df_sent["ID"].values  # paragraph ID per sentence

    n_total = len(df_sent)
    print(f"  Total sentences : {n_total}")
    print(f"  Total paragraphs: {len(df_para)}")
    print(f"  Task            : {args.task}  |  k={args.k}")
    print(f"  Label dist      : {dict(sorted(Counter(y_all).items()))}")

    # ── Stratified group k-fold ───────────────────────────────────────────
    sgkf = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    all_metrics = []
    all_per_sample = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        sgkf.split(df_sent, y=y_all, groups=groups)
    ):
        train_df = df_sent.iloc[train_idx].reset_index(drop=True)
        test_df  = df_sent.iloc[test_idx].reset_index(drop=True)

        print(f"\n  Fold {fold_idx + 1}/{args.k}:"
              f"  train paragraphs={train_df['ID'].nunique()}"
              f"  test paragraphs={test_df['ID'].nunique()}")

        metrics, per_sample = run_fold(
            fold_idx, train_df, test_df, para_lookup, args.task, args,
        )
        all_metrics.append(metrics)
        all_per_sample.append(per_sample)

    # ── Aggregate across folds ────────────────────────────────────────────
    metric_cols = [c for c in all_metrics[0].keys()
                   if c not in ("fold", "n_train", "n_val", "n_test")]

    print(f"\n{'=' * 70}")
    print(f"  {args.k}-FOLD CV RESULTS  |  task={args.task}")
    print(f"{'=' * 70}")
    for col in metric_cols:
        vals = [m[col] for m in all_metrics]
        print(f"  {col:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    aggregate_rows = []
    for col in metric_cols:
        vals = [m[col] for m in all_metrics]
        aggregate_rows.append({
            "metric":   col,
            "mean":     round(float(np.mean(vals)), 4),
            "std":      round(float(np.std(vals)), 4),
            "min":      round(float(np.min(vals)), 4),
            "max":      round(float(np.max(vals)), 4),
            **{f"fold_{i+1}": round(v, 4) for i, v in enumerate(vals)},
        })

    df_fold    = pd.DataFrame(all_metrics)
    df_agg     = pd.DataFrame(aggregate_rows)
    df_samples = pd.concat(all_per_sample, ignore_index=True)

    # Sort per-sample by paragraph then by original sentence order
    df_samples.sort_values(["paragraph_id", "fold"], inplace=True)

    # ── Save to Excel ─────────────────────────────────────────────────────
    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        df_fold.to_excel(writer, sheet_name="PerFold", index=False)
        df_agg.to_excel(writer, sheet_name="Aggregate", index=False)
        df_samples.to_excel(writer, sheet_name="PerSample", index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
