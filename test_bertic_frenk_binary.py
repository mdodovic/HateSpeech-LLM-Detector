"""
Test the pretrained classla/bcms-bertic-frenk-hate model on the fixed
sentence-level test split.

This script does not train or fine-tune anything. It downloads the Hugging Face
checkpoint if needed, evaluates sentences from paragraphs 1..101 one sentence
at a time, and writes per-sentence predictions plus binary metrics and
bootstrap confidence intervals to Excel.

Default input:
  data/single_sentence_hate_speech_no_offenses.xlsx

Default output:
  results/bertic/bertic_frenk_binary_test_101.xlsx

Example:
  python test_bertic_frenk_binary.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "classla/bcms-bertic-frenk-hate"

VALID_CODES = {
    "0",
    "1a",
    "1b",
    "1c",
    "2",
    "3a",
    "3b",
    "4a",
    "4b",
    "5",
    "6a",
    "6b",
    "6c",
    "7",
}

LABEL_NAMES = {0: "No hate", 1: "Hate"}


def extract_codes(category_value: object) -> list[str]:
    """Extract valid dataset category codes from a possibly multi-label cell."""
    if category_value is None or pd.isna(category_value):
        return []

    codes: list[str] = []
    for part in str(category_value).split(","):
        part = part.strip()
        if not part:
            continue

        if part.startswith("(") and part.endswith(")"):
            candidates: Iterable[str] = part[1:-1].split(";")
        else:
            candidates = part.split(":")

        for candidate in candidates:
            code = candidate.strip().lower()
            if code in VALID_CODES:
                codes.append(code)

    return codes


def parse_binary_label(category_value: object) -> tuple[int, bool]:
    """
    Convert a dataset category cell to (binary_label, is_ambiguous).

    Strict label: any valid non-zero category code means hate.
    Ambiguous: both 0 and a hate code are present in the same annotation.
    """
    codes = set(extract_codes(category_value))
    has_hate = any(code != "0" for code in codes)
    has_no_hate = "0" in codes
    return int(has_hate), bool(has_hate and has_no_hate)


def load_test_sentences(args: argparse.Namespace) -> pd.DataFrame:
    sentence_path = Path(args.sentence_path)
    if not sentence_path.exists():
        raise FileNotFoundError(f"Sentence dataset not found: {sentence_path}")

    df = pd.read_excel(sentence_path)
    missing = {"ID", "Text", "Category"} - set(df.columns)
    if missing:
        raise ValueError(f"{sentence_path} is missing required columns: {sorted(missing)}")

    numeric_id = pd.to_numeric(df["ID"], errors="coerce")
    mask = numeric_id.between(
        args.min_paragraph_id,
        args.max_paragraph_id,
        inclusive="both",
    )
    test_df = df.loc[mask].copy().reset_index(drop=True)
    if test_df.empty:
        raise ValueError(
            "No test rows found for paragraph IDs "
            f"{args.min_paragraph_id}..{args.max_paragraph_id}"
        )

    parsed = [parse_binary_label(value) for value in test_df["Category"]]
    test_df["binary_label"] = [label for label, _ in parsed]
    test_df["is_ambiguous"] = [ambiguous for _, ambiguous in parsed]
    test_df["sentence_index_in_paragraph"] = test_df.groupby("ID").cumcount() + 1
    test_df["Text"] = test_df["Text"].fillna("").astype(str)

    if args.use_paragraph_context:
        paragraph_path = Path(args.paragraph_path)
        if not paragraph_path.exists():
            raise FileNotFoundError(f"Paragraph dataset not found: {paragraph_path}")
        para_df = pd.read_excel(paragraph_path)
        missing_para = {"ID", "Text"} - set(para_df.columns)
        if missing_para:
            raise ValueError(
                f"{paragraph_path} is missing required columns: {sorted(missing_para)}"
            )
        para_lookup = dict(zip(para_df["ID"], para_df["Text"]))
        test_df["paragraph_text"] = test_df["ID"].map(para_lookup).fillna("").astype(str)
        missing_context = int((test_df["paragraph_text"].str.len() == 0).sum())
        if missing_context:
            raise ValueError(f"Missing paragraph context for {missing_context} test rows")

    return test_df


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def batched_prediction(
    df: pd.DataFrame,
    tokenizer,
    model,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    logits_batches: list[np.ndarray] = []
    n_rows = len(df)
    batch_starts = range(0, n_rows, args.batch_size)

    iterator = tqdm(
        batch_starts,
        total=(n_rows + args.batch_size - 1) // args.batch_size,
        desc="Testing",
        unit="batch",
        disable=args.no_progress,
    )

    model.eval()
    with torch.inference_mode():
        for start in iterator:
            batch = df.iloc[start : start + args.batch_size]
            sentences = batch["Text"].tolist()

            if args.use_paragraph_context:
                encoded = tokenizer(
                    batch["paragraph_text"].tolist(),
                    sentences,
                    truncation="only_first",
                    padding=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
            else:
                encoded = tokenizer(
                    sentences,
                    truncation=True,
                    padding=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )

            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            logits_batches.append(outputs.logits.detach().cpu().numpy())

    logits = np.concatenate(logits_batches, axis=0)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return logits, probs


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> tuple[float, float, float]:
    """Return point estimate plus percentile bootstrap confidence interval."""
    point = float(metric_fn(y_true, y_pred))
    if n_bootstrap <= 0:
        return point, np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(scores, alpha * 100.0))
    upper = float(np.percentile(scores, (1.0 - alpha) * 100.0))
    return point, lower, upper


def bootstrap_ci_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    best_case_true: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Build strict and best-case bootstrap CIs for binary metrics."""
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "Accuracy": lambda yt, yp: accuracy_score(yt, yp),
        "F1": lambda yt, yp: f1_score(yt, yp, average="binary", zero_division=0),
        "Precision": lambda yt, yp: precision_score(
            yt, yp, average="binary", zero_division=0
        ),
        "Recall": lambda yt, yp: recall_score(yt, yp, average="binary", zero_division=0),
    }

    rows = []
    for mode, labels in [("Strict", y_true), ("Best-case", best_case_true)]:
        for offset, (metric_name, fn) in enumerate(metric_fns.items()):
            point, lower, upper = bootstrap_ci(
                labels,
                y_pred,
                fn,
                n_bootstrap=args.n_bootstrap,
                ci=args.ci,
                seed=args.bootstrap_seed + offset,
            )
            rows.append(
                {
                    "Mode": mode,
                    "Metric": metric_name,
                    "Value": point,
                    "CI_lower": lower,
                    "CI_upper": upper,
                    "CI": args.ci,
                    "N_bootstrap": args.n_bootstrap,
                    "Seed": args.bootstrap_seed + offset,
                }
            )
    return pd.DataFrame(rows)


def report_to_dataframe(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    report_df.insert(0, "label", report_df.index)
    return report_df.reset_index(drop=True)


def model_id2label(model) -> dict[str, str]:
    raw = getattr(model.config, "id2label", {}) or {}
    return {str(key): str(value) for key, value in raw.items()}


def write_results(
    output_path: Path,
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    best_case_true: np.ndarray,
    logits: np.ndarray,
    probs: np.ndarray,
    model,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    label_ids = np.argmax(logits, axis=1)
    id2label = model_id2label(model)

    pred_df = pd.DataFrame(
        {
            "ID": df["ID"].values,
            "sentence_index_in_paragraph": df["sentence_index_in_paragraph"].values,
            "Text": df["Text"].values,
            "Category": df["Category"].values,
            "GT_label_id": y_true,
            "GT_label": [LABEL_NAMES[int(label)] for label in y_true],
            "Model_label_id": label_ids,
            "Model_label": [id2label.get(str(int(label)), str(int(label))) for label in label_ids],
            "Predicted_label_id": y_pred,
            "Predicted_label": [LABEL_NAMES[int(label)] for label in y_pred],
            "Prob_no_hate": probs[:, 0] if probs.shape[1] > 0 else np.nan,
            "Prob_hate": probs[:, args.hate_label_id],
            "Is_ambiguous": df["is_ambiguous"].values,
            "Strict_correct": y_pred == y_true,
            "BestCase_correct": y_pred == best_case_true,
        }
    )

    strict_metrics = compute_binary_metrics(y_true, y_pred)
    best_metrics = compute_binary_metrics(best_case_true, y_pred)
    bootstrap_df = bootstrap_ci_dataframe(y_true, y_pred, best_case_true, args)
    metrics_df = pd.DataFrame(
        {
            "Metric": list(strict_metrics.keys()),
            "Strict": list(strict_metrics.values()),
            "Best-case": [best_metrics[key] for key in strict_metrics.keys()],
        }
    )

    meta_df = pd.DataFrame(
        {
            "key": [
                "model_name",
                "sentence_path",
                "paragraph_path",
                "input_mode",
                "paragraph_id_min",
                "paragraph_id_max",
                "n_sentences",
                "n_paragraphs",
                "batch_size",
                "max_length",
                "device",
                "hate_label_id",
                "n_bootstrap",
                "ci",
                "bootstrap_seed",
                "id2label",
            ],
            "value": [
                args.model_name,
                args.sentence_path,
                args.paragraph_path if args.use_paragraph_context else "",
                "paragraph_plus_sentence" if args.use_paragraph_context else "sentence_only",
                args.min_paragraph_id,
                args.max_paragraph_id,
                len(df),
                df["ID"].nunique(),
                args.batch_size,
                args.max_length,
                str(device),
                args.hate_label_id,
                args.n_bootstrap,
                args.ci,
                args.bootstrap_seed,
                json.dumps(id2label, ensure_ascii=False),
            ],
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pred_df.to_excel(writer, sheet_name="Predictions", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        bootstrap_df.to_excel(writer, sheet_name="BootstrapCI", index=False)
        report_to_dataframe(y_true, y_pred).to_excel(
            writer,
            sheet_name="StrictReport",
            index=False,
        )
        report_to_dataframe(best_case_true, y_pred).to_excel(
            writer,
            sheet_name="BestCaseReport",
            index=False,
        )
        meta_df.to_excel(writer, sheet_name="Meta", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test classla/bcms-bertic-frenk-hate on sentences from paragraphs "
            "1..101 without training."
        )
    )
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument(
        "--sentence_path",
        default="data/single_sentence_hate_speech_no_offenses.xlsx",
    )
    parser.add_argument(
        "--paragraph_path",
        default="data/paragraph_hate_speech_no_offenses.xlsx",
        help="Only used with --use_paragraph_context.",
    )
    parser.add_argument(
        "--output",
        default="results/bertic/bertic_frenk_binary_test_101.xlsx",
    )
    parser.add_argument("--min_paragraph_id", type=int, default=1)
    parser.add_argument("--max_paragraph_id", type=int, default=101)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--hate_label_id",
        type=int,
        default=1,
        help="Model label ID to count as hate/offensive. Default: 1.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:0. Default: auto.",
    )
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for confidence intervals. Use 0 to skip CIs.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Bootstrap confidence level. Default: 0.95.",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling. Default: 42.",
    )
    parser.add_argument(
        "--use_paragraph_context",
        action="store_true",
        help="Encode paragraph as text A and sentence as text B. Default is sentence only.",
    )
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate and summarize the test split without loading the model.",
    )
    args = parser.parse_args()
    if not 0 < args.ci < 1:
        raise ValueError("--ci must be between 0 and 1")
    if args.n_bootstrap < 0:
        raise ValueError("--n_bootstrap must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    test_df = load_test_sentences(args)

    y_true = test_df["binary_label"].to_numpy(dtype=int)
    ambiguous = test_df["is_ambiguous"].to_numpy(dtype=bool)

    print("Loaded test split")
    print(f"  Sentences : {len(test_df)}")
    print(f"  Paragraphs: {test_df['ID'].nunique()}")
    print(f"  ID range  : {args.min_paragraph_id}..{args.max_paragraph_id}")
    print(f"  Hate      : {int(y_true.sum())}")
    print(f"  No hate   : {int((y_true == 0).sum())}")
    print(f"  Ambiguous : {int(ambiguous.sum())}")

    if args.dry_run:
        print("Dry run complete. Model was not loaded.")
        return

    device = select_device(args.device)
    print(f"\nLoading model: {args.model_name}")
    print("Hugging Face will download the checkpoint if it is not already cached.")

    pretrained_kwargs = {
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
    }
    if args.revision:
        pretrained_kwargs["revision"] = args.revision

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **pretrained_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        **pretrained_kwargs,
    )
    model.to(device)

    num_labels = int(getattr(model.config, "num_labels", 0) or 0)
    if num_labels <= args.hate_label_id:
        raise ValueError(
            f"Model has {num_labels} labels, but --hate_label_id={args.hate_label_id}"
        )

    print(f"Model labels: {model_id2label(model)}")
    print(f"Counting model label {args.hate_label_id} as Hate")

    logits, probs = batched_prediction(test_df, tokenizer, model, device, args)
    model_label_ids = np.argmax(logits, axis=1)
    y_pred = (model_label_ids == args.hate_label_id).astype(int)

    best_case_true = y_true.copy()
    best_case_true[ambiguous] = y_pred[ambiguous]

    strict_metrics = compute_binary_metrics(y_true, y_pred)
    best_metrics = compute_binary_metrics(best_case_true, y_pred)

    print("\nStrict metrics")
    for key, value in strict_metrics.items():
        print(f"  {key:10s}: {value:.4f}" if isinstance(value, float) else f"  {key:10s}: {value}")

    print("\nBest-case metrics")
    for key, value in best_metrics.items():
        print(f"  {key:10s}: {value:.4f}" if isinstance(value, float) else f"  {key:10s}: {value}")

    if args.n_bootstrap > 0:
        print(
            f"\nBootstrap CI: {args.n_bootstrap} resamples, "
            f"{args.ci:.0%} confidence, seed {args.bootstrap_seed}"
        )

    output_path = Path(args.output)
    write_results(
        output_path=output_path,
        df=test_df,
        y_true=y_true,
        y_pred=y_pred,
        best_case_true=best_case_true,
        logits=logits,
        probs=probs,
        model=model,
        device=device,
        args=args,
    )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
