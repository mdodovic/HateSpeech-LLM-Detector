"""
Error space analysis: Zero-shot LLaMA vs Zero-shot Qwen.

For each task (binary / category / subcategory) the script builds a 4-quadrant
error map:
    ┌─────────────────┬───────────────────┐
    │  Both Correct   │  Qwen Only        │
    ├─────────────────┼───────────────────┤
    │  LLaMA Only     │  Both Wrong       │
    └─────────────────┴───────────────────┘
where "LLaMA Only" means LLaMA is correct and Qwen is wrong, etc.

Binary task   – all 548 samples, individual FP/FN/TP/TN labels, breakdown by
                ground-truth category.
Category task – aligned on all GT hate-speech samples (n = |binary y_true=1|).
                Samples missed at the binary stage are counted as wrong on
                category regardless of the (missing) category prediction.
Subcategory   – same alignment as category.

Output: results/error_space_zeroshot.xlsx
  Sheets:
    Summary                  – overall metrics + 4-quadrant counts for all tasks
    Binary_PerSample         – per-sample table (text + predictions + quadrant)
    Binary_QuadrantMatrix    – 2×2 contingency table (counts + %)
    Binary_ErrorsByTrueLabel – FP/FN/TP/TN per GT category
    Category_PerSample       – aligned hate-speech samples, category task
    Category_QuadrantMatrix  – 2×2 for category task
    Subcategory_PerSample    – aligned hate-speech samples, subcategory task
    Subcategory_QuadrantMatrix

Usage:
    python error_space_analysis.py
    python error_space_analysis.py --prompt two_prompts --output results/my_analysis.xlsx
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VALID_CODES = {
    "0", "1a", "1b", "1c", "2", "3a", "3b",
    "4a", "4b", "5", "6a", "6b", "6c", "7",
}
SUBCAT_TO_CAT = {
    "0": 0, "1a": 1, "1b": 1, "1c": 1, "2": 2,
    "3a": 3, "3b": 3, "4a": 4, "4b": 4, "5": 5,
    "6a": 6, "6b": 6, "6c": 6, "7": 7,
}
SUBCAT_TO_IDX = {
    "0": 0, "1a": 1, "1b": 2, "1c": 3, "2": 4,
    "3a": 5, "3b": 6, "4a": 7, "4b": 8, "5": 9,
    "6a": 10, "6b": 11, "6c": 12, "7": 13,
}
CATEGORY_NAMES = {
    0: "0 No hate", 1: "1 Racial/ethnic", 2: "2 Religious",
    3: "3 Sex/gender", 4: "4 Physical/health", 5: "5 Age",
    6: "6 Socioeconomic/political", 7: "7 Sports/fan",
}
SUBCATEGORY_NAMES = {
    0: "0 No hate", 1: "1a Race/skin", 2: "1b Ethnic", 3: "1c Nationality",
    4: "2 Religious", 5: "3a Sex/sexism", 6: "3b LGBTQ+", 7: "4a Appearance",
    8: "4b Illness/disability", 9: "5 Age", 10: "6a Socioeconomic",
    11: "6b Political", 12: "6c Regional", 13: "7 Sports/fan",
}

QUADRANT_LABELS = {
    (True,  True):  "Both Correct",
    (True,  False): "LLaMA Only",
    (False, True):  "Qwen Only",
    (False, False): "Both Wrong",
}

# ─────────────────────────────────────────────────────────────────────────────
# Label parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_codes(cat_str: str):
    codes = []
    for part in str(cat_str).split(","):
        part = part.strip()
        if part.startswith("(") and part.endswith(")"):
            for c in part[1:-1].split(";"):
                codes.append(c.strip().lower())
        else:
            for sub in part.split(":"):
                codes.append(sub.strip().lower())
    return [c for c in codes if c in VALID_CODES]


def parse_binary(cat_str):
    codes = set(_extract_codes(cat_str))
    return int(any(c != "0" for c in codes))


def parse_category(cat_str):
    codes = _extract_codes(cat_str)
    return next((SUBCAT_TO_CAT[c] for c in codes if c != "0"), 0)


def parse_subcategory(cat_str):
    codes = _extract_codes(cat_str)
    return next((SUBCAT_TO_IDX[c] for c in codes if c != "0"), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_binary(per_sample_path: Path, model: str, prompt: str) -> pd.Series:
    """Return a Series of predictions (548 values) for the given model/prompt."""
    df = pd.read_excel(str(per_sample_path), sheet_name="Binary")
    sub = df[(df["model"] == model) & (df["prompt_type"] == prompt)].reset_index(drop=True)
    return sub["y_pred"]


def load_classification(
    per_sample_path: Path,
    binary_preds_llama: np.ndarray,
    binary_preds_qwen: np.ndarray,
    binary_true: np.ndarray,
    model: str,
    prompt: str,
    sheet: str,
) -> np.ndarray:
    """
    Reconstruct a correct/wrong boolean array aligned to ALL GT hate-speech
    samples (where binary_true == 1).

    For each GT hate sample:
      - If the model's binary pred = 0  → wrong on classification (missed)
      - If the model's binary pred = 1  → look up category sheet to check if
                                          category pred == GT category label

    Returns a boolean array of length n_hate = sum(binary_true == 1).
    """
    df_cls = pd.read_excel(str(per_sample_path), sheet_name=sheet)
    sub = df_cls[(df_cls["model"] == model) & (df_cls["prompt_type"] == prompt)].reset_index(drop=True)

    cls_true = sub["y_true"].values
    cls_pred = sub["y_pred"].values

    # Map binary detected positions → classification sheet rows
    # binary_preds[i] == 1 means row i was sent to the classification stage
    detected_positions = np.where(binary_preds_llama if model == "llama3" else binary_preds_qwen)[0]
    # Build a dict: sample_index → classification_row_index
    det_to_cls = {sample_i: j for j, sample_i in enumerate(detected_positions)}

    # Ground-truth hate positions
    hate_positions = np.where(binary_true == 1)[0]

    correct = np.zeros(len(hate_positions), dtype=bool)
    for out_idx, sample_i in enumerate(hate_positions):
        if sample_i not in det_to_cls:
            correct[out_idx] = False   # missed (binary FN)
        else:
            j = det_to_cls[sample_i]
            correct[out_idx] = cls_pred[j] == cls_true[j]

    return correct


def load_classification_preds(
    per_sample_path: Path,
    binary_preds: np.ndarray,
    binary_true: np.ndarray,
    model: str,
    prompt: str,
    sheet: str,
) -> tuple:
    """
    Return (y_true_aligned, y_pred_aligned) for ALL GT hate-speech samples.
    Missed samples get y_pred = -1 to signal 'not detected'.

    Category sheet  : rows correspond to binary_pred==1 positions (in order).
    Subcategory sheet: rows correspond to binary_pred==1 AND binary_true==1
                       positions (TPs only, in order) — the pipeline only runs
                       subcategory for GT hate-speech samples that were detected.
    """
    df_cls = pd.read_excel(str(per_sample_path), sheet_name=sheet)
    sub = df_cls[(df_cls["model"] == model) & (df_cls["prompt_type"] == prompt)].reset_index(drop=True)

    cls_true = sub["y_true"].values
    cls_pred = sub["y_pred"].values

    hate_positions = np.where(binary_true == 1)[0]

    if sheet == "Subcategory":
        # Subcategory rows = TPs (binary_pred=1 AND binary_true=1), in sample order
        # y_true/y_pred are letter strings ("a", "b", ...) or "" for no-subcat categories
        tp_positions = np.where((binary_preds == 1) & (binary_true == 1))[0]
        tp_to_sub = {orig_i: j for j, orig_i in enumerate(tp_positions)}
        # For subcategory, return object arrays (strings); missed = "__missed__"
        yt = np.array([str(cls_true[tp_to_sub[i]]) if i in tp_to_sub else "__gt__"
                       for i in hate_positions], dtype=object)
        yp = np.array([str(cls_pred[tp_to_sub[i]]) if i in tp_to_sub else "__missed__"
                       for i in hate_positions], dtype=object)
    else:
        # Category rows = all binary_pred==1 positions, in sample order
        detected_positions = np.where(binary_preds == 1)[0]
        det_to_cls = {sample_i: j for j, sample_i in enumerate(detected_positions)}
        yt = np.array([cls_true[det_to_cls[i]] if i in det_to_cls else binary_true[i]
                       for i in hate_positions])
        yp = np.array([cls_pred[det_to_cls[i]] if i in det_to_cls else -1
                       for i in hate_positions])

    return yt, yp


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def quadrant_matrix(correct_a: np.ndarray, correct_b: np.ndarray) -> pd.DataFrame:
    """2×2 contingency table: rows = model A, cols = model B."""
    cc  = int(np.sum( correct_a &  correct_b))  # both correct
    cw  = int(np.sum( correct_a & ~correct_b))  # A only
    wc  = int(np.sum(~correct_a &  correct_b))  # B only
    ww  = int(np.sum(~correct_a & ~correct_b))  # both wrong
    n   = len(correct_a)
    data = {
        "": ["LLaMA Correct", "LLaMA Wrong"],
        "Qwen Correct":  [f"{cc} ({cc/n:.1%})", f"{wc} ({wc/n:.1%})"],
        "Qwen Wrong":    [f"{cw} ({cw/n:.1%})", f"{ww} ({ww/n:.1%})"],
    }
    return pd.DataFrame(data)


def error_type(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:  return "TP"
    if y_true == 0 and y_pred == 0:  return "TN"
    if y_true == 0 and y_pred == 1:  return "FP"
    return "FN"


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, average="binary", zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="binary", zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, average="binary", zero_division=0)), 4),
    }


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = y_pred >= 0  # exclude missed samples (-1) from metric computation
    if mask.sum() == 0:
        return {"accuracy": 0, "f1_macro": 0, "precision_macro": 0, "recall_macro": 0}
    return {
        "accuracy":         round(float(accuracy_score(y_true[mask], y_pred[mask])), 4),
        "f1_macro":         round(float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)), 4),
        "precision_macro":  round(float(precision_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)), 4),
        "recall_macro":     round(float(recall_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Error space analysis: LLaMA vs Qwen zero-shot")
    parser.add_argument("--prompt",  default="two_prompts",
                        help="Prompt type to analyse (default: two_prompts)")
    parser.add_argument("--per_sample", default="results/single_sentence_per_sample.xlsx")
    parser.add_argument("--sentences",  default="data/single_sentence_hate_speech_no_offenses.xlsx")
    parser.add_argument("--output",     default="results/error_space_zeroshot.xlsx")
    args = parser.parse_args()

    per_sample_path = Path(args.per_sample)
    output_path     = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load original 548 sentences ───────────────────────────────────────
    df_sent = pd.read_excel(args.sentences)
    sent548 = df_sent[df_sent["ID"] <= 101].reset_index(drop=True)
    assert len(sent548) == 548, f"Expected 548 sentences, got {len(sent548)}"

    gt_binary   = sent548["Category"].apply(parse_binary).values
    gt_category = sent548["Category"].apply(parse_category).values
    gt_subcat   = sent548["Category"].apply(parse_subcategory).values

    # ── Load binary predictions ───────────────────────────────────────────
    llama_bin = load_binary(per_sample_path, "llama3", args.prompt).values
    qwen_bin  = load_binary(per_sample_path, "qwen3",  args.prompt).values
    assert len(llama_bin) == 548 and len(qwen_bin) == 548

    # ── Binary per-sample table ───────────────────────────────────────────
    llama_correct_bin = llama_bin == gt_binary
    qwen_correct_bin  = qwen_bin  == gt_binary

    binary_per_sample = pd.DataFrame({
        "paragraph_id":      sent548["ID"].values,
        "sentence":          sent548["Text"].values,
        "gt_category_str":   sent548["Category"].values,
        "gt_binary":         gt_binary,
        "gt_category":       [CATEGORY_NAMES.get(c, str(c)) for c in gt_category],
        "llama_pred":        llama_bin,
        "qwen_pred":         qwen_bin,
        "llama_error_type":  [error_type(t, p) for t, p in zip(gt_binary, llama_bin)],
        "qwen_error_type":   [error_type(t, p) for t, p in zip(gt_binary, qwen_bin)],
        "quadrant": [
            QUADRANT_LABELS[(bool(lc), bool(qc))]
            for lc, qc in zip(llama_correct_bin, qwen_correct_bin)
        ],
    })

    # ── Binary: errors grouped by GT category ─────────────────────────────
    error_by_cat_rows = []
    for cat_int, cat_name in sorted(CATEGORY_NAMES.items()):
        mask = gt_category == cat_int
        if mask.sum() == 0:
            continue
        n = int(mask.sum())
        sub = binary_per_sample[mask]
        for model_col, err_col in [("llama_pred", "llama_error_type"),
                                   ("qwen_pred",  "qwen_error_type")]:
            model_name = "LLaMA" if "llama" in model_col else "Qwen"
            vc = sub[err_col].value_counts()
            error_by_cat_rows.append({
                "gt_category":  cat_name,
                "n_samples":    n,
                "model":        model_name,
                "TP": int(vc.get("TP", 0)),
                "TN": int(vc.get("TN", 0)),
                "FP": int(vc.get("FP", 0)),
                "FN": int(vc.get("FN", 0)),
            })

    errors_by_cat = pd.DataFrame(error_by_cat_rows)

    # ── Binary metrics summary ────────────────────────────────────────────
    llama_bin_metrics = binary_metrics(gt_binary, llama_bin)
    qwen_bin_metrics  = binary_metrics(gt_binary, qwen_bin)

    b10 = int(np.sum( llama_correct_bin & ~qwen_correct_bin))
    b01 = int(np.sum(~llama_correct_bin &  qwen_correct_bin))
    n_agree_correct = int(np.sum( llama_correct_bin &  qwen_correct_bin))
    n_agree_wrong   = int(np.sum(~llama_correct_bin & ~qwen_correct_bin))
    n_discordant    = b10 + b01

    summary_rows = [
        {"task": "Binary", "metric": "n_samples",        "LLaMA": 548,             "Qwen": 548},
        {"task": "Binary", "metric": "accuracy",         "LLaMA": llama_bin_metrics["accuracy"], "Qwen": qwen_bin_metrics["accuracy"]},
        {"task": "Binary", "metric": "f1",               "LLaMA": llama_bin_metrics["f1"],        "Qwen": qwen_bin_metrics["f1"]},
        {"task": "Binary", "metric": "precision",        "LLaMA": llama_bin_metrics["precision"],  "Qwen": qwen_bin_metrics["precision"]},
        {"task": "Binary", "metric": "recall",           "LLaMA": llama_bin_metrics["recall"],     "Qwen": qwen_bin_metrics["recall"]},
        {"task": "Binary", "metric": "TP",               "LLaMA": int(np.sum((llama_bin==1)&(gt_binary==1))), "Qwen": int(np.sum((qwen_bin==1)&(gt_binary==1)))},
        {"task": "Binary", "metric": "TN",               "LLaMA": int(np.sum((llama_bin==0)&(gt_binary==0))), "Qwen": int(np.sum((qwen_bin==0)&(gt_binary==0)))},
        {"task": "Binary", "metric": "FP",               "LLaMA": int(np.sum((llama_bin==1)&(gt_binary==0))), "Qwen": int(np.sum((qwen_bin==1)&(gt_binary==0)))},
        {"task": "Binary", "metric": "FN",               "LLaMA": int(np.sum((llama_bin==0)&(gt_binary==1))), "Qwen": int(np.sum((qwen_bin==0)&(gt_binary==1)))},
        {"task": "Binary", "metric": "quadrant_both_correct",  "LLaMA": n_agree_correct, "Qwen": n_agree_correct},
        {"task": "Binary", "metric": "quadrant_llama_only",    "LLaMA": b10,             "Qwen": b10},
        {"task": "Binary", "metric": "quadrant_qwen_only",     "LLaMA": b01,             "Qwen": b01},
        {"task": "Binary", "metric": "quadrant_both_wrong",    "LLaMA": n_agree_wrong,   "Qwen": n_agree_wrong},
        {"task": "Binary", "metric": "error_agreement_rate",   "LLaMA": round(n_agree_wrong/548, 4), "Qwen": round(n_agree_wrong/548, 4)},
        {"task": "Binary", "metric": "discordant_pairs",       "LLaMA": n_discordant,    "Qwen": n_discordant},
    ]
    summary_df = pd.DataFrame(summary_rows)[["task", "metric", "LLaMA", "Qwen"]]

    # ── Category task – aligned on GT hate speech samples ─────────────────
    hate_positions = np.where(gt_binary == 1)[0]
    n_hate = len(hate_positions)

    llama_cat_yt, llama_cat_yp = load_classification_preds(
        per_sample_path, llama_bin, gt_binary, "llama3", args.prompt, "Category"
    )
    qwen_cat_yt, qwen_cat_yp = load_classification_preds(
        per_sample_path, qwen_bin, gt_binary, "qwen3", args.prompt, "Category"
    )

    llama_cat_correct = np.array([
        (int(yp) >= 0 and yp == yt) for yt, yp in zip(llama_cat_yt, llama_cat_yp)
    ])
    qwen_cat_correct = np.array([
        (int(yp) >= 0 and yp == yt) for yt, yp in zip(qwen_cat_yt, qwen_cat_yp)
    ])

    cat_per_sample = pd.DataFrame({
        "paragraph_id":     sent548.loc[hate_positions, "ID"].values,
        "sentence":         sent548.loc[hate_positions, "Text"].values,
        "gt_category_str":  sent548.loc[hate_positions, "Category"].values,
        "gt_category_int":  llama_cat_yt,
        "gt_category_name": [CATEGORY_NAMES.get(int(c), str(c)) for c in llama_cat_yt],
        "llama_cat_pred":   [int(p) if p >= 0 else "missed" for p in llama_cat_yp],
        "qwen_cat_pred":    [int(p) if p >= 0 else "missed" for p in qwen_cat_yp],
        "llama_correct":    llama_cat_correct,
        "qwen_correct":     qwen_cat_correct,
        "quadrant": [
            QUADRANT_LABELS[(bool(lc), bool(qc))]
            for lc, qc in zip(llama_cat_correct, qwen_cat_correct)
        ],
    })

    # ── Subcategory task ──────────────────────────────────────────────────
    llama_sub_yt, llama_sub_yp = load_classification_preds(
        per_sample_path, llama_bin, gt_binary, "llama3", args.prompt, "Subcategory"
    )
    qwen_sub_yt, qwen_sub_yp = load_classification_preds(
        per_sample_path, qwen_bin, gt_binary, "qwen3", args.prompt, "Subcategory"
    )

    llama_sub_correct = np.array([
        (yp != "__missed__" and yp == yt) for yt, yp in zip(llama_sub_yt, llama_sub_yp)
    ])
    qwen_sub_correct = np.array([
        (yp != "__missed__" and yp == yt) for yt, yp in zip(qwen_sub_yt, qwen_sub_yp)
    ])

    sub_per_sample = pd.DataFrame({
        "paragraph_id":       sent548.loc[hate_positions, "ID"].values,
        "sentence":           sent548.loc[hate_positions, "Text"].values,
        "gt_category_str":    sent548.loc[hate_positions, "Category"].values,
        "gt_subcat_letter":   llama_sub_yt,
        "llama_subcat_pred":  ["missed" if p == "__missed__" else p for p in llama_sub_yp],
        "qwen_subcat_pred":   ["missed" if p == "__missed__" else p for p in qwen_sub_yp],
        "llama_correct":      llama_sub_correct,
        "qwen_correct":       qwen_sub_correct,
        "quadrant": [
            QUADRANT_LABELS[(bool(lc), bool(qc))]
            for lc, qc in zip(llama_sub_correct, qwen_sub_correct)
        ],
    })

    # ── Print summary to console ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Error Space Analysis — Zero-shot LLaMA vs Zero-shot Qwen")
    print(f"  Prompt type : {args.prompt}")
    print(f"{'='*70}")

    print(f"\n── Binary (n=548) ───────────────────────────────────────────────────")
    for m, lv, qv in [("accuracy",  llama_bin_metrics["accuracy"],  qwen_bin_metrics["accuracy"]),
                       ("f1",        llama_bin_metrics["f1"],        qwen_bin_metrics["f1"]),
                       ("precision", llama_bin_metrics["precision"], qwen_bin_metrics["precision"]),
                       ("recall",    llama_bin_metrics["recall"],    qwen_bin_metrics["recall"])]:
        print(f"  {m:12s}  LLaMA={lv:.4f}  Qwen={qv:.4f}")
    print(f"\n  4-Quadrant breakdown (n={548}):")
    print(f"    Both Correct  : {n_agree_correct:4d}  ({n_agree_correct/548:.1%})")
    print(f"    LLaMA Only    : {b10:4d}  ({b10/548:.1%})")
    print(f"    Qwen Only     : {b01:4d}  ({b01/548:.1%})")
    print(f"    Both Wrong    : {n_agree_wrong:4d}  ({n_agree_wrong/548:.1%})")

    print(f"\n── Category (n={n_hate} GT hate-speech samples) ──────────────────────")
    lc_n = int(llama_cat_correct.sum());  qc_n = int(qwen_cat_correct.sum())
    cat_b10 = int(np.sum( llama_cat_correct & ~qwen_cat_correct))
    cat_b01 = int(np.sum(~llama_cat_correct &  qwen_cat_correct))
    cat_bc  = int(np.sum( llama_cat_correct &  qwen_cat_correct))
    cat_bw  = int(np.sum(~llama_cat_correct & ~qwen_cat_correct))
    print(f"  LLaMA correct : {lc_n}/{n_hate}  ({lc_n/n_hate:.1%})")
    print(f"  Qwen correct  : {qc_n}/{n_hate}  ({qc_n/n_hate:.1%})")
    print(f"  Both Correct  : {cat_bc:4d}  ({cat_bc/n_hate:.1%})")
    print(f"  LLaMA Only    : {cat_b10:4d}  ({cat_b10/n_hate:.1%})")
    print(f"  Qwen Only     : {cat_b01:4d}  ({cat_b01/n_hate:.1%})")
    print(f"  Both Wrong    : {cat_bw:4d}  ({cat_bw/n_hate:.1%})")

    print(f"\n── Subcategory (n={n_hate} GT hate-speech samples) ───────────────────")
    ls_n = int(llama_sub_correct.sum());  qs_n = int(qwen_sub_correct.sum())
    sub_b10 = int(np.sum( llama_sub_correct & ~qwen_sub_correct))
    sub_b01 = int(np.sum(~llama_sub_correct &  qwen_sub_correct))
    sub_bc  = int(np.sum( llama_sub_correct &  qwen_sub_correct))
    sub_bw  = int(np.sum(~llama_sub_correct & ~qwen_sub_correct))
    print(f"  LLaMA correct : {ls_n}/{n_hate}  ({ls_n/n_hate:.1%})")
    print(f"  Qwen correct  : {qs_n}/{n_hate}  ({qs_n/n_hate:.1%})")
    print(f"  Both Correct  : {sub_bc:4d}  ({sub_bc/n_hate:.1%})")
    print(f"  LLaMA Only    : {sub_b10:4d}  ({sub_b10/n_hate:.1%})")
    print(f"  Qwen Only     : {sub_b01:4d}  ({sub_b01/n_hate:.1%})")
    print(f"  Both Wrong    : {sub_bw:4d}  ({sub_bw/n_hate:.1%})")

    # ── Save to Excel ─────────────────────────────────────────────────────
    bin_quad   = quadrant_matrix(llama_correct_bin, qwen_correct_bin)
    cat_quad   = quadrant_matrix(llama_cat_correct, qwen_cat_correct)
    sub_quad   = quadrant_matrix(llama_sub_correct, qwen_sub_correct)

    # ── Confusion matrices (actual vs predicted, per model) ───────────────
    def make_conf_df(y_true, y_pred, labels, label_names):
        """Build a labelled confusion matrix as a DataFrame."""
        from sklearn.metrics import confusion_matrix as sk_cm
        cm = sk_cm(y_true, y_pred, labels=labels)
        col_names = [f"Pred: {label_names.get(l, str(l))}" for l in labels]
        row_names = [f"True: {label_names.get(l, str(l))}" for l in labels]
        df = pd.DataFrame(cm, index=row_names, columns=col_names)
        df.index.name = ""
        return df

    bin_labels = [0, 1]
    bin_names  = {0: "No hate", 1: "Hate"}
    conf_bin_llama = make_conf_df(gt_binary, llama_bin, bin_labels, bin_names)
    conf_bin_qwen  = make_conf_df(gt_binary, qwen_bin,  bin_labels, bin_names)

    # Category confusion matrix – aligned GT hate samples, exclude missed (-1)
    cat_labels_present = sorted(set(llama_cat_yt.tolist() + qwen_cat_yt.tolist()))
    cat_names = CATEGORY_NAMES
    # For missed predictions (-1), show as "Missed" column
    llama_cat_yp_display = np.where(llama_cat_yp == -1, -1, llama_cat_yp)
    qwen_cat_yp_display  = np.where(qwen_cat_yp  == -1, -1, qwen_cat_yp)
    cat_all_pred_labels = sorted(
        set(llama_cat_yp_display.tolist() + qwen_cat_yp_display.tolist())
    )
    # Include -1 as "Missed" if any model missed samples
    cat_col_labels = cat_all_pred_labels
    cat_col_names  = {**{-1: "Missed"}, **cat_names}

    def make_conf_df_mixed(y_true, y_pred, row_labels, col_labels, label_names):
        """Confusion matrix supporting heterogeneous row/col label sets."""
        rows = [f"True: {label_names.get(l, str(l))}" for l in row_labels]
        cols = [f"Pred: {label_names.get(l, str(l))}" for l in col_labels]
        data = np.zeros((len(row_labels), len(col_labels)), dtype=int)
        row_idx = {l: i for i, l in enumerate(row_labels)}
        col_idx = {l: i for i, l in enumerate(col_labels)}
        for yt, yp in zip(y_true, y_pred):
            if yt in row_idx and yp in col_idx:
                data[row_idx[yt], col_idx[yp]] += 1
        df = pd.DataFrame(data, index=rows, columns=cols)
        df.index.name = ""
        return df

    conf_cat_llama = make_conf_df_mixed(
        llama_cat_yt, llama_cat_yp_display,
        cat_labels_present, cat_col_labels, cat_col_names,
    )
    conf_cat_qwen = make_conf_df_mixed(
        qwen_cat_yt, qwen_cat_yp_display,
        cat_labels_present, cat_col_labels, cat_col_names,
    )

    # Subcategory confusion matrix – letter-based
    all_sub_true  = [t for t in llama_sub_yt if t != "__gt__"]
    all_sub_preds = [p for p in list(llama_sub_yp) + list(qwen_sub_yp)
                     if p not in ("__missed__", "__gt__")]
    sub_row_labels = sorted(set(all_sub_true))
    sub_col_labels = sorted(set(all_sub_preds)) + (
        ["__missed__"] if "__missed__" in list(llama_sub_yp) + list(qwen_sub_yp) else []
    )
    sub_col_names  = {l: l if l != "__missed__" else "Missed" for l in sub_col_labels}
    sub_row_names  = {l: l for l in sub_row_labels}

    conf_sub_llama = make_conf_df_mixed(
        llama_sub_yt, llama_sub_yp, sub_row_labels, sub_col_labels,
        {**sub_row_names, **sub_col_names},
    )
    conf_sub_qwen = make_conf_df_mixed(
        qwen_sub_yt, qwen_sub_yp, sub_row_labels, sub_col_labels,
        {**sub_row_names, **sub_col_names},
    )

    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        summary_df.to_excel(writer,       sheet_name="Summary",                    index=False)
        binary_per_sample.to_excel(writer, sheet_name="Binary_PerSample",          index=False)
        bin_quad.to_excel(writer,          sheet_name="Binary_Quadrant",            index=False)
        conf_bin_llama.to_excel(writer,    sheet_name="Conf_Binary_LLaMA")
        conf_bin_qwen.to_excel(writer,     sheet_name="Conf_Binary_Qwen")
        errors_by_cat.to_excel(writer,     sheet_name="Binary_ErrorsByTrueLabel",   index=False)
        cat_per_sample.to_excel(writer,    sheet_name="Category_PerSample",         index=False)
        cat_quad.to_excel(writer,          sheet_name="Category_Quadrant",          index=False)
        conf_cat_llama.to_excel(writer,    sheet_name="Conf_Category_LLaMA")
        conf_cat_qwen.to_excel(writer,     sheet_name="Conf_Category_Qwen")
        sub_per_sample.to_excel(writer,    sheet_name="Subcategory_PerSample",      index=False)
        sub_quad.to_excel(writer,          sheet_name="Subcategory_Quadrant",       index=False)
        conf_sub_llama.to_excel(writer,    sheet_name="Conf_Subcategory_LLaMA")
        conf_sub_qwen.to_excel(writer,     sheet_name="Conf_Subcategory_Qwen")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
