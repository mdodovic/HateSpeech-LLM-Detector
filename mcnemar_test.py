"""
McNemar's paired significance test for hate speech detection models.

Tests whether two classifiers' error patterns are significantly different
on the same test set. Uses the exact binomial McNemar test (two-sided).

Pairs tested:
  ── Zero-shot ─────────────────────────────────────────────────────────────
  Zero-shot LLaMA  vs  Zero-shot Qwen          (binary / category / subcategory)
  Zero-shot LLaMA  vs  BERTić                  (binary)
  Zero-shot LLaMA  vs  General-purpose LLM     (binary)
  Zero-shot Qwen   vs  BERTić                  (binary)
  Zero-shot Qwen   vs  General-purpose LLM     (binary)

  ── Few-shot ──────────────────────────────────────────────────────────────
  Few-shot LLaMA   vs  Zero-shot LLaMA         (binary / category / subcategory)
  Few-shot Qwen    vs  Zero-shot Qwen           (binary / category / subcategory)
  Few-shot Qwen    vs  Few-shot LLaMA           (binary / category / subcategory)

NOTE: Cross-source comparisons (LLM vs BERTić, LLM vs Gemini) require that
both models were evaluated on the exact same test samples in the same row order.
A sample-count mismatch warning is printed when sizes differ.

For category/subcategory tasks the 2-stage pipeline only stores classification
predictions for samples the model itself flagged as hate speech (binary pred=1).
Different models may flag different subsets, causing a size mismatch that makes
naïve row-alignment unreliable.  Instead, load_llm_class() reconstructs a
correct/wrong indicator for ALL ground-truth hate speech samples using the
Binary sheet: samples the model missed (binary pred=0) are always wrong on
classification.  This guarantees both models are compared on the same set.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

ZERO_SHOT_PATH     = Path("results/single_sentence_per_sample.xlsx")
FEW_SHOT_PATH      = Path("results/single_sentence_few_shot_per_sample.xlsx")
BERTIC_BINARY_PATH = Path("results/bertic/bertic_binary_results_per_sample.xlsx")
FRENK_BINARY_PATH  = Path("results/bertic/bertic_frenk_binary_test_101.xlsx")
GEMINI_GT_PATH     = Path("data/single_sentence_hate_speech_no_offenses.xlsx")
GEMINI_LLM_PATH    = Path("data/single_sentence_llm_predictions.xlsx")

OUTPUT = Path("results/mcnemar_test.xlsx")

# Which prompt type to use when loading zero-shot results
ZERO_SHOT_PROMPT = "two_prompts"   # "two_prompts" | "one_prompt"

ALPHA     = 0.05    # significance level
LLAMA_TAG = "llama3"
QWEN_TAG  = "qwen3"

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders  — all return Optional[(y_true, y_pred)] as int numpy arrays
# ─────────────────────────────────────────────────────────────────────────────

def load_llm(
    path: Path,
    sheet: str,
    model: str,
    prompt_type: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load (y_true, y_pred) for a model from a per-sample LLM Excel file.

    Parameters
    ----------
    path        : path to the per-sample Excel file
    sheet       : sheet name ("Binary", "Category", or "Subcategory")
    model       : model tag, e.g. "llama3"
    prompt_type : if the sheet has a prompt_type column, filter by this value
    """
    if not path.exists():
        return None
    df = pd.read_excel(path, sheet_name=sheet)
    mask = df["model"] == model
    if prompt_type is not None and "prompt_type" in df.columns:
        mask &= df["prompt_type"] == prompt_type
    sub = df[mask].reset_index(drop=True)
    if sub.empty:
        return None
    # Keep as object array so string labels (category/subcategory) work alongside int (binary)
    return sub["y_true"].values, sub["y_pred"].values


def load_llm_class(
    path: Path,
    sheet_cls: str,
    model: str,
    prompt_type: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Return a boolean correct[] array over ALL ground-truth hate speech samples.

    For the 2-stage pipeline (binary detection → classification), the
    classification sheet only contains predictions for samples the model
    detected as hate speech.  This function aligns to all GT hate speech samples
    using the Binary sheet so both models are compared on the same set.

    Samples where the model predicted binary=0 (missed detection) are treated
    as wrong on classification (correct=False).

    Returns
    -------
    np.ndarray of shape (n_gt_hate,) dtype=bool, or None if data unavailable.
    """
    if not path.exists():
        return None

    # Binary sheet — all N test samples
    df_bin = pd.read_excel(path, sheet_name="Binary")
    mask_bin = df_bin["model"] == model
    if prompt_type is not None and "prompt_type" in df_bin.columns:
        mask_bin &= df_bin["prompt_type"] == prompt_type
    bin_sub = df_bin[mask_bin].reset_index(drop=True)
    if bin_sub.empty:
        return None

    # Classification sheet — only model-detected hate speech samples, in order
    df_cls = pd.read_excel(path, sheet_name=sheet_cls)
    mask_cls = df_cls["model"] == model
    if prompt_type is not None and "prompt_type" in df_cls.columns:
        mask_cls &= df_cls["prompt_type"] == prompt_type
    cls_sub = df_cls[mask_cls].reset_index(drop=True)

    bin_y_true = bin_sub["y_true"].astype(int).values
    bin_y_pred = bin_sub["y_pred"].astype(int).values

    # Positions (in binary sheet order) where model detected hate speech
    detected_pos = np.where(bin_y_pred == 1)[0]

    # For each detected position, was the classification correct?
    correct_at: Dict[int, bool] = {}
    for k, pos in enumerate(detected_pos):
        if k < len(cls_sub):
            correct_at[int(pos)] = (
                str(cls_sub["y_pred"].iloc[k]) == str(cls_sub["y_true"].iloc[k])
            )

    # Build correct array over all GT hate speech samples
    gt_hate_pos = np.where(bin_y_true == 1)[0]
    correct = np.array(
        [correct_at.get(int(pos), False) for pos in gt_hate_pos],
        dtype=bool,
    )
    return correct


def load_bertic_binary(mode: str = "strict") -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load BERTić binary per-sample predictions.

    mode: "strict" | "best_case"
    """
    if not BERTIC_BINARY_PATH.exists():
        return None
    df = pd.read_excel(BERTIC_BINARY_PATH, sheet_name="Predictions")
    y_true = (df["GT_label"].str.lower().str.startswith("hate")).astype(int).values
    col = "Strict_correct" if mode == "strict" else "BestCase_correct"
    correct = df[col].astype(bool).values
    y_pred = np.where(correct, y_true, 1 - y_true)
    return y_true, y_pred


def load_frenk_binary(mode: str = "strict") -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load off-the-shelf FRENK-hate binary per-sample predictions.

    mode: "strict" | "best_case"
    """
    if not FRENK_BINARY_PATH.exists():
        return None
    df = pd.read_excel(FRENK_BINARY_PATH, sheet_name="Predictions")
    y_true = (df["GT_label"].str.lower().str.startswith("hate")).astype(int).values
    col = "Strict_correct" if mode == "strict" else "BestCase_correct"
    correct = df[col].astype(bool).values
    y_pred = np.where(correct, y_true, 1 - y_true)
    return y_true, y_pred


def _parse_gemini_codes(cell) -> List[Tuple[int, str]]:
    raw = str(cell).strip() if pd.notna(cell) else ""
    if not raw or raw == "nan":
        return [(0, "")]
    cleaned = re.sub(r"[{}()]", "", raw)
    parts = [c.strip() for c in re.split(r"[,;]", cleaned) if c.strip()]
    if not parts:
        return [(0, "")]
    from src.utils import parse_category_and_subcategory
    result = []
    for p in parts:
        parsed = parse_category_and_subcategory(p)
        result.append((int(parsed["category"]), str(parsed["subcategory"])))
    return result or [(0, "")]


def load_gemini_binary(max_paragraph_id: int = 101) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load Gemini binary predictions, restricted to paragraphs with ID <= max_paragraph_id.

    Filtering to the first 101 paragraphs aligns the Gemini dataset with the
    BERTić evaluation scope (both cover the same 548 sentences from paragraphs
    1–101).  The first 47 rows also match the LLM evaluation samples exactly.
    """
    if not GEMINI_GT_PATH.exists() or not GEMINI_LLM_PATH.exists():
        return None
    gt_df  = pd.read_excel(GEMINI_GT_PATH)
    llm_df = pd.read_excel(GEMINI_LLM_PATH)
    # Filter both files to first max_paragraph_id paragraphs
    if "ID" in gt_df.columns:
        gt_df  = gt_df[gt_df["ID"] <= max_paragraph_id].reset_index(drop=True)
    if "ID" in llm_df.columns:
        llm_df = llm_df[llm_df["ID"] <= max_paragraph_id].reset_index(drop=True)
    n = min(len(gt_df), len(llm_df))
    y_true, y_pred = [], []
    for i in range(n):
        gt_codes = _parse_gemini_codes(gt_df["Category"].iloc[i])
        pr_codes = _parse_gemini_codes(llm_df["Category"].iloc[i])
        y_true.append(int(any(c != 0 for c, _ in gt_codes)))
        y_pred.append(int(any(c != 0 for c, _ in pr_codes)))
    return np.array(y_true), np.array(y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# McNemar's exact test
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar(
    yt_a: np.ndarray, yp_a: np.ndarray,
    yt_b: np.ndarray, yp_b: np.ndarray,
    label_a: str,
    label_b: str,
    task: str,
) -> Dict:
    """Run McNemar's exact binomial test on two paired classifiers.

    Builds the 2×2 contingency table from correct/incorrect per sample:
        b10 = A correct, B wrong  (A wins)
        b01 = A wrong, B correct  (B wins)

    Uses scipy.stats.binomtest(min(b10, b01), b10+b01, 0.5, two-sided).
    chi2 = (b10 - b01)^2 / (b10 + b01)  (standard McNemar chi-squared, no continuity correction).
    """
    n_a, n_b = len(yt_a), len(yt_b)
    note = ""
    if n_a != n_b:
        note = f"size mismatch ({n_a} vs {n_b}), used first {min(n_a, n_b)}"
        print(f"  [warn] {label_a} n={n_a}, {label_b} n={n_b} — {note}")
    n = min(n_a, n_b)

    correct_a = (yp_a[:n] == yt_a[:n]).astype(int)
    correct_b = (yp_b[:n] == yt_b[:n]).astype(int)

    b10 = int(np.sum((correct_a == 1) & (correct_b == 0)))
    b01 = int(np.sum((correct_a == 0) & (correct_b == 1)))
    n_discordant = b10 + b01

    if n_discordant == 0:
        return {
            "model_A": label_a, "model_B": label_b, "task": task,
            "n_samples": n, "A_wins (b10)": b10, "B_wins (b01)": b01,
            "discordant": 0, "chi2": 0.0, "p_value": 1.0, "significant": False,
            "note": "no discordant pairs",
        }

    chi2 = round((b10 - b01) ** 2 / n_discordant, 4)
    result = binomtest(min(b10, b01), n_discordant, 0.5, alternative="two-sided")

    return {
        "model_A":       label_a,
        "model_B":       label_b,
        "task":          task,
        "n_samples":     n,
        "A_wins (b10)":  b10,
        "B_wins (b01)":  b01,
        "discordant":    n_discordant,
        "chi2":          chi2,
        "p_value":       round(result.pvalue, 4),
        "significant":   result.pvalue < ALPHA,
        "note":          note,
    }


def mcnemar_correct(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    label_a: str,
    label_b: str,
    task: str,
) -> Dict:
    """McNemar's exact test when both models have been evaluated on the SAME
    set of ground-truth hate speech samples (aligned via load_llm_class).
    """
    n = len(correct_a)
    b10 = int(np.sum(correct_a & ~correct_b))
    b01 = int(np.sum(~correct_a & correct_b))
    n_discordant = b10 + b01

    if n_discordant == 0:
        return {
            "model_A": label_a, "model_B": label_b, "task": task,
            "n_samples": n, "A_wins (b10)": b10, "B_wins (b01)": b01,
            "discordant": 0, "chi2": 0.0, "p_value": 1.0, "significant": False,
            "note": "no discordant pairs",
        }

    chi2 = round((b10 - b01) ** 2 / n_discordant, 4)
    result = binomtest(min(b10, b01), n_discordant, 0.5, alternative="two-sided")
    return {
        "model_A":       label_a,
        "model_B":       label_b,
        "task":          task,
        "n_samples":     n,
        "A_wins (b10)":  b10,
        "B_wins (b01)":  b01,
        "discordant":    n_discordant,
        "chi2":          chi2,
        "p_value":       round(result.pvalue, 4),
        "significant":   result.pvalue < ALPHA,
        "note":          "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_pair(
    name_a: str, data_a: Optional[Tuple],
    name_b: str, data_b: Optional[Tuple],
    task: str,
    rows: List[Dict],
) -> None:
    """Attempt one McNemar test and append result to rows list."""
    if data_a is None:
        print(f"  [skip] {name_a} — data not available  (task={task})")
        return
    if data_b is None:
        print(f"  [skip] {name_b} — data not available  (task={task})")
        return
    row = mcnemar(*data_a, *data_b, name_a, name_b, task)
    sig_marker = "* p<0.05" if row["significant"] else ""
    print(
        f"  {name_a:28s} vs {name_b:28s}  [{task:12s}]"
        f"  n={row['n_samples']:4d}"
        f"  b10={row['A_wins (b10)']:4d}  b01={row['B_wins (b01)']:4d}"
        f"  chi2={row['chi2']:.4f}  p={row['p_value']:.4f}  {sig_marker}"
    )
    rows.append(row)


def run_pair_class(
    name_a: str, data_a: Optional[np.ndarray],
    name_b: str, data_b: Optional[np.ndarray],
    task: str,
    rows: List[Dict],
) -> None:
    """McNemar for a classification task using aligned correct[] arrays.

    Both arrays must cover the same ground-truth hate speech samples so that
    samples missed by one model (binary pred=0) are correctly penalised.
    """
    if data_a is None:
        print(f"  [skip] {name_a} — data not available  (task={task})")
        return
    if data_b is None:
        print(f"  [skip] {name_b} — data not available  (task={task})")
        return
    if len(data_a) != len(data_b):
        print(
            f"  [skip] {name_a} vs {name_b} [{task}] "
            f"— GT hate-speech count mismatch ({len(data_a)} vs {len(data_b)})"
        )
        return
    row = mcnemar_correct(data_a, data_b, name_a, name_b, task)
    sig_marker = "* p<0.05" if row["significant"] else ""
    print(
        f"  {name_a:28s} vs {name_b:28s}  [{task:12s}]"
        f"  n={row['n_samples']:4d}"
        f"  b10={row['A_wins (b10)']:4d}  b01={row['B_wins (b01)']:4d}"
        f"  chi2={row['chi2']:.4f}  p={row['p_value']:.4f}  {sig_marker}"
    )
    rows.append(row)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    workspace = Path(__file__).parent
    rows: List[Dict] = []

    print(f"McNemar's Test  |  exact binomial  |  alpha={ALPHA}")
    print(f"Zero-shot prompt type: {ZERO_SHOT_PROMPT}")
    print("=" * 95)

    # ── Zero-shot LLM comparisons ──────────────────────────────────────────
    print("\n── Zero-shot comparisons ────────────────────────────────────────────────────────────────")

    bertic_bin = load_bertic_binary(mode="strict")
    frenk_bin  = load_frenk_binary(mode="strict")
    gemini_bin = load_gemini_binary()

    for task, sheet in [("binary", "Binary"), ("category", "Category"), ("subcategory", "Subcategory")]:
        if task == "binary":
            zs_llama = load_llm(ZERO_SHOT_PATH, sheet, LLAMA_TAG, ZERO_SHOT_PROMPT)
            zs_qwen  = load_llm(ZERO_SHOT_PATH, sheet, QWEN_TAG,  ZERO_SHOT_PROMPT)
            run_pair("Zero-shot LLaMA", zs_llama, "Zero-shot Qwen",       zs_qwen,    task, rows)
            run_pair("Zero-shot LLaMA", zs_llama, "BERTić",               bertic_bin, task, rows)
            run_pair("Zero-shot LLaMA", zs_llama, "FRENK-hate",           frenk_bin,  task, rows)
            run_pair("Zero-shot LLaMA", zs_llama, "General-purpose LLM",  gemini_bin, task, rows)
            run_pair("Zero-shot Qwen",  zs_qwen,  "BERTić",               bertic_bin, task, rows)
            run_pair("Zero-shot Qwen",  zs_qwen,  "FRENK-hate",           frenk_bin,  task, rows)
            run_pair("Zero-shot Qwen",  zs_qwen,  "General-purpose LLM",  gemini_bin, task, rows)
            # Full-sample cross-source: BERTić vs General-purpose LLM (both on same 548 sentences)
            run_pair("BERTić",          bertic_bin, "General-purpose LLM",  gemini_bin, task, rows)
            run_pair("BERTić",          bertic_bin, "FRENK-hate",           frenk_bin,  task, rows)
            run_pair("FRENK-hate",       frenk_bin,  "General-purpose LLM",  gemini_bin, task, rows)
        else:
            # Classification: align on ALL GT hate speech samples via binary sheet
            zs_llama_c = load_llm_class(ZERO_SHOT_PATH, sheet, LLAMA_TAG, ZERO_SHOT_PROMPT)
            zs_qwen_c  = load_llm_class(ZERO_SHOT_PATH, sheet, QWEN_TAG,  ZERO_SHOT_PROMPT)
            run_pair_class("Zero-shot LLaMA", zs_llama_c, "Zero-shot Qwen", zs_qwen_c, task, rows)

    # ── Few-shot comparisons ───────────────────────────────────────────────
    print("\n── Few-shot comparisons ─────────────────────────────────────────────────────────────────")

    for task, sheet in [("binary", "Binary"), ("category", "Category"), ("subcategory", "Subcategory")]:
        if task == "binary":
            zs_llama = load_llm(ZERO_SHOT_PATH, sheet, LLAMA_TAG, ZERO_SHOT_PROMPT)
            zs_qwen  = load_llm(ZERO_SHOT_PATH, sheet, QWEN_TAG,  ZERO_SHOT_PROMPT)
            fs_llama = load_llm(FEW_SHOT_PATH,  sheet, LLAMA_TAG)
            fs_qwen  = load_llm(FEW_SHOT_PATH,  sheet, QWEN_TAG)
            run_pair("Few-shot LLaMA", fs_llama, "Zero-shot LLaMA", zs_llama, task, rows)
            run_pair("Few-shot Qwen",  fs_qwen,  "Zero-shot Qwen",  zs_qwen,  task, rows)
            run_pair("Few-shot Qwen",  fs_qwen,  "Few-shot LLaMA",  fs_llama, task, rows)
        else:
            # Classification: align on ALL GT hate speech samples via binary sheet
            zs_llama_c = load_llm_class(ZERO_SHOT_PATH, sheet, LLAMA_TAG, ZERO_SHOT_PROMPT)
            zs_qwen_c  = load_llm_class(ZERO_SHOT_PATH, sheet, QWEN_TAG,  ZERO_SHOT_PROMPT)
            fs_llama_c = load_llm_class(FEW_SHOT_PATH,  sheet, LLAMA_TAG)
            fs_qwen_c  = load_llm_class(FEW_SHOT_PATH,  sheet, QWEN_TAG)
            run_pair_class("Few-shot LLaMA", fs_llama_c, "Zero-shot LLaMA", zs_llama_c, task, rows)
            run_pair_class("Few-shot Qwen",  fs_qwen_c,  "Zero-shot Qwen",  zs_qwen_c,  task, rows)
            run_pair_class("Few-shot Qwen",  fs_qwen_c,  "Few-shot LLaMA",  fs_llama_c, task, rows)

    # ── Save & summarise ───────────────────────────────────────────────────
    print("\n" + "=" * 95)
    if not rows:
        print("No results — no data was available for any pair.")
        return

    result_df = pd.DataFrame(rows)
    output = workspace / OUTPUT if not OUTPUT.is_absolute() else OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(output), engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="McNemar")
    print(f"Results saved to: {output}")

    # Highlight significant pairs
    sig = result_df[result_df["significant"] == True]
    if not sig.empty:
        print(f"\nSignificant pairs (p < {ALPHA}):")
        for _, r in sig.iterrows():
            print(f"  {r['model_A']} vs {r['model_B']}  [{r['task']}]  p={r['p_value']}")
    else:
        print(f"\nNo pairs reached significance at alpha={ALPHA}.")


if __name__ == "__main__":
    main()
