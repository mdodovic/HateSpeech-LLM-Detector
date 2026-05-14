"""
Bootstrap Confidence Intervals for Hate Speech Detection Metrics.

Reads per-sample prediction results from Excel files and computes 95% bootstrap
confidence intervals for all relevant metrics.

Supported result file types (auto-detected from columns):
  - binary       : GT_label, Predicted_label, Strict_correct, BestCase_correct
  - ternary      : GT_label, Predicted_label, Strict_correct, BestCase_correct, Valid_classes
  - categories   : GT_label, Predicted_label, Correct
  - subcategories: GT_label_code, Predicted_label_code, Correct

Edit the FILES list and N_BOOTSTRAP / OUTPUT constants at the bottom to configure runs.
"""

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI Configuration 
# ─────────────────────────────────────────────────────────────────────────────

N_BOOTSTRAP = 10_000
CI          = 0.95
SEED        = 42
OUTPUT      = Path("results/bootstrap_ci_all.xlsx")

# Gemini evaluation paths (two separate files, aligned by row order)
GEMINI_GT_PATH  = Path("data/single_sentence_hate_speech_no_offenses.xlsx")
GEMINI_LLM_PATH = Path("data/single_sentence_llm_predictions.xlsx")


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap core
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Return (point_estimate, lower_bound, upper_bound) for a metric.

    Parameters
    ----------
    y_true       : ground-truth labels
    y_pred       : predicted labels
    metric_fn    : callable(y_true, y_pred) -> float
    n_bootstrap  : number of bootstrap resamples
    ci           : confidence level (default 0.95 -> 95% CI)
    seed         : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)

    scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1.0 - alpha) * 100))
    return float(point), lower, upper


# ─────────────────────────────────────────────────────────────────────────────
# File-type detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_file_type(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if "y_true" in cols and "y_pred" in cols:
        return "llm_persample"
    if "GT_label_code" in cols and "Predicted_label_code" in cols:
        return "subcategories"
    if "Valid_classes" in cols:
        return "ternary"
    if "Strict_correct" in cols and "BestCase_correct" in cols:
        return "binary"
    if "GT_label" in cols and "Predicted_label" in cols:
        return "categories"
    raise ValueError(
        f"Cannot detect file type from columns: {sorted(cols)}. "
        "Expected one of: binary, ternary, categories, subcategories, llm_persample."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-file-type metric suites
# ─────────────────────────────────────────────────────────────────────────────

def _label_encode(series: pd.Series) -> np.ndarray:
    """Convert string labels to integer codes via pandas Categorical."""
    return pd.Categorical(series).codes.astype(int)


def _build_metric_fns(average: str, labels=None) -> Dict[str, Callable]:
    """Return a dict of metric functions for classification tasks."""
    kw = dict(zero_division=0)
    if average != "binary" and labels is not None:
        kw["labels"] = labels

    return {
        "accuracy":           lambda yt, yp: accuracy_score(yt, yp),
        f"f1_{average}":      lambda yt, yp: f1_score(yt, yp, average=average, **kw),
        f"precision_{average}": lambda yt, yp: precision_score(yt, yp, average=average, **kw),
        f"recall_{average}":  lambda yt, yp: recall_score(yt, yp, average=average, **kw),
    }


def run_binary(df: pd.DataFrame, n_bootstrap: int, seed: int) -> List[Dict]:
    """Binary: strict mode and best-case mode."""
    rows = []
    for mode, correct_col in [("strict", "Strict_correct"), ("best_case", "BestCase_correct")]:
        # Re-derive y_true / y_pred from correct/incorrect flag + GT
        # Strict: predicted = GT when correct else flip
        gt = df["GT_label"].map(lambda x: 1 if str(x).lower().startswith("hate") else 0).values
        # Use the Correct boolean directly to derive y_pred
        correct = df[correct_col].astype(bool).values
        # y_pred: same as gt when correct, flipped when wrong (binary only)
        y_pred = np.where(correct, gt, 1 - gt)
        y_true = gt

        metric_fns = {
            "accuracy":   lambda yt, yp: accuracy_score(yt, yp),
            "f1":         lambda yt, yp: f1_score(yt, yp, average="binary", zero_division=0),
            "precision":  lambda yt, yp: precision_score(yt, yp, average="binary", zero_division=0),
            "recall":     lambda yt, yp: recall_score(yt, yp, average="binary", zero_division=0),
        }
        for metric_name, fn in metric_fns.items():
            point, lo, hi = bootstrap_ci(y_true, y_pred, fn, n_bootstrap=n_bootstrap, seed=seed)
            rows.append({"mode": mode, "metric": metric_name, "value": point, "CI_lower": lo, "CI_upper": hi})
    return rows


def run_ternary(df: pd.DataFrame, n_bootstrap: int, seed: int) -> List[Dict]:
    """Ternary (No offense / Hate speech / Offense): strict and best-case."""
    rows = []
    for mode, correct_col in [("strict", "Strict_correct"), ("best_case", "BestCase_correct")]:
        y_true_enc = _label_encode(df["GT_label"])
        correct = df[correct_col].astype(bool).values
        # Reconstruct y_pred: when correct use GT encoding; when wrong we only know
        # "not GT", so use a placeholder label that differs from GT.
        # For accuracy/F1 purposes the bootstrap will resample the (gt, pred) pairs.
        # We keep the actual predicted labels from the Predicted_label column.
        pred_enc = _label_encode(df["Predicted_label"])

        all_labels = sorted(set(y_true_enc) | set(pred_enc))
        for avg in ("macro", "micro", "weighted"):
            kw = dict(zero_division=0, labels=all_labels)
            metric_fns = {
                "accuracy":          lambda yt, yp: accuracy_score(yt, yp),
                f"f1_{avg}":         lambda yt, yp: f1_score(yt, yp, average=avg, **kw),
                f"precision_{avg}":  lambda yt, yp: precision_score(yt, yp, average=avg, **kw),
                f"recall_{avg}":     lambda yt, yp: recall_score(yt, yp, average=avg, **kw),
            }
            for metric_name, fn in metric_fns.items():
                if metric_name == "accuracy" and avg != "macro":
                    continue  # accuracy is the same regardless of avg; compute once
                point, lo, hi = bootstrap_ci(y_true_enc, pred_enc, fn, n_bootstrap=n_bootstrap, seed=seed)
                rows.append({"mode": mode, "average": avg, "metric": metric_name, "value": point, "CI_lower": lo, "CI_upper": hi})

        # accuracy (once per mode)
        point, lo, hi = bootstrap_ci(y_true_enc, pred_enc, accuracy_score, n_bootstrap=n_bootstrap, seed=seed)
        rows.append({"mode": mode, "average": "-", "metric": "accuracy", "value": point, "CI_lower": lo, "CI_upper": hi})
    return rows


def run_categories(df: pd.DataFrame, n_bootstrap: int, seed: int) -> List[Dict]:
    """Multiclass category predictions (8 classes)."""
    y_true_enc = _label_encode(df["GT_label"])
    y_pred_enc = _label_encode(df["Predicted_label"])
    all_labels = sorted(set(y_true_enc) | set(y_pred_enc))

    rows = []
    for avg in ("macro", "micro", "weighted"):
        kw = dict(zero_division=0, labels=all_labels)
        metric_fns = {
            f"f1_{avg}":         lambda yt, yp, a=avg: f1_score(yt, yp, average=a, **kw),
            f"precision_{avg}":  lambda yt, yp, a=avg: precision_score(yt, yp, average=a, **kw),
            f"recall_{avg}":     lambda yt, yp, a=avg: recall_score(yt, yp, average=a, **kw),
        }
        for metric_name, fn in metric_fns.items():
            point, lo, hi = bootstrap_ci(y_true_enc, y_pred_enc, fn, n_bootstrap=n_bootstrap, seed=seed)
            rows.append({"average": avg, "metric": metric_name, "value": point, "CI_lower": lo, "CI_upper": hi})

    # accuracy (independent of averaging)
    point, lo, hi = bootstrap_ci(y_true_enc, y_pred_enc, accuracy_score, n_bootstrap=n_bootstrap, seed=seed)
    rows.append({"average": "-", "metric": "accuracy", "value": point, "CI_lower": lo, "CI_upper": hi})
    return rows


def run_subcategories(df: pd.DataFrame, n_bootstrap: int, seed: int) -> List[Dict]:
    """Subcategory predictions (fine-grained codes like 1a, 3b, …)."""
    y_true_enc = _label_encode(df["GT_label_code"].astype(str))
    y_pred_enc = _label_encode(df["Predicted_label_code"].astype(str))
    all_labels = sorted(set(y_true_enc) | set(y_pred_enc))

    rows = []
    for avg in ("macro", "micro", "weighted"):
        kw = dict(zero_division=0, labels=all_labels)
        metric_fns = {
            f"f1_{avg}":         lambda yt, yp, a=avg: f1_score(yt, yp, average=a, **kw),
            f"precision_{avg}":  lambda yt, yp, a=avg: precision_score(yt, yp, average=a, **kw),
            f"recall_{avg}":     lambda yt, yp, a=avg: recall_score(yt, yp, average=a, **kw),
        }
        for metric_name, fn in metric_fns.items():
            point, lo, hi = bootstrap_ci(y_true_enc, y_pred_enc, fn, n_bootstrap=n_bootstrap, seed=seed)
            rows.append({"average": avg, "metric": metric_name, "value": point, "CI_lower": lo, "CI_upper": hi})

    point, lo, hi = bootstrap_ci(y_true_enc, y_pred_enc, accuracy_score, n_bootstrap=n_bootstrap, seed=seed)
    rows.append({"average": "-", "metric": "accuracy", "value": point, "CI_lower": lo, "CI_upper": hi})
    return rows


def run_llm_persample(df: pd.DataFrame, n_bootstrap: int, seed: int) -> List[Dict]:
    """LLM per-sample sheet: columns model, (prompt_type), y_true, y_pred.

    Groups by all available grouping columns (model, prompt_type) and computes
    CIs per group.  Works for binary (0/1), multiclass integer, or string labels.
    """
    group_cols = [c for c in ("model", "prompt_type") if c in df.columns]
    rows = []

    groups = df.groupby(group_cols) if group_cols else [((), df)]
    for group_key, group in groups:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        key_dict = dict(zip(group_cols, group_key))

        y_true_enc = _label_encode(pd.Series(group["y_true"].values))
        y_pred_enc = _label_encode(pd.Series(group["y_pred"].values))
        all_labels = sorted(set(y_true_enc) | set(y_pred_enc))
        n_classes = len(all_labels)

        averages = ("macro", "micro", "weighted")
        if n_classes == 2:
            averages = ("binary",) + averages

        for avg in averages:
            kw = dict(zero_division=0)
            if avg != "binary":
                kw["labels"] = all_labels
            metric_fns = {
                f"f1_{avg}":         lambda yt, yp, a=avg, k=kw: f1_score(yt, yp, average=a, **k),
                f"precision_{avg}":  lambda yt, yp, a=avg, k=kw: precision_score(yt, yp, average=a, **k),
                f"recall_{avg}":     lambda yt, yp, a=avg, k=kw: recall_score(yt, yp, average=a, **k),
            }
            for metric_name, fn in metric_fns.items():
                point, lo, hi = bootstrap_ci(y_true_enc, y_pred_enc, fn, n_bootstrap=n_bootstrap, seed=seed)
                rows.append({**key_dict, "average": avg, "metric": metric_name, "value": point, "CI_lower": lo, "CI_upper": hi})

        point, lo, hi = bootstrap_ci(y_true_enc, y_pred_enc, accuracy_score, n_bootstrap=n_bootstrap, seed=seed)
        rows.append({**key_dict, "average": "-", "metric": "accuracy", "value": point, "CI_lower": lo, "CI_upper": hi})

    return rows


def run_gemini(gt_path: Path, llm_path: Path, n_bootstrap: int, seed: int) -> List[Dict]:
    """Gemini evaluation: load GT + LLM prediction files, compute CIs for
    binary, category, and subcategory tasks using the same parsing logic
    as gemini_results.py.

    GT file  : has a 'Category' column with raw annotation codes (e.g. '1a', '0', '3b;4c').
    LLM file : same format — LLM-predicted category codes.
    """
    import re
    from src.utils import parse_category_and_subcategory

    def _parse_codes(cell):
        raw = str(cell).strip() if pd.notna(cell) else ""
        if not raw or raw == "nan":
            return [(0, "")]
        cleaned = raw.replace('{', '').replace('}', '').replace('(', '').replace(')', '')
        codes = [c.strip() for c in re.split(r"[,;]", cleaned) if c.strip()]
        if not codes:
            return [(0, "")]
        result = []
        for c in codes:
            p = parse_category_and_subcategory(c)
            result.append((int(p["category"]), str(p["subcategory"])))
        return result or [(0, "")]

    def _best_match(gt_codes, pr_codes):
        best, best_score = None, -1
        gt_hate = [(c, s) for c, s in gt_codes if c != 0] or gt_codes
        for gt_cat, gt_sub in gt_hate:
            for pr_cat, pr_sub in pr_codes:
                gs = f"{gt_cat}{gt_sub}" if gt_sub else str(gt_cat)
                ps = f"{pr_cat}{pr_sub}" if pr_sub else str(pr_cat)
                score = 2 if gs == ps else (1 if gt_cat == pr_cat else 0)
                if score > best_score:
                    best_score = score
                    best = (gt_cat, gt_sub, pr_cat, pr_sub)
                    if score == 2:
                        return best
        return best

    gt_df  = pd.read_excel(gt_path)
    llm_df = pd.read_excel(llm_path)
    if len(gt_df) != len(llm_df):
        print(f"  [warn] Gemini row mismatch GT={len(gt_df)} LLM={len(llm_df)}")

    y_true_bin, y_pred_bin = [], []
    y_true_cat, y_pred_cat = [], []
    y_true_sub, y_pred_sub = [], []

    for i in range(min(len(gt_df), len(llm_df))):
        gt_codes = _parse_codes(gt_df["Category"].iloc[i])
        pr_codes = _parse_codes(llm_df["Category"].iloc[i])

        gt_hate = any(c != 0 for c, _ in gt_codes)
        pr_hate = any(c != 0 for c, _ in pr_codes)
        y_true_bin.append(int(gt_hate))
        y_pred_bin.append(int(pr_hate))

        if gt_hate:
            gt_cat, gt_sub, pr_cat, pr_sub = _best_match(gt_codes, pr_codes)
            y_true_cat.append(gt_cat)
            y_pred_cat.append(pr_cat)
            y_true_sub.append(f"{gt_cat}{gt_sub}" if gt_sub else str(gt_cat))
            y_pred_sub.append(f"{pr_cat}{pr_sub}" if pr_sub else str(pr_cat))

    rows = []

    # ── Binary ──────────────────────────────────────────────────────────────
    yt_bin = np.array(y_true_bin)
    yp_bin = np.array(y_pred_bin)
    bin_fns = {
        "accuracy":  lambda yt, yp: accuracy_score(yt, yp),
        "f1":        lambda yt, yp: f1_score(yt, yp, average="binary", zero_division=0),
        "precision": lambda yt, yp: precision_score(yt, yp, average="binary", zero_division=0),
        "recall":    lambda yt, yp: recall_score(yt, yp, average="binary", zero_division=0),
    }
    for metric_name, fn in bin_fns.items():
        pt, lo, hi = bootstrap_ci(yt_bin, yp_bin, fn, n_bootstrap=n_bootstrap, seed=seed)
        rows.append({"task": "binary", "average": "binary", "metric": metric_name, "value": pt, "CI_lower": lo, "CI_upper": hi})

    # ── Category ────────────────────────────────────────────────────────────
    yt_cat = _label_encode(pd.Series(y_true_cat))
    yp_cat = _label_encode(pd.Series(y_pred_cat))
    all_cat = sorted(set(yt_cat) | set(yp_cat))
    for avg in ("macro", "micro", "weighted"):
        kw = dict(zero_division=0, labels=all_cat)
        cat_fns = {
            f"f1_{avg}":         lambda yt, yp, a=avg: f1_score(yt, yp, average=a, **kw),
            f"precision_{avg}":  lambda yt, yp, a=avg: precision_score(yt, yp, average=a, **kw),
            f"recall_{avg}":     lambda yt, yp, a=avg: recall_score(yt, yp, average=a, **kw),
        }
        for metric_name, fn in cat_fns.items():
            pt, lo, hi = bootstrap_ci(yt_cat, yp_cat, fn, n_bootstrap=n_bootstrap, seed=seed)
            rows.append({"task": "category", "average": avg, "metric": metric_name, "value": pt, "CI_lower": lo, "CI_upper": hi})
    pt, lo, hi = bootstrap_ci(yt_cat, yp_cat, accuracy_score, n_bootstrap=n_bootstrap, seed=seed)
    rows.append({"task": "category", "average": "-", "metric": "accuracy", "value": pt, "CI_lower": lo, "CI_upper": hi})

    # ── Subcategory ──────────────────────────────────────────────────────────
    yt_sub = _label_encode(pd.Series(y_true_sub))
    yp_sub = _label_encode(pd.Series(y_pred_sub))
    all_sub = sorted(set(yt_sub) | set(yp_sub))
    for avg in ("macro", "micro", "weighted"):
        kw = dict(zero_division=0, labels=all_sub)
        sub_fns = {
            f"f1_{avg}":         lambda yt, yp, a=avg: f1_score(yt, yp, average=a, **kw),
            f"precision_{avg}":  lambda yt, yp, a=avg: precision_score(yt, yp, average=a, **kw),
            f"recall_{avg}":     lambda yt, yp, a=avg: recall_score(yt, yp, average=a, **kw),
        }
        for metric_name, fn in sub_fns.items():
            pt, lo, hi = bootstrap_ci(yt_sub, yp_sub, fn, n_bootstrap=n_bootstrap, seed=seed)
            rows.append({"task": "subcategory", "average": avg, "metric": metric_name, "value": pt, "CI_lower": lo, "CI_upper": hi})
    pt, lo, hi = bootstrap_ci(yt_sub, yp_sub, accuracy_score, n_bootstrap=n_bootstrap, seed=seed)
    rows.append({"task": "subcategory", "average": "-", "metric": "accuracy", "value": pt, "CI_lower": lo, "CI_upper": hi})

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def print_ci_table(rows: List[Dict], file_label: str, file_type: str) -> None:
    df = pd.DataFrame(rows)

    # Column order: put grouping columns first
    group_cols = [c for c in ("task", "mode", "model", "prompt_type", "average") if c in df.columns]
    value_cols = ["metric", "value", "CI_lower", "CI_upper"]
    df = df[group_cols + value_cols]

    df["CI_width"] = (df["CI_upper"] - df["CI_lower"]).map("{:.3f}".format)
    df["value"]    = df["value"].map("{:.3f}".format)
    df["CI_lower"] = df["CI_lower"].map("{:.3f}".format)
    df["CI_upper"] = df["CI_upper"].map("{:.3f}".format)
    df["95% CI"]   = "[" + df["CI_lower"] + ", " + df["CI_upper"] + "]"
    display_cols = group_cols + ["metric", "value", "95% CI", "CI_width"]

    print(f"\n{'='*70}")
    print(f"  Bootstrap CIs  |  {file_label}  |  type={file_type}")
    print(f"{'='*70}")
    print(df[display_cols].to_string(index=False))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Single-file processor
# ─────────────────────────────────────────────────────────────────────────────

def process_file(
    path: Path,
    n_bootstrap: int,
    seed: int,
    output: Path | None,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name or 0)
    file_type = detect_file_type(df)
    label = f"{path.stem}[{sheet_name}]" if sheet_name else path.stem
    print(f"\nProcessing: {label}  (detected type: {file_type}, n={len(df)})")

    dispatch = {
        "binary":        run_binary,
        "ternary":       run_ternary,
        "categories":    run_categories,
        "subcategories": run_subcategories,
        "llm_persample": run_llm_persample,
    }
    rows = dispatch[file_type](df, n_bootstrap=n_bootstrap, seed=seed)
    result_df = pd.DataFrame(rows)

    print_ci_table(rows, label, file_type)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(str(output), engine="openpyxl") as writer:
            result_df.to_excel(writer, index=False, sheet_name="bootstrap_ci")
        print(f"  Saved to: {output}")

    return result_df


# Per-sample prediction files to process.
# Each entry: (path, sheet_name|None, output|None)
#   sheet_name: None = first sheet; set for multi-sheet per-sample files.
#   output: None = skip per-file save (all results still go to OUTPUT below).
FILES: List[Tuple[Path, str | None, Path | None]] = [
    # ── BERTić: binary ──────────────────────────────────────────────────────
    (Path("results/bertic/bertic_binary_results_per_sample.xlsx"),       "Predictions", None),
    # ── BERTić: categories ──────────────────────────────────────────────────
    (Path("results/bertic/bertic_categories_results_per_sample.xlsx"),   "Predictions", None),
    # ── BERTić: subcategories ───────────────────────────────────────────────
    (Path("results/bertic/bertic_subcategories_results_per_sample.xlsx"), "Predictions", None),
    # # ── LLM: single sentence (generated by single_sentence_run.py) ─────────
    # (Path("results/single_sentence_per_sample.xlsx"), "Binary",      None),
    # (Path("results/single_sentence_per_sample.xlsx"), "Category",    None),
    # (Path("results/single_sentence_per_sample.xlsx"), "Subcategory", None),
    # ── LLM: single sentence few-shot (single_sentence_few_shot_run.py) ────
    (Path("results/single_sentence_few_shot_per_sample.xlsx"), "Binary",      None),
    (Path("results/single_sentence_few_shot_per_sample.xlsx"), "Category",    None),
    (Path("results/single_sentence_few_shot_per_sample.xlsx"), "Subcategory", None),
    # # ── LLM: full text (generated by full_text_run.py) ──────────────────────
    # (Path("results/full_text_per_sample.xlsx"), "Binary",      None),
    # (Path("results/full_text_per_sample.xlsx"), "Category",    None),
    # (Path("results/full_text_per_sample.xlsx"), "Subcategory", None),
    # # ── LLM: single sentence ensemble (single_sentence_run_ensemble.py) ────
    # (Path("results/single_sentence_ensemble.xlsx"), "Binary",      None),
    # (Path("results/single_sentence_ensemble.xlsx"), "Category",    None),
    # (Path("results/single_sentence_ensemble.xlsx"), "Subcategory", None),
    # # ── LLM: full text ensemble (full_text_run_ensemble.py) ─────────────────
    # (Path("results/full_text_ensemble.xlsx"), "Binary",      None),
    # (Path("results/full_text_ensemble.xlsx"), "Category",    None),
    # (Path("results/full_text_ensemble.xlsx"), "Subcategory", None),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    workspace = Path(__file__).parent
    print(f"Bootstrap CIs  |  n_bootstrap={N_BOOTSTRAP}  ci={CI*100:.0f}%  seed={SEED}")

    all_results: Dict[str, pd.DataFrame] = {}
    for rel_path, sheet_name, per_file_out in FILES:
        path = rel_path if rel_path.is_absolute() else workspace / rel_path
        if not path.exists():
            print(f"  [skip] not found: {path.name}{'['+sheet_name+']' if sheet_name else ''}")
            continue
        try:
            out = (workspace / per_file_out) if per_file_out else None
            result_df = process_file(path, n_bootstrap=N_BOOTSTRAP, seed=SEED, output=out, sheet_name=sheet_name)
            key = f"{path.stem}_{sheet_name}" if sheet_name else path.stem
            all_results[key] = result_df
        except Exception as exc:
            print(f"  [error] {path.name}: {exc}")

    # ── Gemini (two-file alignment) ─────────────────────────────────────────
    gt_path  = workspace / GEMINI_GT_PATH  if not GEMINI_GT_PATH.is_absolute()  else GEMINI_GT_PATH
    llm_path = workspace / GEMINI_LLM_PATH if not GEMINI_LLM_PATH.is_absolute() else GEMINI_LLM_PATH
    if gt_path.exists() and llm_path.exists():
        print(f"\nProcessing: gemini  (GT={gt_path.name}, LLM={llm_path.name})")
        try:
            gemini_rows = run_gemini(gt_path, llm_path, n_bootstrap=N_BOOTSTRAP, seed=SEED)
            gemini_df = pd.DataFrame(gemini_rows)
            print_ci_table(gemini_rows, "gemini", "gemini")
            all_results["gemini"] = gemini_df
        except Exception as exc:
            print(f"  [error] gemini: {exc}")
    else:
        missing = [p.name for p in (gt_path, llm_path) if not p.exists()]
        print(f"  [skip] gemini — not found: {', '.join(missing)}")

    if OUTPUT is not None and all_results:
        out = workspace / OUTPUT if not OUTPUT.is_absolute() else OUTPUT
        out.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(str(out), engine="openpyxl") as writer:
            for sheet, df in all_results.items():
                df.to_excel(writer, index=False, sheet_name=sheet[:31])
        print(f"\nAll results saved to: {out}")


if __name__ == "__main__":
    main()
