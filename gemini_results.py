"""
Compare LLM (Gemini) predictions against ground-truth annotations
on the single-sentence hate speech dataset.

Metrics:
  - Binary detection (hate vs. no-hate): accuracy, F1
  - Top-level category (0-7):            accuracy, F1 (micro)
  - Subcategory (e.g. 1a, 6c):           accuracy, F1 (micro)
"""

import re
import argparse
import pandas as pd
from typing import List, Tuple

from src.utils import parse_category_and_subcategory
from src.evaluation import HateSpeechEvaluator
from src.categories import HATE_SPEECH_CATEGORIES, HATE_SPEECH_CATEGORIES_EN


def _parse_all_codes(cell) -> List[Tuple[int, str]]:
    """Parse a Category cell into a list of (category, subcategory) tuples.

    For multi-code cells like '1b,0' or '(6c;4b)', returns all parsed codes.
    Returns at least [(0, '')] for empty/no-hate cells.
    """
    raw = str(cell).strip() if pd.notna(cell) else ""
    if not raw or raw == "nan":
        return [(0, "")]

    # Strip outer parens/braces, split on , and ;
    cleaned = raw.replace('{', '').replace('}', '').replace('(', '').replace(')', '')
    codes = [c.strip() for c in re.split(r"[,;]", cleaned) if c.strip()]
    if not codes:
        return [(0, "")]

    result = []
    for c in codes:
        p = parse_category_and_subcategory(c)
        result.append((int(p["category"]), str(p["subcategory"])))
    return result if result else [(0, "")]


def _best_match(gt_codes: List[Tuple[int, str]], pr_codes: List[Tuple[int, str]]) -> Tuple[int, str, int, str]:
    """Find the best matching pair across all GT x LLM codes.

    Priority: exact subcategory match > same top-level category > no match.
    Returns (gt_cat, gt_sub, pr_cat, pr_sub) for the best pair.
    """
    best = None
    best_score = -1  # 0=no match, 1=category match, 2=subcategory match

    # Only consider non-zero GT codes (hate codes)
    gt_hate_codes = [(c, s) for c, s in gt_codes if c != 0]
    if not gt_hate_codes:
        gt_hate_codes = gt_codes

    for gt_cat, gt_sub in gt_hate_codes:
        for pr_cat, pr_sub in pr_codes:
            gt_sub_code = f"{gt_cat}{gt_sub}" if gt_sub else str(gt_cat)
            pr_sub_code = f"{pr_cat}{pr_sub}" if pr_sub else str(pr_cat)

            if gt_sub_code == pr_sub_code:
                score = 2
            elif gt_cat == pr_cat:
                score = 1
            else:
                score = 0

            if score > best_score:
                best_score = score
                best = (gt_cat, gt_sub, pr_cat, pr_sub)
                if score == 2:
                    return best  # can't do better

    return best


def load_and_align(gt_path: str, llm_path: str) -> pd.DataFrame:
    """Load both Excel files and return a merged DataFrame aligned by row index."""
    gt_df = pd.read_excel(gt_path)
    llm_df = pd.read_excel(llm_path)

    if len(gt_df) != len(llm_df):
        print(f"WARNING: row count mismatch — GT={len(gt_df)}, LLM={len(llm_df)}")

    df = pd.DataFrame({
        "ID": gt_df["ID"],
        "Text": gt_df["Text"],
        "GT_Category": gt_df["Category"],
        "LLM_Category": llm_df["Category"],
    })
    return df


def evaluate(df: pd.DataFrame):
    """Run binary, category, and subcategory evaluation and print results."""
    y_true_bin: List[bool] = []
    y_pred_bin: List[bool] = []
    y_true_cat: List[int] = []
    y_pred_cat: List[int] = []
    y_true_sub: List[str] = []
    y_pred_sub: List[str] = []

    for _, row in df.iterrows():
        gt_codes = _parse_all_codes(row["GT_Category"])
        pr_codes = _parse_all_codes(row["LLM_Category"])

        gt_hate = any(c != 0 for c, _ in gt_codes)
        pr_hate = any(c != 0 for c, _ in pr_codes)

        y_true_bin.append(gt_hate)
        y_pred_bin.append(pr_hate)

        # Category & Subcategory: all samples where GT has hate (gt != 0)
        if gt_hate:
            gt_cat, gt_sub, pr_cat, pr_sub = _best_match(gt_codes, pr_codes)

            y_true_cat.append(gt_cat)
            y_pred_cat.append(pr_cat)

            gt_sub_code = f"{gt_cat}{gt_sub}" if gt_sub else str(gt_cat)
            # If category doesn't match, subcategory is automatically wrong
            if gt_cat == pr_cat:
                pr_sub_code = f"{pr_cat}{pr_sub}" if pr_sub else str(pr_cat)
            else:
                pr_sub_code = f"{pr_cat}{pr_sub}" if pr_sub else str(pr_cat)
            y_true_sub.append(gt_sub_code)
            y_pred_sub.append(pr_sub_code)

    evaluator = HateSpeechEvaluator()

    binary_metrics = evaluator.evaluate_binary_classification(y_true_bin, y_pred_bin)
    category_metrics = evaluator.evaluate_multiclass_classification(y_true_cat, y_pred_cat)
    subcategory_metrics = evaluator.evaluate_multiclass_classification(y_true_sub, y_pred_sub)

    # Build category report excluding 0
    hate_categories = {k: v for k, v in HATE_SPEECH_CATEGORIES_EN.items() if k != 0}

    print("=" * 50)
    print("Gemini LLM vs Ground Truth — Single Sentence")
    print("=" * 50)

    print(f"\nTotal samples: {len(df)}")
    print(f"GT has hate (category & subcategory eval): {len(y_true_cat)}")

    print("\n--- Task 1: Binary Detection (hate / no-hate) ---")
    print(f"  Accuracy: {binary_metrics['accuracy']:.4f}")
    print(f"  F1:       {binary_metrics['f1']:.4f}")

    print("\n--- Task 2: Category Classification (1-7, GT hate only) ---")
    print(f"  Accuracy: {category_metrics['accuracy']:.4f}")
    print(f"  F1 micro: {category_metrics['f1']:.4f}")

    print("\n--- Task 3: Subcategory Classification (GT hate only, category mismatch = fail) ---")
    print(f"  Accuracy: {subcategory_metrics['accuracy']:.4f}")
    print(f"  F1 micro: {subcategory_metrics['f1']:.4f}")

    # Detailed classification report for categories (excluding 0)
    print("\n--- Category Classification Report (GT hate only) ---")
    report = evaluator.generate_classification_report(y_true_cat, y_pred_cat, hate_categories)
    print(report)

    return {
        "binary_metrics": binary_metrics,
        "category_metrics": category_metrics,
        "subcategory_metrics": subcategory_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemini LLM predictions against ground truth")
    parser.add_argument("--gt", default="data/single_sentence_hate_speech_no_offenses.xlsx",
                        help="Ground-truth Excel file")
    parser.add_argument("--llm", default="data/single_sentence_llm_predictions.xlsx",
                        help="LLM predictions Excel file")
    args = parser.parse_args()

    df = load_and_align(args.gt, args.llm)
    evaluate(df)


if __name__ == "__main__":
    main()
