"""
Fixed-parameter ensemble majority voting for single-sentence hate speech detection.

Reads the small Excel dataset, queries ALL models from models/models.json (or data/models.json),
and produces a majority-vote result written to `results/single_sentence_ensemble.xlsx` (new file).

Voting rules:
  has_hate_speech: majority True/False
  category: majority among non-zero categories from models that predicted hate; tie -> lowest id
  subcategory: majority letter among models predicting chosen category; tie -> alphabetical
If majority vote is no-hate -> category=0, subcategory="".

No command-line interface; adjust constants below if needed.
"""

from typing import List, Dict, Tuple
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from src.llm_detector import LLMDetector
from src.categories import get_category_prompt
from src.utils import load_excel_dataset, build_model_tags
from src.evaluation import HateSpeechEvaluator

# Constants
DATASET_PATH = "data/single_sentence_hate_speech.xlsx"
RESULTS_XLSX = "results/single_sentence_ensemble.xlsx"
USE_ONE_PROMPT = False  # Set False to use two-prompt (slower)
MODEL_SUBSET: List[str] = []  # Empty -> use all from JSON


def _normalize_subcat(raw_sub: str) -> str:
    if not isinstance(raw_sub, str):
        return ""
    s = raw_sub.strip().lower()
    if not s:
        return ""
    m = re.match(r"^([0-7])\s*([a-z])$", s)
    if m:
        return m.group(2)
    if re.match(r"^[a-z]$", s):
        return s
    m2 = re.match(r"^([0-7])([a-z])$", s)
    if m2:
        return m2.group(2)
    return ""


def _majority(values: List) -> Tuple:
    if not values:
        return None, 0
    freq: Dict = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    max_count = max(freq.values())
    winners = [v for v, c in freq.items() if c == max_count]
    chosen = sorted(winners)[0]
    return chosen, max_count


def perform_ensemble(debug: int = 0):
    print("=" * 70)
    print("Fixed Ensemble Run")
    print("=" * 70)

    records = load_excel_dataset(DATASET_PATH)

    # Ograniči na jedan pasus (za sada)
    if debug > 0 and len(records) > debug:
        print(f"VAŽNO: Ograničavam na prva {debug} uzorak iz razloga testiranja.")
        records = records[:debug]

    model_tags = build_model_tags(MODEL_SUBSET)
    categories_prompt = get_category_prompt()

    detectors: Dict[str, LLMDetector] = {name: LLMDetector(tag) for name, tag in model_tags.items()}
    print(f"Models in ensemble ({len(detectors)}): {list(detectors.keys())}")

    y_true_bin: List[bool] = []
    y_pred_bin: List[bool] = []
    y_true_cat: List[int] = []
    y_pred_cat: List[int] = []
    y_true_sub: List[str] = []
    y_pred_sub: List[str] = []

    total_time_s = 0.0
    n_calls = 0

    try:
        per_record_rows: List[Dict] = []  # For detailed per-model logging

        for idx, rec in enumerate(records, start=1):
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            gt_has = bool(rec.get("has_hate_speech", False))
            gt_cat = int(rec.get("category", 0))
            gt_sub = str(rec.get("subcategory", ""))

            per_model_has: List[bool] = []
            per_model_cat: List[int] = []
            per_model_sub: List[str] = []
            per_model_raw: List[str] = []

            def _predict(mname: str, det: LLMDetector):
                t0 = time.perf_counter()
                if USE_ONE_PROMPT:
                    res = det.detect_and_categorize(text, categories_prompt)
                    elapsed = time.perf_counter() - t0
                    return (
                        mname,
                        elapsed,
                        bool(res.get("has_hate_speech", False)),
                        int(res.get("category", 0)),
                        _normalize_subcat(str(res.get("subcategory", ""))) if int(res.get("category", 0)) != 0 else "",
                        str(res.get("raw", ""))[:4000],  # truncate extremely long reasoning
                    )
                # two-prompt path
                has = det.detect_hate_speech_binary(text)
                elapsed = time.perf_counter() - t0
                if has:
                    t1 = time.perf_counter()
                    codes = det.categorize_hate_speech(text, categories_prompt)
                    elapsed += (time.perf_counter() - t1)
                    # choose primary (first non-zero if present)
                    primary_cat = 0
                    primary_sub = ""
                    for code in (codes or []):
                        m = re.match(r"^([0-7])([a-z])?$", code.strip().lower())
                        if m:
                            c = int(m.group(1))
                            s = m.group(2) or ""
                            if c != 0:
                                primary_cat = c
                                primary_sub = s
                                break
                    if primary_cat == 0 and codes:
                        m = re.match(r"^([0-7])([a-z])?$", codes[0].strip().lower())
                        if m:
                            primary_cat = int(m.group(1))
                            primary_sub = m.group(2) or ""

                    return (mname, elapsed, True, int(primary_cat), _normalize_subcat(primary_sub) if int(primary_cat) != 0 else "", "")
                return (mname, elapsed, False, 0, "", "")

            # Parallel execution across models
            with ThreadPoolExecutor(max_workers=len(detectors)) as ex:
                futures = {ex.submit(_predict, mname, det): mname for mname, det in detectors.items()}
                for fut in as_completed(futures):
                    mname = futures[fut]
                    try:
                        name, elapsed, has, cat, sub, raw_out = fut.result()
                        n_calls += 1
                        total_time_s += elapsed
                    except Exception as e:
                        print(f"    [WARN] Model '{mname}' failed: {e}")
                        has, cat, sub, raw_out = False, 0, "", ""
                    per_model_has.append(has)
                    per_model_cat.append(cat)
                    per_model_sub.append(sub)
                    per_model_raw.append(raw_out)
                    # Add per-model fields to row dict
                    # We'll create row dict after ensemble voting decision.

            # Build row scaffold for logging
            row: Dict = {
                "index": idx,
                "text": text,
                "gt_has_hate": gt_has,
                "gt_category": gt_cat,
                "gt_subcategory": gt_sub,
            }
            for m_i, mname in enumerate(detectors.keys()):
                row[f"{mname}_has"] = per_model_has[m_i] if m_i < len(per_model_has) else False
                row[f"{mname}_category"] = per_model_cat[m_i] if m_i < len(per_model_cat) else 0
                row[f"{mname}_subcategory"] = per_model_sub[m_i] if m_i < len(per_model_sub) else ""
                row[f"{mname}_raw"] = per_model_raw[m_i] if m_i < len(per_model_raw) else ""

            ensemble_has, _ = _majority(per_model_has)
            if not ensemble_has:
                ensemble_cat = 0
                ensemble_sub = ""
            else:
                cats_for_vote = [c for h, c in zip(per_model_has, per_model_cat) if h and c != 0]
                ensemble_cat, _ = _majority(cats_for_vote) if cats_for_vote else (0, 0)
                if ensemble_cat == 0:
                    ensemble_sub = ""
                else:
                    subs_for_vote = [s for h, c, s in zip(per_model_has, per_model_cat, per_model_sub) if h and c == ensemble_cat and s]
                    ensemble_sub, _ = _majority(subs_for_vote) if subs_for_vote else ("", 0)

            y_true_bin.append(gt_has)
            y_pred_bin.append(bool(ensemble_has))
            y_true_cat.append(gt_cat)
            y_pred_cat.append(ensemble_cat)
            if gt_cat != 0:
                y_true_sub.append(gt_sub or "")
                y_pred_sub.append(ensemble_sub or "")

            # Add ensemble results to row
            row["ensemble_has"] = bool(ensemble_has)
            row["ensemble_category"] = ensemble_cat
            row["ensemble_subcategory"] = ensemble_sub
            per_record_rows.append(row)

            short_text = (text[:100] + "…") if len(text) > 100 else text
            print(f"[{idx}] {short_text}")
            print(f"  GroundTruth: hate={gt_has} cat={gt_cat} sub={gt_sub or ''}")
            print("  Per-Model predictions:")
            for m_i, mname in enumerate(detectors.keys()):
                pm_has = per_model_has[m_i]
                pm_cat = per_model_cat[m_i]
                pm_sub = per_model_sub[m_i]
                print(f"    {mname:<10} hate={pm_has} cat={pm_cat} sub={pm_sub or ''}")
            print(f"  Ensemble    : hate={ensemble_has} cat={ensemble_cat} sub={ensemble_sub or ''}")
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user; computing metrics on processed samples.")

    evaluator = HateSpeechEvaluator()
    bin_metrics = evaluator.evaluate_binary_classification(y_true_bin, y_pred_bin)
    cat_metrics = evaluator.evaluate_multiclass_classification(y_true_cat, y_pred_cat)
    sub_metrics = evaluator.evaluate_multiclass_classification(y_true_sub, y_pred_sub) if y_true_sub else {"accuracy": 0.0}

    avg_ms = (total_time_s / n_calls * 1000.0) if n_calls else 0.0
    timing = {
        "ensemble_total_s": total_time_s,
        "ensemble_calls": n_calls,
        "ensemble_avg_ms": avg_ms,
    }

    print("\n=== ENSEMBLE METRICS ===")
    print(f"Binary accuracy:      {bin_metrics['accuracy']:.4f}")
    print(f"Category accuracy:    {cat_metrics['accuracy']:.4f}")
    if 'accuracy' in sub_metrics:
        print(f"Subcategory accuracy: {sub_metrics['accuracy']:.4f}")
    print(f"Timing: total={total_time_s:.3f}s, calls={n_calls}, avg={avg_ms:.1f} ms/call")

    # Prepare dataframe and write to Excel (new file each run)
    hs_eval = HateSpeechEvaluator()
    hs_eval.save_results("ensemble", {
        "binary_metrics": bin_metrics,
        "category_metrics": cat_metrics,
        "subcategory_metrics": sub_metrics,
        "timing": timing,
    })
    df_summary = hs_eval.to_dataframe()
    df_detailed = pd.DataFrame(per_record_rows)

    # Build standardised per-sample sheet for bootstrap CI
    ps_rows = []
    for row in per_record_rows:
        gt_has  = bool(row.get("gt_has_hate", False))
        ens_has = bool(row.get("ensemble_has", False))
        gt_cat  = int(row.get("gt_category", 0))
        ens_cat = int(row.get("ensemble_category", 0))
        gt_sub  = str(row.get("gt_subcategory", "") or "")
        ens_sub = str(row.get("ensemble_subcategory", "") or "")
        ps_rows.append({"model": "ensemble", "task": "binary",   "y_true": int(gt_has),  "y_pred": int(ens_has)})
        ps_rows.append({"model": "ensemble", "task": "category", "y_true": gt_cat,        "y_pred": ens_cat})
        if gt_cat != 0:
            ps_rows.append({"model": "ensemble", "task": "subcategory", "y_true": gt_sub, "y_pred": ens_sub})
    df_ps = pd.DataFrame(ps_rows)

    Path("results").mkdir(exist_ok=True)
    with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="Metrics")
        df_detailed.to_excel(writer, index=False, sheet_name="PerModel")
        for task in ("binary", "category", "subcategory"):
            subset = df_ps[df_ps["task"] == task].drop(columns="task").reset_index(drop=True)
            if not subset.empty:
                subset.to_excel(writer, index=False, sheet_name=task.capitalize())
    print(f"\nEnsemble results written to: {RESULTS_XLSX}")


if __name__ == "__main__":
    perform_ensemble()
