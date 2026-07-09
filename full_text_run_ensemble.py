"""
Ensemble majority voting for full-text inputs, evaluated per sentence.

- Loads a full-text dataset where 'Category' contains comma-separated codes
  aligned to sentences (e.g., "0, 0, 1c, 6a, 0").
- For each model (from models/models.json or data/models.json), calls
  classify_all_sentences once per record using the combined prompt
  (src/prompts/classify_full_all.txt).
- Performs majority voting per sentence across models:
    has_hate_speech: majority True/False
    category: majority among non-zero categories from models that predicted hate; tie -> lowest id
    subcategory: majority letter among models predicting chosen category; tie -> alphabetical
  If majority vote is no-hate -> category=0, subcategory="".
- Writes results to results/full_text_ensemble.xlsx

No command-line interface; adjust constants below if needed.
"""

from typing import List, Dict, Tuple, Optional
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from src.llm_detector import LLMDetector
from src.categories import get_category_prompt, code_to_label
from src.utils import build_model_tags, load_excel_full_text_dataset, parse_category_and_subcategory
from src.evaluation import HateSpeechEvaluator

# Constants
DATASET_PATH = "data/paragraph_hate_speech.xlsx"
RESULTS_XLSX = "results/full_text_ensemble.xlsx"
MODEL_SUBSET: List[str] = []  # Empty -> use all from JSON
SEED = 42


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


def _majority(values: List):
    if not values:
        return None, 0
    freq: Dict = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    max_count = max(freq.values())
    winners = [v for v, c in freq.items() if c == max_count]
    chosen = sorted(winners)[0]
    return chosen, max_count


def _split_gt_entries(raw: str) -> List[str]:
    """Split by commas that are outside parentheses to align per-sentence GT.
    Example: "(6c;0), 0, 1a" -> ["(6c;0)", "0", "1a"]
    """
    s = str(raw or "")
    entries: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in s:
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == ',' and depth == 0:
            token = ''.join(buf).strip()
            if token:
                entries.append(token)
            buf = []
        else:
            buf.append(ch)
    token = ''.join(buf).strip()
    if token:
        entries.append(token)
    return entries


def _parse_gt_codes_for_sentence(entry: str) -> List[str]:
    """Parse a single per-sentence GT entry into a list of codes.
    Supports forms like "0", "3", "3b" or parenthesized multiple: "(6c;0)".
    Returns a list like ["0"], ["3"], ["3b"], or ["6c","0"].
    """
    s = str(entry or "").strip().lower()
    if not s:
        return ["0"]
    codes: List[str] = []
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1]
        parts = [p.strip() for p in inner.split(";") if p.strip()]
        for p in parts:
            m = re.match(r"^([0-7])\s*([a-z])?$", p)
            if m:
                codes.append(m.group(1) + (m.group(2) or ""))
            elif p == "0":
                codes.append("0")
    else:
        m = re.match(r"^([0-7])\s*([a-z])?$", s)
        if m:
            codes = [m.group(1) + (m.group(2) or "")]
        elif s == "0":
            codes = ["0"]
        else:
            codes = ["0"]
    return codes or ["0"]


def _split_sentences_fallback(text: str) -> List[str]:
    """Fallback sentence splitter mirroring data scripts.

    Splits on whitespace after sentence-ending punctuation while retaining punctuation.
    Supports Latin/Cyrillic letters and common emoji blocks.
    """
    s = (text or "").strip()
    if not s:
        return []
    pattern = (
        r"(?<=[.!?…])\s+(?=(?:[A-Za-zČĆŠĐŽčćšđž\u0400-\u04FF0-9'\"“”„‘’()]|"
        r"[\u2600-\u26FF\u2700-\u27BF\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF]))"
    )
    parts = re.split(pattern, s)
    return [p.strip() for p in parts if p and p.strip()]


def perform_ensemble(excel_path: str = DATASET_PATH, models: List[str] = MODEL_SUBSET, debug: int = -1, output_path: Optional[str] = RESULTS_XLSX, seed: int = SEED):
    print("=" * 70)
    print("Full-text Ensemble Run (per sentence)")
    print("=" * 70)

    print("Učitavam dataset iz Excel fajla…")
    records = load_excel_full_text_dataset(excel_path)
    if debug > 0 and len(records) > debug:
        print(f"VAŽNO: Ograničavam na prva {debug} uzorak iz razloga testiranja.")
        records = records[:debug]
    model_tags = build_model_tags(models)
    categories_prompt = get_category_prompt()

    detectors: Dict[str, LLMDetector] = {name: LLMDetector(tag, default_seed=seed) for name, tag in model_tags.items()}
    print(f"Models in ensemble ({len(detectors)}): {list(detectors.keys())}")

    # Per-sentence metric vectors
    y_true_bin_sent: List[bool] = []
    y_pred_bin_sent: List[bool] = []
    y_true_cat_sent: List[int] = []
    y_pred_cat_sent: List[int] = []
    y_true_sub_sent: List[str] = []
    y_pred_sub_sent: List[str] = []

    total_time_s = 0.0
    n_calls = 0

    per_sentence_rows: List[Dict] = []

    try:
        for ridx, rec in enumerate(records, start=1):
            text = (rec.get("text") or "").strip()
            if not text:
                continue

            raw_cell = str(rec.get("category_raw", "") or "")
            gt_entries = _split_gt_entries(raw_cell)

            # Fetch predictions from all models in parallel (one call per model per record)
            def _predict(mname: str, det: LLMDetector):
                t0 = time.perf_counter()
                res = det.classify_all_sentences(text, categories_prompt)
                elapsed = time.perf_counter() - t0
                return (
                    mname,
                    elapsed,
                    res.get("sentences", []) if isinstance(res, dict) else [],
                    str(res.get("raw", ""))[:4000],
                )

            per_model_sentlists: Dict[str, List[Dict]] = {}
            per_model_raw: Dict[str, str] = {}
            with ThreadPoolExecutor(max_workers=len(detectors)) as ex:
                futures = {ex.submit(_predict, mname, det): mname for mname, det in detectors.items()}
                for fut in as_completed(futures):
                    mname = futures[fut]
                    try:
                        name, elapsed, sent_list, raw_out = fut.result()
                        n_calls += 1
                        total_time_s += elapsed
                        per_model_sentlists[name] = sent_list
                        per_model_raw[name] = raw_out
                    except Exception as e:
                        print(f"    [WARN] Model '{mname}' failed: {e}")
                        per_model_sentlists[mname] = []
                        per_model_raw[mname] = ""

            # Determine reference sentence list for alignment
            model_names = list(detectors.keys())
            lengths = {m: len(per_model_sentlists.get(m, [])) for m in model_names}
            ref_model = None
            ref_len = 0
            for m, l in lengths.items():
                if l > ref_len:
                    ref_len = l
                    ref_model = m

            ref_sentences: List[Dict] = per_model_sentlists.get(ref_model, []) if ref_len > 0 else []
            # If no model produced sentences, fallback to simple splitter
            if ref_len == 0:
                fb_sents = _split_sentences_fallback(text)
                if fb_sents:
                    ref_sentences = [{"sentence": s, "category": 0, "subcategory": ""} for s in fb_sents]
                    ref_len = len(ref_sentences)
                else:
                    print(f"[WARN] Skipping record {ridx}: empty predictions and no fallback sentences.")
                    continue

            # Ensure GT has entries; if missing/empty, assume zeros per sentence
            if not gt_entries:
                gt_entries = ["0"] * ref_len

            short_text = (text[:120] + "…") if len(text) > 120 else text
            print(f"\n[{ridx}] {short_text}")
            if any(len(per_model_sentlists.get(m, [])) != len(gt_entries) for m in model_names):
                print("    Upozorenje: razlika u broju rečenica; koristi se referentno poravnanje.")

            # Iterate per sentence index over reference length and perform voting
            for sidx in range(ref_len):
                # Collect per-model predictions for this sentence
                pm_has: List[bool] = []
                pm_cat: List[int] = []
                pm_sub: List[str] = []
                sent_text = ""
                for mname in model_names:
                    preds = per_model_sentlists.get(mname, [])
                    if sidx < len(preds) and preds:
                        p = preds[sidx]
                        c = int(p.get("category", 0))
                        s = _normalize_subcat(str(p.get("subcategory", "") or "")) if c != 0 else ""
                        pm_has.append(c != 0)
                        pm_cat.append(c)
                        pm_sub.append(s)
                        if not sent_text:
                            sent_text = str(p.get("sentence", ""))
                    else:
                        pm_has.append(False)
                        pm_cat.append(0)
                        pm_sub.append("")
                if not sent_text:
                    # Use reference sentence text
                    if sidx < len(ref_sentences):
                        sent_text = str(ref_sentences[sidx].get("sentence", ""))
                    else:
                        sent_text = ""

                # Majority voting
                ens_has, _ = _majority(pm_has)
                if not ens_has:
                    ens_cat = 0
                    ens_sub = ""
                else:
                    cats_for_vote = [c for h, c in zip(pm_has, pm_cat) if h and c != 0]
                    ens_cat, _ = _majority(cats_for_vote) if cats_for_vote else (0, 0)
                    if ens_cat == 0:
                        ens_sub = ""
                    else:
                        subs_for_vote = [s for h, c, s in zip(pm_has, pm_cat, pm_sub) if h and c == ens_cat and s]
                        ens_sub, _ = _majority(subs_for_vote) if subs_for_vote else ("", 0)

                # Parse GT for this sentence (default to 0 when beyond provided GT entries)
                if sidx < len(gt_entries):
                    gt_codes = _parse_gt_codes_for_sentence(gt_entries[sidx])
                else:
                    gt_codes = ["0"]
                gt_has = any(c != "0" for c in gt_codes)

                # Metrics accumulation (match semantics mirrors full_text_run.py)
                y_true_bin_sent.append(gt_has)
                y_pred_bin_sent.append(bool(ens_cat != 0))

                gt_base_nums = {int(c[0]) for c in gt_codes if re.match(r"^[0-7]", c)}
                cat_match = (ens_cat == 0 and not gt_has) or (ens_cat != 0 and ens_cat in gt_base_nums)
                y_pred_cat_sent.append(ens_cat)
                y_true_cat_sent.append(ens_cat if cat_match else -1)

                if gt_has and ens_cat != 0:
                    gt_exact = {c for c in gt_codes if re.match(r"^[0-7][a-z]$", c)}
                    gt_no_sub = {c for c in gt_codes if re.match(r"^[0-7]$", c)}
                    sub_match = False
                    if ens_sub and f"{ens_cat}{ens_sub}" in gt_exact:
                        sub_match = True
                    elif not ens_sub and str(ens_cat) in gt_no_sub:
                        sub_match = True
                    elif str(ens_cat) in {c[0] for c in gt_exact}:
                        sub_match = False
                    y_pred_sub_sent.append(ens_sub or "")
                    y_true_sub_sent.append((ens_sub or "") if sub_match else "#")

                # Detailed logging row
                row: Dict = {
                    "record_index": ridx,
                    "sentence_index": sidx + 1,
                    "sentence": sent_text,
                    "gt_codes": ";".join(gt_codes) if gt_codes else "0",
                    "ensemble_has": bool(ens_cat != 0),
                    "ensemble_category": ens_cat,
                    "ensemble_subcategory": ens_sub,
                }
                for i, mname in enumerate(model_names):
                    row[f"{mname}_has"] = pm_has[i]
                    row[f"{mname}_category"] = pm_cat[i]
                    row[f"{mname}_subcategory"] = pm_sub[i]
                per_sentence_rows.append(row)

                print(f"    [S{sidx+1}] {sent_text}")
                print(f"         GT: {row['gt_codes']}")
                print("         Per-Model predictions:")
                for i, mname in enumerate(model_names):
                    code_str = (str(pm_cat[i]) + (pm_sub[i] or "")) if pm_cat[i] != 0 else "0"
                    label = code_to_label(code_str)
                    print(f"           {mname:<10} hate={pm_has[i]} cat={pm_cat[i]} sub={pm_sub[i] or ''}  |  {code_str}: {label}")
                ens_code = (str(ens_cat) + (ens_sub or "")) if ens_cat != 0 else "0"
                ens_label = code_to_label(ens_code)
                print(f"         ENS: hate={bool(ens_cat != 0)} cat={ens_cat} sub={ens_sub or ''}  |  {ens_code}: {ens_label}")
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user; computing metrics on processed samples.")

    evaluator = HateSpeechEvaluator()
    bin_metrics = evaluator.evaluate_binary_classification(y_true_bin_sent, y_pred_bin_sent)
    cat_metrics = evaluator.evaluate_multiclass_classification(y_true_cat_sent, y_pred_cat_sent)
    sub_metrics = evaluator.evaluate_multiclass_classification(y_true_sub_sent, y_pred_sub_sent) if y_true_sub_sent else {"accuracy": 0.0}

    avg_ms = (total_time_s / n_calls * 1000.0) if n_calls else 0.0
    timing = {
        "ensemble_total_s": total_time_s,
        "ensemble_calls": n_calls,
        "ensemble_avg_ms": avg_ms,
    }

    print("\n=== ENSEMBLE METRICS (per sentence) ===")
    print(f"Binary accuracy:      {bin_metrics['accuracy']:.4f}")
    print(f"Category accuracy:    {cat_metrics['accuracy']:.4f}")
    if 'accuracy' in sub_metrics:
        print(f"Subcategory accuracy: {sub_metrics['accuracy']:.4f}")
    print(f"Timing: total={total_time_s:.3f}s, calls={n_calls}, avg={avg_ms:.1f} ms/call")

    # Excel outputs
    hs_eval = HateSpeechEvaluator()
    hs_eval.save_results("full_text_ensemble", {
        "binary_metrics": bin_metrics,
        "category_metrics": cat_metrics,
        "subcategory_metrics": sub_metrics,
        "timing": timing,
    })
    df_summary = hs_eval.to_dataframe()
    df_detailed = pd.DataFrame(per_sentence_rows)

    # Build standardised per-sample sheet for bootstrap CI
    ps_rows = []
    for row in per_sentence_rows:
        # gt_codes is a semicolon-joined string like "0", "1a;0", etc.
        gt_codes_str = str(row.get("gt_codes", "0") or "0")
        gt_codes_list = [c.strip() for c in gt_codes_str.split(";") if c.strip()]
        gt_has  = any(c != "0" for c in gt_codes_list)
        gt_cat  = next((int(c[0]) for c in gt_codes_list if c != "0" and c[0].isdigit()), 0)
        gt_sub  = next((c[1:] for c in gt_codes_list if len(c) > 1 and c[0].isdigit() and c[1:].isalpha()), "")
        ens_has = bool(row.get("ensemble_has", False))
        ens_cat = int(row.get("ensemble_category", 0))
        ens_sub = str(row.get("ensemble_subcategory", "") or "")
        ps_rows.append({"model": "ensemble", "task": "binary",   "y_true": int(gt_has),  "y_pred": int(ens_has)})
        ps_rows.append({"model": "ensemble", "task": "category", "y_true": gt_cat,        "y_pred": ens_cat})
        if gt_cat != 0:
            ps_rows.append({"model": "ensemble", "task": "subcategory", "y_true": gt_sub, "y_pred": ens_sub})
    df_ps = pd.DataFrame(ps_rows)

    if output_path is not None:
        Path("results").mkdir(exist_ok=True)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_summary.to_excel(writer, index=False, sheet_name="Metrics")
            df_detailed.to_excel(writer, index=False, sheet_name="PerSentence")
            for task in ("binary", "category", "subcategory"):
                subset = df_ps[df_ps["task"] == task].drop(columns="task").reset_index(drop=True)
                if not subset.empty:
                    subset.to_excel(writer, index=False, sheet_name=task.capitalize())
        print(f"\nEnsemble results written to: {output_path}")

    per_sample = {}
    for task, suffix in (
        ("binary", "binary"),
        ("category", "category"),
        ("subcategory", "subcategory"),
    ):
        subset = df_ps[df_ps["task"] == task] if not df_ps.empty else pd.DataFrame()
        per_sample[f"y_true_{suffix}"] = subset["y_true"].tolist() if not subset.empty else []
        per_sample[f"y_pred_{suffix}"] = subset["y_pred"].tolist() if not subset.empty else []

    return {
        "binary_metrics": bin_metrics,
        "category_metrics": cat_metrics,
        "subcategory_metrics": sub_metrics,
        "timing": timing,
        "per_sample": per_sample,
    }


if __name__ == "__main__":
    perform_ensemble(seed=SEED)
