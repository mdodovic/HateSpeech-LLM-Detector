import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import textwrap

from src.utils import (
    load_excel_full_text_dataset,
    load_excel_dataset,
    parse_category_and_subcategory,
)
from src.categories import code_to_label, code_to_label_en


# --- Helpers ---
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
    plt.style.use('ggplot')
    # Increase global font sizes for readability
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 24,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
        "figure.titlesize": 24,
    })
except Exception:
    _HAS_MPL = False

def wrap_two_lines(lbl: str, base_width: int = 20) -> str:
    """Wrap a label into at most two lines.

    Tries the given width; if result has >2 lines, increases width until it fits
    in ≤2 lines, else falls back to splitting roughly in half.
    """
    s = textwrap.fill(lbl, width=base_width)
    parts = s.split("\n")
    if len(parts) <= 2:
        return s
    for w in range(base_width + 1, base_width + 12):
        s2 = textwrap.fill(lbl, width=w)
        if s2.count("\n") <= 1:
            return s2
    mid = len(parts) // 2
    return " ".join(parts[:mid]) + "\n" + " ".join(parts[mid:])

def split_sentences(text: str) -> List[str]:
    """Split text into sentences without dropping punctuation.

    Mirrors the regex used in data/paragraph_checker_annotators.py.
    """
    s = (text or "").strip()
    if not s:
        return []
    pattern = "(?<=[.!?…])\\.*[,;\\s]*(?=(?:[A-Za-zČĆŠĐŽčćšđž\\u0400-\\u04FF0-9@#:'\"""„''()]|[\\u2600-\\u26FF\\u2700-\\u27BF\\U0001F1E6-\\U0001F1FF\\U0001F300-\\U0001F5FF\\U0001F600-\\U0001F64F\\U0001F680-\\U0001F6FF\\U0001F900-\\U0001F9FF\\U0001FA70-\\U0001FAFF]))"
    parts = re.split(pattern, s)
    return [p.strip() for p in parts if p and p.strip()]


def split_gt_entries(raw: str) -> List[str]:
    """Split GT 'Category' cell into entries by commas outside parentheses.

    Example: "(6c;0), 0, 1a" -> ["(6c;0)", "0", "1a"].
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


def parse_codes_from_entry(entry: str) -> List[str]:
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
            elif p.lower() == "u":
                codes.append("u")
    else:
        m = re.match(r"^([0-7])\s*([a-z])?$", s)
        if m:
            codes = [m.group(1) + (m.group(2) or "")]
        elif s == "0":
            codes = ["0"]
        elif s == "u":
            codes = ["u"]
        else:
            codes = ["0"]
    return codes or ["0"]


# --- Analyses ---

def analyze_full_text_dataset(excel_path: str, limit: int = -1) -> Dict:
    print("\n=== FULL-TEXT DATASET ===")
    records = load_excel_full_text_dataset(excel_path)
    if limit > 0 and len(records) > limit:
        print(f"VAŽNO: Ograničavam na prva {limit} uzorak iz razloga testiranja.")
        records = records[:limit]

    n_samples = len(records)
    print(f"Samples: {n_samples}")

    split_counts: List[int] = []
    gt_counts: List[int] = []
    hate_per_sample: List[int] = []
    mismatch = 0

    # Per-category sentence counts (exclude 0)
    per_cat_sentences: Dict[int, int] = {i: 0 for i in range(1, 8)}
    total_hate_sentences = 0
    # Three-way (mutually exclusive) counts with priority: HATE > OFFENSE > NO-HATE
    threeway = {"hate": 0, "offense": 0, "no_hate": 0}

    for i, rec in enumerate(records):
        text = (rec.get("text") or "").strip()
        sents = split_sentences(text)
        split_counts.append(len(sents))
        if len(sents) == 2:
            print(f"[SENTENCE SPLIT] Sample ID {rec.get('id', 'N/A')} (Index {i}) split into 2 sentences: {sents}")

        raw_cell = rec.get("category_raw", "")
        entries = split_gt_entries(raw_cell)
        gt_counts.append(len(entries))
        if len(entries) != len(sents):
            print(f"[MISMATCH] Sample ID {rec.get('id', 'N/A')} (Index {i}): Split into {len(sents)} sentences but has {len(entries)} GT entries.")
            mismatch += 1

        # Count per-sentence categories (exclude 0) and three-way bucket
        hate_cnt_this = 0
        for e in entries:
            codes = parse_codes_from_entry(e)
            # If any non-zero appears, this sentence is hate
            nonzero_cats = sorted({int(c[0]) for c in codes if re.match(r"^[1-7]", c)})
            if nonzero_cats:
                total_hate_sentences += 1
                for cid in nonzero_cats:
                    per_cat_sentences[cid] += 1
                threeway["hate"] += 1
                hate_cnt_this += 1
            elif any(c == "u" or c.endswith("u") for c in codes):
                threeway["offense"] += 1
            else:
                threeway["no_hate"] += 1
        hate_per_sample.append(hate_cnt_this)

    # Stats
    def stats(arr: List[int]) -> Dict[str, float]:
        if not arr:
            return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
        a = np.array(arr)
        return {
            "min": int(a.min()),
            "max": int(a.max()),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
        }

    split_stats = stats(split_counts)
    gt_stats = stats(gt_counts)

    print("Sentence count (splitter):", split_stats)
    print("Sentence count (GT entries):", gt_stats)
    print(f"Samples with sentence count mismatch: {mismatch}")

    print("\nPer-category sentence counts (hate only):")
    for cid in range(1, 8):
        print(f"  {cid}: {per_cat_sentences[cid]}")
    print(f"Total hate sentences: {total_hate_sentences}")

    print("\nThree-way sentence buckets (priority: hate > offense > no-hate):")
    print(f"  Hate speech : {threeway['hate']}")
    print(f"  Offense     : {threeway['offense']}")
    print(f"  No hate     : {threeway['no_hate']}")
    # Distribution of hate-sentence counts per paragraph
    if hate_per_sample:
        from collections import Counter
        dist = Counter(hate_per_sample)
        print("\nHate sentences per paragraph (distribution):")
        for k in sorted(dist.keys()):
            print(f"  {k}: {dist[k]} samples")

    # Plots
    try:
        if _HAS_MPL:
            out_dir = Path("results/plots")
            out_dir.mkdir(parents=True, exist_ok=True)
            # horiz_dir = out_dir / "horizontal"
            # horiz_dir.mkdir(parents=True, exist_ok=True)

            # Histograms of sentence counts (splitter vs GT)
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.hist(split_counts, bins='auto', alpha=0.6, label='Splitter count')
            ax.hist(gt_counts, bins='auto', alpha=0.6, label='GT entry count')
            # ax.set_title('Sentence Count per Sample (Full-text)')
            ax.set_xlabel('Number of sentences')
            ax.set_ylabel('Number of samples')
            fig.tight_layout()
            fig.savefig(out_dir / 'full_text_sentence_counts_hist.png', dpi=300)
            plt.close(fig)

            # Bar chart for per-category sentence counts (hate only)
            cats = list(range(1, 8))
            values = [per_cat_sentences[c] for c in cats]
            labels = [code_to_label_en(str(c)) for c in cats]
            wrapped = [textwrap.fill(lbl, width=18) for lbl in labels]

            # Pie chart (three-way)
            sizes = [threeway["no_hate"], threeway["offense"], threeway["hate"]]
            labels = ["No hate", "Offense", "Hate speech"]
            colors = ["#9EBCDA", "#F28E2B", "#E15759"]
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                autopct=lambda p: f"{p:.1f}%\n({int(round(p/100.0*sum(sizes)))})" if sum(sizes) else "0%\n(0)",
                colors=colors,
                startangle=90,
                textprops={"fontsize": 24},
            )
            ax.axis('equal')
            # ax.set_title('Full-text: No-hate vs Offense vs Hate')
            fig.tight_layout()
            fig.savefig(out_dir / 'full_text_threeway_pie.png', dpi=300)
            plt.close(fig)

            # Bar distribution: number of hate sentences per paragraph
            from collections import Counter
            dist = Counter(hate_per_sample)
            xs = sorted(dist.keys())
            ys = [dist[x] for x in xs]
            # Vertical
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.bar(xs, ys, color='#E15759')
            # ax.set_title('Hate Sentences per Paragraph — Distribution')
            ax.set_xlabel('Hate sentences in paragraph')
            ax.set_ylabel('Number of paragraphs')
            fig.tight_layout()
            fig.savefig(out_dir / 'full_text_hate_per_paragraph_bar.png', dpi=300)
            plt.close(fig)

    except Exception as e:
        print(f"[WARN] Plotting (full-text) failed: {e}")

    return {
        "n_samples": n_samples,
        "split_counts": split_counts,
        "gt_counts": gt_counts,
        "mismatch_samples": mismatch,
        "per_cat_sentences": per_cat_sentences,
        "total_hate_sentences": total_hate_sentences,
        "hate_per_sample": hate_per_sample,
        "threeway": threeway,
        "split_stats": split_stats,
        "gt_stats": gt_stats,
    }


def analyze_single_sentence_dataset(excel_path: str, limit: int = -1) -> Dict:
    print("\n=== SINGLE-SENTENCE DATASET ===")
    records = load_excel_dataset(excel_path)
    if limit > 0 and len(records) > limit:
        print(f"VAŽNO: Ograničavam na prva {limit} uzorak iz razloga testiranja.")
        records = records[:limit]

    n_samples = len(records)
    print(f"Samples: {n_samples}")

    per_cat: Dict[int, int] = {i: 0 for i in range(1, 8)}
    per_sub: Dict[str, int] = {}
    total_hate_sentences = 0
    threeway = {"hate": 0, "offense": 0, "no_hate": 0}

    for rec in records:
        # Prefer all_codes if present to detect 'u' or multiple
        all_codes = rec.get("all_codes") if isinstance(rec.get("all_codes"), list) else None
        if all_codes is None:
            cat = int(rec.get("category", 0))
            sub = str(rec.get("subcategory", "") or "")
            code_str = (str(cat) + sub) if cat != 0 else "0"
            all_codes = [code_str]
        # Three-way priority
        if any(re.match(r"^[1-7]", str(c)) for c in all_codes):
            threeway["hate"] += 1
        elif any(str(c).lower() == "u" or str(c).lower().endswith("u") for c in all_codes):
            threeway["offense"] += 1
        else:
            threeway["no_hate"] += 1

        # Category/subcategory tallies for hate only
        cat_top = rec.get("category", 0)
        sub_top = str(rec.get("subcategory", "") or "")
        if int(cat_top) != 0:
            total_hate_sentences += 1
            per_cat[int(cat_top)] += 1
            if sub_top:
                per_sub[f"{int(cat_top)}{sub_top}"] = per_sub.get(f"{int(cat_top)}{sub_top}", 0) + 1

    print("Per-category counts (hate only):")
    for cid in range(1, 8):
        print(f"  {cid}: {per_cat[cid]}")
    if per_sub:
        print("Per-subcategory counts (hate only):")
        for k in sorted(per_sub.keys()):
            print(f"  {k}: {per_sub[k]}")
    print(f"Total hate sentences: {total_hate_sentences}")

    print("\nThree-way sentence buckets (priority: hate > offense > no-hate):")
    print(f"  Hate speech : {threeway['hate']}")
    print(f"  Offense     : {threeway['offense']}")
    print(f"  No hate     : {threeway['no_hate']}")

    # Plots
    try:
        if _HAS_MPL:
            out_dir = Path("results/plots")
            out_dir.mkdir(parents=True, exist_ok=True)
            # horiz_dir = out_dir / "horizontal"
            # horiz_dir.mkdir(parents=True, exist_ok=True)

            # Bar chart per-category (horizontal for readability)
            cats = list(range(1, 8))
            values = [per_cat[c] for c in cats]
            labels = [code_to_label_en(str(c)) for c in cats]
            wrapped = [wrap_two_lines(lbl, base_width=20) for lbl in labels]

            fig, ax = plt.subplots(figsize=(14, 8))
            bars = ax.barh(range(len(cats)), values, color='#55A868')
            ax.set_yticks(range(len(cats)))
            ax.set_yticklabels(wrapped, fontsize=24)
            ax.invert_yaxis()
            ax.set_xlabel('Sentence count')
            ax.set_ylabel('Category')
            # Show counts at end of bars
            try:
                ax.bar_label(bars, fmt='%d', padding=4, fontsize=24)
            except Exception:
                pass
            # Extra left margin for long labels
            plt.subplots_adjust(left=0.30)
            fig.tight_layout()
            fig.savefig(out_dir / 'single_category_bar.png', dpi=300)
            plt.close(fig)

            # Bar chart per-subcategory (if any)
            if per_sub:
                sub_keys = sorted(per_sub.keys())
                sub_vals = [per_sub[k] for k in sub_keys]
                sub_labels = [code_to_label_en(k) for k in sub_keys]
                sub_wrapped = [wrap_two_lines(lbl, base_width=18) for lbl in sub_labels]
                # Horizontal for readability
                fig, ax = plt.subplots(figsize=(16, 9))
                bars = ax.barh(range(len(sub_keys)), sub_vals, color='#C44E52')
                ax.set_yticks(range(len(sub_keys)))
                ax.set_yticklabels(sub_wrapped, fontsize=24)
                ax.invert_yaxis()
                ax.set_xlabel('Sentence count')
                ax.set_ylabel('Subcategory')
                try:
                    ax.bar_label(bars, fmt='%d', padding=4, fontsize=24)
                except Exception:
                    pass
                plt.subplots_adjust(left=0.35)
                fig.tight_layout()
                fig.savefig(out_dir / 'single_subcategory_bar.png', dpi=300)
                plt.close(fig)

            # (Removed) Pie chart for single-sentence three-way breakdown per request
    except Exception as e:
        print(f"[WARN] Plotting (single) failed: {e}")

    return {
        "n_samples": n_samples,
        "per_cat": per_cat,
        "per_sub": per_sub,
        "total_hate_sentences": total_hate_sentences,
        "threeway": threeway,
    }


def analyze_annotator_agreement(excel_path: str) -> Dict:
    """Compare Annotator1 vs Annotator2 at sample and sentence level."""
    print("\n=== ANNOTATOR AGREEMENT (Annotator1 vs Annotator2) ===")
    df = pd.read_excel(excel_path)

    # Resolve columns
    id_col = "ID"
    text_col = "Text"
    a1_col = "Annotator1"
    a2_col = "Annotator2"
    for c in df.columns:
        cl = str(c).lower()
        if "annotator1" in cl or "anotator1" in cl:
            a1_col = c
        elif "annotator2" in cl or "anotator2" in cl:
            a2_col = c

    n_samples = len(df)
    samples_differ = 0
    sentences_total = 0
    sentences_differ = 0
    diff_sample_ids: List = []

    for _, row in df.iterrows():
        sample_id = row.get(id_col, _)
        text = str(row.get(text_col, "") or "")
        sents = split_sentences(text)
        cats1 = split_gt_entries(str(row.get(a1_col, "") or ""))
        cats2 = split_gt_entries(str(row.get(a2_col, "") or ""))

        n = min(len(sents), len(cats1), len(cats2))
        sentences_total += n

        sample_has_diff = False
        for i in range(n):
            if cats1[i].strip().lower() != cats2[i].strip().lower():
                sentences_differ += 1
                sample_has_diff = True

        # Also flag if lengths differ (beyond the min overlap)
        if len(cats1) != len(cats2):
            sample_has_diff = True

        if sample_has_diff:
            samples_differ += 1
            diff_sample_ids.append(sample_id)

    print(f"Total samples: {n_samples}")
    print(f"Samples with different annotations: {samples_differ} ({100*samples_differ/n_samples:.1f}%)")
    print(f"Total sentences compared: {sentences_total}")
    print(f"Sentences with different annotations: {sentences_differ} ({100*sentences_differ/sentences_total:.1f}%)")

    return {
        "n_samples": n_samples,
        "samples_differ": samples_differ,
        "sentences_total": sentences_total,
        "sentences_differ": sentences_differ,
        "diff_sample_ids": diff_sample_ids,
    }


# --- Default run (no CLI params) ---

DEFAULT_ANNOTATORS_DATASET_FULL_PATH = "data/access_paragraph_hate_speech_with_offenses.xlsx"
DEFAULT_FULL_PATH = "data/paragraph_hate_speech_with_offenses.xlsx"

DEFAULT_SINGLE_PATH = "data/single_sentence_hate_speech_with_offenses.xlsx"
DEFAULT_OUT_PATH = "results/test_dataset_analysis.xlsx"


def main():
    # Run full analysis on both datasets using defaults
    full_res = analyze_full_text_dataset(DEFAULT_FULL_PATH)
    single_res = analyze_single_sentence_dataset(DEFAULT_SINGLE_PATH)
    agreement_res = analyze_annotator_agreement(DEFAULT_ANNOTATORS_DATASET_FULL_PATH)

    # Always export a tidy Excel summary
    Path(DEFAULT_OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(DEFAULT_OUT_PATH, engine="openpyxl") as writer:
        if full_res:
            df_full_cat = pd.DataFrame(
                {"category": list(full_res["per_cat_sentences"].keys()), "count": list(full_res["per_cat_sentences"].values())}
            )
            df_full_stats = pd.DataFrame([
                {"stat": "split_min", "value": full_res["split_stats"]["min"]},
                {"stat": "split_mean", "value": full_res["split_stats"]["mean"]},
                {"stat": "split_median", "value": full_res["split_stats"]["median"]},
                {"stat": "split_max", "value": full_res["split_stats"]["max"]},
                {"stat": "gt_min", "value": full_res["gt_stats"]["min"]},
                {"stat": "gt_mean", "value": full_res["gt_stats"]["mean"]},
                {"stat": "gt_median", "value": full_res["gt_stats"]["median"]},
                {"stat": "gt_max", "value": full_res["gt_stats"]["max"]},
                {"stat": "mismatch_samples", "value": full_res["mismatch_samples"]},
                {"stat": "total_hate_sentences", "value": full_res["total_hate_sentences"]},
            ])
            df_full_cat.to_excel(writer, index=False, sheet_name="FullText_CatCounts")
            df_full_stats.to_excel(writer, index=False, sheet_name="FullText_Stats")
        if single_res:
            df_single_cat = pd.DataFrame(
                {"category": list(single_res["per_cat"].keys()), "count": list(single_res["per_cat"].values())}
            )
            df_single_sub = pd.DataFrame(
                {"subcategory": list(single_res["per_sub"].keys()), "count": list(single_res["per_sub"].values())}
            ) if single_res["per_sub"] else pd.DataFrame(columns=["subcategory", "count"])
            df_single_cat.to_excel(writer, index=False, sheet_name="Single_CatCounts")
            df_single_sub.to_excel(writer, index=False, sheet_name="Single_SubCounts")
    print(f"\nAnalysis summary written to: {DEFAULT_OUT_PATH}")


if __name__ == "__main__":
    main()
