"""
This is cumulative run for task 1 (yes/no hatespeech) and 3 (categorization).

First way is to combine 2 different prompts by asking if there is hatespeech and then if does what category it belongs to.

Second way is to use single prompt to do both tasks at once.

Task 2 is extraction of the hate speech sentences extraction from long texts and token coverage, which is not included here, because we sentence-by-sentence approach.

"""

import re
import pandas as pd
from sklearn.metrics import accuracy_score
from pathlib import Path
from typing import  List, Dict

from src.llm_detector import LLMDetector
from src.categories import get_category_prompt, HATE_SPEECH_CATEGORIES
from src.utils import load_excel_dataset, build_model_tags, parse_category_and_subcategory, load_excel_full_text_dataset
from src.evaluation import HateSpeechEvaluator

def one_prompt_evaluation_model_on_records(model_tag: str, records: List[Dict]) -> Dict:
    """Evaluacija koristeći jedan kombinovani prompt (detekcija + kategorija)."""
    print(f"Uzoraka za obradu: {len(records)}")
    print("Inicijalizujem LLM detektor…")
    detector = LLMDetector(model_tag)
    categories_prompt = get_category_prompt()

    y_true_bin: List[bool] = []
    y_pred_bin: List[bool] = []
    y_true_cat: List[int] = []
    y_pred_cat: List[int] = []
    y_true_sub: List[str] = []
    y_pred_sub: List[str] = []
    # Extra metrics for full-text sheets with multiple GT categories
    category_any_hits = 0
    category_any_total = 0
    subcategory_any_hits = 0
    subcategory_any_total = 0

    for idx, rec in enumerate(records, start=1):
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        gt_has_hate = bool(rec.get("has_hate_speech", False))
        gt_cat = int(rec.get("category", 0))
        gt_subcat = str(rec.get("subcategory", ""))
        gt_all_cats = rec.get("all_categories") or []
        gt_all_subs = rec.get("all_subcategories") or []

        # Classify ALL sentences and also extract hate-speech-only for coverage
        all_cls = detector.classify_all_sentences(text, categories_prompt)
        ext = detector.extract_hate_speech_sentences(text, categories_prompt)
        has_hate = bool(ext.get("has_hate_speech", False))
        # Primary predicted category/subcategory = first non-zero from all sentences
        pred_cat = 0
        pred_sub = ""
        for e in all_cls.get("sentences", []):
            if int(e.get("category", 0)) != 0:
                pred_cat = int(e.get("category", 0))
                pred_sub = str(e.get("subcategory", "") or "")
                break

        y_true_bin.append(gt_has_hate)
        y_pred_bin.append(has_hate)
        y_true_cat.append(gt_cat)
        y_pred_cat.append(pred_cat)
        # normalize predicted subcategory to a single letter when possible
        m = re.match(r"^\s*([0-7])\s*([a-z])\s*$", pred_sub, flags=re.IGNORECASE)
        pred_sub_letter = m.group(2).lower() if m else (pred_sub.lower() if re.match(r"^[a-z]$", pred_sub, flags=re.IGNORECASE) else "")
        if gt_cat != 0:
            y_true_sub.append(gt_subcat or "")
            y_pred_sub.append(pred_sub_letter or "")

        # Any-of-list metrics (use the full list if present)
        # Define allowed category set: if GT has non-zero cats, use those; else {0}
        nonzero_gt_cats = [c for c in gt_all_cats if isinstance(c, int) and c != 0]
        allowed_cats = set(nonzero_gt_cats) if nonzero_gt_cats else {0}
        if pred_cat in allowed_cats:
            category_any_hits += 1
        category_any_total += 1

        # For subcategory: allowed pairs (cat, sub) for non-zero entries; if no hate then (0, "")
        allowed_pairs = []
        if nonzero_gt_cats:
            for c, s in zip(gt_all_cats, gt_all_subs):
                if isinstance(c, int) and c != 0:
                    allowed_pairs.append((c, (s or "").lower()))
        else:
            allowed_pairs.append((0, ""))

        if (pred_cat, pred_sub_letter or "") in allowed_pairs:
            subcategory_any_hits += 1
        subcategory_any_total += 1

        short_text = (text[:120] + "…") if len(text) > 120 else text
        print(f"\n[{idx}] {short_text}")
        print(f"\tone-prompt has_hate={has_hate} vs gt={gt_has_hate}")
        print(f"\tone-prompt category={pred_cat} vs gt={gt_cat}")
        if gt_cat != 0:
            print(f"\tone-prompt subcat={pred_sub or ''} vs gt={gt_subcat or ''}")
        # Print per-sentence classification for full transparency
        if isinstance(all_cls, dict):
            for s in all_cls.get("sentences", []):
                print(f"\t[SENT] {s['sentence']} => cat={s['category']}, sub={s['subcategory'] or ''}")

        # Token coverage reporting per record (optional):
        if isinstance(ext, dict):
            cov = ext.get("tokens_covered", 0)
            tot = ext.get("total_tokens", 0)
            ratio = (cov / tot) if tot else 0
            print(f"\tToken coverage: {cov}/{tot} = {ratio:.2%}")

    evaluator = HateSpeechEvaluator()
    binary_metrics = evaluator.evaluate_binary_classification(y_true_bin, y_pred_bin)
    category_metrics = evaluator.evaluate_multiclass_classification(y_true_cat, y_pred_cat)
    subcategory_metrics = evaluator.evaluate_multiclass_classification(y_true_sub, y_pred_sub)

    print("\n--- Jedan prompt: tačnost ---")
    print(f"Binary accuracy:      {binary_metrics['accuracy']:.4f} ({len(y_true_bin)} samples)")
    print(f"Category accuracy:    {category_metrics['accuracy']:.4f} ({len(y_true_cat)} samples)")
    print(f"Subcategory accuracy: {subcategory_metrics['accuracy']:.4f} ({len(y_true_sub)} samples)")
    if category_any_total > 0:
        print(f"Category accuracy (any-of-list):    {category_any_hits / category_any_total:.4f} ({category_any_total} samples)")
    if subcategory_any_total > 0:
        print(f"Subcategory accuracy (any-of-list): {subcategory_any_hits / subcategory_any_total:.4f} ({subcategory_any_total} samples)")

    return {
        "binary_metrics": binary_metrics,
        "category_metrics": {**category_metrics, "accuracy_any": (category_any_hits / category_any_total) if category_any_total else 0.0},
        "subcategory_metrics": {**subcategory_metrics, "accuracy_any": (subcategory_any_hits / subcategory_any_total) if subcategory_any_total else 0.0}
    }


def run(excel_path: str, models: List[str]) -> None:
    """Pokreni evaluaciju za više LLM-ova i prikaži metrike."""
    print("Učitavam dataset iz Excel fajla…")
    # For full-text spreadsheets where 'Category' contains comma-separated codes,
    # use the dedicated loader. Fallback to generic loader if needed.
    try:
        records = load_excel_full_text_dataset(excel_path)
    except Exception:
        # Fallback to generic loader if anything goes wrong
        records = load_excel_dataset(excel_path)

    # Izgradi mapiranje ime->tag (ako models nije zadan, koristi sve iz JSON-a)
    model_tags: Dict[str, str] = build_model_tags(models)

    one_prompt_evaluator = HateSpeechEvaluator()

    for model_name, tag in model_tags.items():
        print("\n" + "=" * 70)
        print(f"Evaluacija za model: {model_name} (tag: {tag})")
        print("=" * 70)
        print("-- Jedan prompt (detekcija + kategorija) --")
        res_one = one_prompt_evaluation_model_on_records(tag, records)

        print("\n>> Rezime metrika (jedan prompt)")
        one_prompt_evaluator.save_results(tag, res_one)
        one_prompt_evaluator.print_results(tag)

        # print("\nKonfuziona matrica - binarna:")
        # print(res["confusion_matrix_binary"])
        # print("\nKonfuziona matrica - kategorije 0–7:")
        # print(res["confusion_matrix_categories"])
        # print("\nIzveštaj klasifikacije po kategorijama:")
        # print(res["classification_report"])

    one_prompt_evaluator.save_results_to_excel("results/full_text_comparison.xlsx", sheet_name="Full")

    # Uporedni pregled (tabela)
    print("\n" + "#" * 70)
    print("Uporedni pregled metrika po modelima")
    print("#" * 70)
    # Uporedna tabela samo za kategoriju/subkategoriju između pristupa
    print("\n=== Poređenje pristupa (category/subcategory accuracy) ===")
    print(one_prompt_evaluator.results)
    print("----------------------------")
    for ev1 in one_prompt_evaluator.results:
        model = ev1.get("model")
       

        binary_metrics_1 = ev1["results"].get("binary_metrics", {})
        category_metrics_1 = ev1["results"].get("category_metrics", {})
        subcategory_metrics_1 = ev1["results"].get("subcategory_metrics", {})

        print(f"Model: {ev1.get('model')}")
        print(f"  One prompt   - binary acc: {binary_metrics_1.get('accuracy', 0):.4f}, cat acc: {category_metrics_1.get('accuracy', 0):.4f}, sub acc: {subcategory_metrics_1.get('accuracy', 0):.4f}")


if __name__ == "__main__":
    # Jednostavan podrazumevani poziv: koristi modele iz models/models.json ili data/models.json
    run(
        excel_path="data/text_hate_speech_labeled.xlsx",
        models=["llama", "qwen3"],  # ako je prazno, biće učitano iz models/models.json ili data/models.json
    )
