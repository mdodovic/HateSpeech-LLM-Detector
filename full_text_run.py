"""
This is cumulative run for task 1 (yes/no hatespeech) and 3 (categorization).

First way is to combine 2 different prompts by asking if there is hatespeech and then if does what category it belongs to.

Second way is to use single prompt to do both tasks at once.

Task 2 is extraction of the hate speech sentences extraction from long texts and token coverage, which is not included here, because we sentence-by-sentence approach.

"""

import re
import time
import pandas as pd
from sklearn.metrics import accuracy_score
from pathlib import Path
from typing import  List, Dict

from src.llm_detector import LLMDetector
from src.categories import get_category_prompt, HATE_SPEECH_CATEGORIES
from src.utils import load_excel_dataset, build_model_tags, parse_category_and_subcategory, load_excel_full_text_dataset
from src.evaluation import HateSpeechEvaluator

def one_prompt_evaluation_model_on_records(model_tag: str, records: List[Dict]) -> Dict:
    """Evaluacija koristeći JEDAN prompt (classify_full_all) i merenje po rečenici."""
    print(f"Uzoraka za obradu: {len(records)}")
    print("Inicijalizujem LLM detektor…")
    detector = LLMDetector(model_tag)
    categories_prompt = get_category_prompt()

    # Metričke liste po rečenici
    y_true_bin_sent: List[bool] = []
    y_pred_bin_sent: List[bool] = []
    y_true_cat_sent: List[int] = []
    y_pred_cat_sent: List[int] = []
    y_true_sub_sent: List[str] = []
    y_pred_sub_sent: List[str] = []

    # Timing accumulators for LLM (classify_all_sentences)
    total_classify_all_s = 0.0
    n_classify_all_calls = 0

    for idx, rec in enumerate(records, start=1):
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        gt_all_cats = rec.get("all_categories") or []
        gt_all_subs = rec.get("all_subcategories") or []

        # Jedini poziv modelu: klasifikuj SVE rečenice
        t0 = time.perf_counter()
        all_cls = detector.classify_all_sentences(text, categories_prompt)
        total_classify_all_s += (time.perf_counter() - t0)
        n_classify_all_calls += 1
        pred_sentences = all_cls.get("sentences", []) if isinstance(all_cls, dict) else []

        short_text = (text[:120] + "…") if len(text) > 120 else text
        print(f"\n[{idx}] {short_text}")

        # Uskladi po indeksu: rečenica i-ti GT protiv i-te predikcije
        n = min(len(pred_sentences), len(gt_all_cats))
        if len(pred_sentences) != len(gt_all_cats):
            print(f"\tUpozorenje: broj predikovanih rečenica ({len(pred_sentences)}) != broj GT oznaka ({len(gt_all_cats)}). Poređenje na prvih {n}.")

        for i in range(n):
            pred = pred_sentences[i]
            gt_cat = int(gt_all_cats[i])
            gt_sub = str(gt_all_subs[i] or "").lower()

            # Predikcija
            pred_cat = int(pred.get("category", 0))
            raw_sub = str(pred.get("subcategory", "") or "").strip()
            # Normalizuj subkategoriju na slovo (ako je 3b/"b")
            parsed_sub = parse_category_and_subcategory(raw_sub)
            pred_sub_letter = parsed_sub.get("subcategory", "").lower()
            if not pred_sub_letter and re.match(r"^[a-z]$", raw_sub, flags=re.IGNORECASE):
                pred_sub_letter = raw_sub.lower()

            # Binary po rečenici: (gt_cat != 0) vs (pred_cat != 0)
            y_true_bin_sent.append(gt_cat != 0)
            y_pred_bin_sent.append(pred_cat != 0)
            # Kategorija
            y_true_cat_sent.append(gt_cat)
            y_pred_cat_sent.append(pred_cat)
            # Subkategorija samo kad je GT != 0
            if gt_cat != 0:
                y_true_sub_sent.append(gt_sub)
                y_pred_sub_sent.append(pred_sub_letter or "")

            # Štampa po rečenici
            sent_text = pred.get("sentence", "")
            print(f"\t[SENT {i+1}] {sent_text}")
            print(f"\t         pred: cat={pred_cat}, sub={pred_sub_letter or ''}  |  gt: cat={gt_cat}, sub={gt_sub}")

    evaluator = HateSpeechEvaluator()
    binary_metrics = evaluator.evaluate_binary_classification(y_true_bin_sent, y_pred_bin_sent)
    category_metrics = evaluator.evaluate_multiclass_classification(y_true_cat_sent, y_pred_cat_sent)
    subcategory_metrics = evaluator.evaluate_multiclass_classification(y_true_sub_sent, y_pred_sub_sent)

    print("\n--- Jedan prompt (po rečenici): tačnost ---")
    print(f"Binary accuracy (sent):      {binary_metrics['accuracy']:.4f} ({len(y_true_bin_sent)} sentences)")
    print(f"Category accuracy (sent):    {category_metrics['accuracy']:.4f} ({len(y_true_cat_sent)} sentences)")
    print(f"Subcategory accuracy (sent): {subcategory_metrics['accuracy']:.4f} ({len(y_true_sub_sent)} sentences)")

    # Vreme (samo LLM pozivi)
    avg_classify_all_ms = (total_classify_all_s / n_classify_all_calls * 1000.0) if n_classify_all_calls else 0.0
    print("\n--- Vremenski podaci (LLM) — classify_all_sentences ---")
    print(f"classify_all_sentences: total={total_classify_all_s:.3f}s, calls={n_classify_all_calls}, avg={avg_classify_all_ms:.1f} ms/call")

    return {
        "binary_metrics": binary_metrics,
        "category_metrics": category_metrics,
        "subcategory_metrics": subcategory_metrics,
        "timing": {
            "classify_all_total_s": total_classify_all_s,
            "classify_all_calls": n_classify_all_calls,
            "classify_all_avg_ms": avg_classify_all_ms,
        },
    }


def run(excel_path: str, models: List[str]) -> None:
    """Pokreni evaluaciju za više LLM-ova i prikaži metrike."""
    print("Učitavam dataset iz Excel fajla…")
    # For full-text spreadsheets where 'Category' contains comma-separated codes,
    # use the dedicated loader. Fallback to generic loader if needed.
    records = load_excel_full_text_dataset(excel_path)
    # Ograniči na jedan pasus (za sada)
    if len(records) > 1:
        records = records[:1]

    # Izgradi mapiranje ime->tag (ako models nije zadan, koristi sve iz JSON-a)
    model_tags: Dict[str, str] = build_model_tags(models)

    one_prompt_evaluator = HateSpeechEvaluator()

    for model_name, tag in model_tags.items():
        print("\n" + "=" * 70)
        print(f"Evaluacija za model: {model_name} (tag: {tag})")
        print("=" * 70)
        print("-- Jedan prompt (classify_full_all; po rečenici) --")
        res_one = one_prompt_evaluation_model_on_records(tag, records)

        print("\n>> Rezime metrika (jedan prompt)")
        one_prompt_evaluator.save_results(tag, res_one)
        # Dodatno: prikaži i vremenske podatke po modelu
        t = (res_one or {}).get("timing", {})
        if t:
            print("-- Vremenski podaci (LLM)")
            print(f"classify_all_sentences: total={t.get('classify_all_total_s', 0.0):.3f}s, calls={t.get('classify_all_calls', 0)}, avg={t.get('classify_all_avg_ms', 0.0):.1f} ms/call")

        # print("\nKonfuziona matrica - binarna:")
        # print(res["confusion_matrix_binary"])
        # print("\nKonfuziona matrica - kategorije 0–7:")
        # print(res["confusion_matrix_categories"])
        # print("\nIzveštaj klasifikacije po kategorijama:")
        # print(res["classification_report"])

    one_prompt_evaluator.save_results_to_excel("results/full_text_comparison.xlsx", sheet_name="Full")

    # Uporedni pregled (tabela)
    print("\n" + "#" * 70)
    print("Uporedni pregled metrika po modelima (po rečenici)")
    print("#" * 70)
    for ev in one_prompt_evaluator.results:
        bm = ev["results"].get("binary_metrics", {})
        cm = ev["results"].get("category_metrics", {})
        sm = ev["results"].get("subcategory_metrics", {})
        print(f"Model: {ev.get('model')}")
        print(f"  Jedan prompt - binary acc: {bm.get('accuracy', 0):.4f}, cat acc: {cm.get('accuracy', 0):.4f}, sub acc: {sm.get('accuracy', 0):.4f}")


if __name__ == "__main__":
    # Jednostavan podrazumevani poziv: koristi modele iz models/models.json ili data/models.json
    run(
        excel_path="data/paragraph_hate_speech_labeled.xlsx",
        models=["llama", "qwen3"],  # ako je prazno, biće učitano iz models/models.json ili data/models.json
    )
