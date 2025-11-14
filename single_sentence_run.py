"""
This is cumulative run for task 1 (yes/no hatespeech) and 3 (categorization).

First way is to combine 2 different prompts by asking if there is hatespeech and then if does what category it belongs to.

Second way is to use single prompt to do both tasks at once.

Task 2 is extraction of the hate speech sentences extraction from long texts and token coverage, which is not included here, because we sentence-by-sentence approach.

"""

import argparse
import time
import re
from sklearn.metrics import accuracy_score
import json
from pathlib import Path
from typing import Optional, List, Dict

from src.llm_detector import LLMDetector
from src.categories import get_category_prompt, HATE_SPEECH_CATEGORIES
from src.utils import load_excel_dataset, build_model_tags
from src.evaluation import HateSpeechEvaluator


def two_prompts_evaluation_model_on_records(model_tag: str, records: List[Dict]) -> Dict:
    """Pokreni sve zadatke i evaluaciju za jedan model nad datim rekordima."""

    print(f"Uzoraka za obradu: {len(records)}")
    print("Inicijalizujem LLM detektor…")
    detector = LLMDetector(model_tag)
    categories_prompt = get_category_prompt()

    # Priprema za evaluaciju
    y_true_bin: List[bool] = []
    y_pred_bin: List[bool] = []
    y_true_cat: List[int] = []
    y_pred_cat: List[int] = []
    y_true_sub: List[str] = []
    y_pred_sub: List[str] = []

    # Timing accumulators (LLM-only)
    total_detect_s = 0.0
    total_categorize_s = 0.0
    n_detect_calls = 0
    n_categorize_calls = 0

    for idx, rec in enumerate(records, start=1):
        text = (rec.get("text") or "").strip()
        if not text:
            continue

        # Zadatak 1: Binarna detekcija
        t0 = time.perf_counter()
        has_hate = detector.detect_hate_speech_binary(text)
        total_detect_s += (time.perf_counter() - t0)
        n_detect_calls += 1
        gt_has_hate = bool(rec.get("has_hate_speech", False))
        gt_cat = int(rec.get("category", 0))
        gt_subcat = str(rec.get("subcategory", ""))

        short_text = (text[:120] + "…") if len(text) > 120 else text
        print(f"\n[{idx}] {short_text}")
        print(f"\thas_hate={has_hate} {'  =  ' if has_hate == gt_has_hate else '!='} ground_truth={gt_has_hate}")

        y_true_bin.append(gt_has_hate)
        y_pred_bin.append(bool(has_hate))

        # Zadatak 3: Kategorizacija 0–7
        if has_hate:
            t1 = time.perf_counter()
            pred_cat, pred_sub_cat = detector.categorize_hate_speech(text, categories_prompt)
            total_categorize_s += (time.perf_counter() - t1)
            n_categorize_calls += 1
            print()

            print(f"\tpredicted_category={pred_cat} {'  =  ' if pred_cat == gt_cat else '!='} ground_truth={gt_cat}")
            # Only compare subcategory when GT has hate (category != 0)
            if gt_cat != 0:
                # Normalize predicted subcategory to a letter (e.g., '3b' -> 'b')
                pred_sub_letter = ""
                if isinstance(pred_sub_cat, str):
                    m = re.match(r"^\s*([0-7])\s*([a-z])\s*$", pred_sub_cat, flags=re.IGNORECASE)
                    if m:
                        pred_sub_letter = m.group(2).lower()
                    elif re.match(r"^[a-z]$", pred_sub_cat, flags=re.IGNORECASE):
                        pred_sub_letter = pred_sub_cat.lower()
                print(
                    f"\tpredicted_subcategory={pred_sub_letter or ''} "
                    f"{'  =  ' if (pred_sub_letter or '') == (gt_subcat or '') else '!='} "
                    f"ground_truth_subcategory={gt_subcat or ''}"
                )
                # Accumulate for subcategory accuracy
                y_true_sub.append(gt_subcat or "")
                y_pred_sub.append(pred_sub_letter or "")

            # Akumulacija
            y_true_cat.append(gt_cat)
            y_pred_cat.append(int(pred_cat) if isinstance(pred_cat, int) else 0)

    # Evaluacija
    evaluator = HateSpeechEvaluator()
    binary_metrics = evaluator.evaluate_binary_classification(y_true_bin, y_pred_bin)
    category_metrics = evaluator.evaluate_multiclass_classification(y_true_cat, y_pred_cat)
    subcategory_metrics = evaluator.evaluate_multiclass_classification(y_true_sub, y_pred_sub)

    # Tačnost za kategoriju i podkategoriju
    print("\n--- Dodatne metrike tačnosti ---")
    print(f"Binary accuracy:      {binary_metrics['accuracy']:.4f} ({len(y_true_bin)} samples)")
    print(f"Category accuracy:    {category_metrics['accuracy']:.4f} ({len(y_true_cat)} samples)")
    print(f"Subcategory accuracy: {subcategory_metrics['accuracy']:.4f} ({len(y_true_sub)} samples)")

    # Vreme (samo LLM pozivi)
    avg_detect_ms = (total_detect_s / n_detect_calls * 1000.0) if n_detect_calls else 0.0
    avg_categorize_ms = (total_categorize_s / n_categorize_calls * 1000.0) if n_categorize_calls else 0.0
    total_llm_s = total_detect_s + total_categorize_s
    print("\n--- Vremenski podaci (LLM) — dva prompta ---")
    print(f"detect_hate_speech_binary: total={total_detect_s:.3f}s, calls={n_detect_calls}, avg={avg_detect_ms:.1f} ms/call")
    print(f"categorize_hate_speech:    total={total_categorize_s:.3f}s, calls={n_categorize_calls}, avg={avg_categorize_ms:.1f} ms/call")
    print(f"LLM total (detect+categorize): {total_llm_s:.3f}s")

    return {
        "binary_metrics": binary_metrics,
        "category_metrics": category_metrics,
        "subcategory_metrics": subcategory_metrics,
        "timing": {
            "detect_total_s": total_detect_s,
            "detect_calls": n_detect_calls,
            "detect_avg_ms": avg_detect_ms,
            "categorize_total_s": total_categorize_s,
            "categorize_calls": n_categorize_calls,
            "categorize_avg_ms": avg_categorize_ms,
            "llm_total_s": total_llm_s,
        },
    }


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

    # Timing accumulators (LLM-only)
    total_oneprompt_s = 0.0
    n_oneprompt_calls = 0

    for idx, rec in enumerate(records, start=1):
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        gt_has_hate = bool(rec.get("has_hate_speech", False))
        gt_cat = int(rec.get("category", 0))
        gt_subcat = str(rec.get("subcategory", ""))

        t0 = time.perf_counter()
        result = detector.detect_and_categorize(text, categories_prompt)
        total_oneprompt_s += (time.perf_counter() - t0)
        n_oneprompt_calls += 1
        has_hate = bool(result.get("has_hate_speech", False))
        pred_cat = int(result.get("category", 0))
        pred_sub = str(result.get("subcategory", ""))

        y_true_bin.append(gt_has_hate)
        y_pred_bin.append(has_hate)
        y_true_cat.append(gt_cat)
        y_pred_cat.append(pred_cat)
        if gt_cat != 0:
            # normalize sub to single letter if needed
            m = re.match(r"^\s*([0-7])\s*([a-z])\s*$", pred_sub, flags=re.IGNORECASE)
            pred_sub_letter = m.group(2).lower() if m else (pred_sub.lower() if re.match(r"^[a-z]$", pred_sub, flags=re.IGNORECASE) else "")
            y_true_sub.append(gt_subcat or "")
            y_pred_sub.append(pred_sub_letter or "")

        short_text = (text[:120] + "…") if len(text) > 120 else text
        print(f"\n[{idx}] {short_text}")
        print(f"\tone-prompt has_hate={has_hate} vs gt={gt_has_hate}")
        print(f"\tone-prompt category={pred_cat} vs gt={gt_cat}")
        if gt_cat != 0:
            print(f"\tone-prompt subcat={pred_sub or ''} vs gt={gt_subcat or ''}")

    evaluator = HateSpeechEvaluator()
    binary_metrics = evaluator.evaluate_binary_classification(y_true_bin, y_pred_bin)
    category_metrics = evaluator.evaluate_multiclass_classification(y_true_cat, y_pred_cat)
    subcategory_metrics = evaluator.evaluate_multiclass_classification(y_true_sub, y_pred_sub)

    print("\n--- Jedan prompt: tačnost ---")
    print(f"Binary accuracy:      {binary_metrics['accuracy']:.4f} ({len(y_true_bin)} samples)")
    print(f"Category accuracy:    {category_metrics['accuracy']:.4f} ({len(y_true_cat)} samples)")
    print(f"Subcategory accuracy: {subcategory_metrics['accuracy']:.4f} ({len(y_true_sub)} samples)")

    # Vreme (samo LLM pozivi)
    avg_oneprompt_ms = (total_oneprompt_s / n_oneprompt_calls * 1000.0) if n_oneprompt_calls else 0.0
    print("\n--- Vremenski podaci (LLM) — jedan prompt ---")
    print(f"detect_and_categorize: total={total_oneprompt_s:.3f}s, calls={n_oneprompt_calls}, avg={avg_oneprompt_ms:.1f} ms/call")

    return {
        "binary_metrics": binary_metrics,
        "category_metrics": category_metrics,
        "subcategory_metrics": subcategory_metrics,
        "timing": {
            "oneprompt_total_s": total_oneprompt_s,
            "oneprompt_calls": n_oneprompt_calls,
            "oneprompt_avg_ms": avg_oneprompt_ms,
        }
    }


def run(excel_path: str, models: List[str] = []) -> None:
    """Pokreni evaluaciju za više LLM-ova i prikaži metrike."""
    print("Učitavam dataset iz Excel fajla…")
    records = load_excel_dataset(excel_path)

    # Izgradi mapiranje ime->tag (ako models nije zadan, koristi sve iz JSON-a)
    model_tags: Dict[str, str] = build_model_tags(models)

    one_prompt_evaluator = HateSpeechEvaluator()
    two_prompt_evaluator = HateSpeechEvaluator()

    for model_name, tag in model_tags.items():
        print("\n" + "=" * 70)
        print(f"Evaluacija za model: {model_name} (tag: {tag})")
        print("=" * 70)
        print("-- Dva prompta (detekcija + kategorija) --")
        res_two = two_prompts_evaluation_model_on_records(tag, records)
        print("-- Jedan prompt (detekcija + kategorija) --")
        res_one = one_prompt_evaluation_model_on_records(tag, records)

        # Štampa metrika za tekući model
        print("\n>> Rezime metrika (dva prompta)")
        two_prompt_evaluator.save_results(tag, res_two)
        two_prompt_evaluator.print_results(tag)
        # Dodatno: prikaži i vremenske podatke (LLM) po modelu
        t2 = (res_two or {}).get("timing", {})
        if t2:
            print("-- Vremenski podaci (LLM) — dva prompta")
            print(f"detect_hate_speech_binary: total={t2.get('detect_total_s', 0.0):.3f}s, calls={t2.get('detect_calls', 0)}, avg={t2.get('detect_avg_ms', 0.0):.1f} ms/call")
            print(f"categorize_hate_speech:    total={t2.get('categorize_total_s', 0.0):.3f}s, calls={t2.get('categorize_calls', 0)}, avg={t2.get('categorize_avg_ms', 0.0):.1f} ms/call")
            print(f"LLM total (detect+categorize): {t2.get('llm_total_s', 0.0):.3f}s")

        print("\n>> Rezime metrika (jedan prompt)")
        one_prompt_evaluator.save_results(tag, res_one)
        one_prompt_evaluator.print_results(tag)
        t1 = (res_one or {}).get("timing", {})
        if t1:
            print("-- Vremenski podaci (LLM) — jedan prompt")
            print(f"detect_and_categorize: total={t1.get('oneprompt_total_s', 0.0):.3f}s, calls={t1.get('oneprompt_calls', 0)}, avg={t1.get('oneprompt_avg_ms', 0.0):.1f} ms/call")

        # print("\nKonfuziona matrica - binarna:")
        # print(res["confusion_matrix_binary"])
        # print("\nKonfuziona matrica - kategorije 0–7:")
        # print(res["confusion_matrix_categories"])
        # print("\nIzveštaj klasifikacije po kategorijama:")
        # print(res["classification_report"])

    two_prompt_evaluator.save_results_to_excel("results/single_sentence_comparison.xlsx", sheet_name="Two Prompts")
    one_prompt_evaluator.save_results_to_excel("results/single_sentence_comparison.xlsx", sheet_name="One Prompt")

    # Uporedni pregled (tabela)
    print("\n" + "#" * 70)
    print("Uporedni pregled metrika po modelima")
    print("#" * 70)
    # Uporedna tabela samo za kategoriju/subkategoriju između pristupa
    print("\n=== Poređenje pristupa (category/subcategory accuracy) ===")
    for ev1 in one_prompt_evaluator.results:
        model = ev1.get("model")
        ev2 = next((e for e in two_prompt_evaluator.results if e.get("model") == model), None)
        if not ev2:
            continue

        binary_metrics_1 = ev1["results"].get("binary_metrics", {})
        category_metrics_1 = ev1["results"].get("category_metrics", {})
        subcategory_metrics_1 = ev1["results"].get("subcategory_metrics", {})

        binary_metrics_2 = ev2["results"].get("binary_metrics", {})
        category_metrics_2 = ev2["results"].get("category_metrics", {})
        subcategory_metrics_2 = ev2["results"].get("subcategory_metrics", {})

        print(f"Model: {ev1.get('model'), ev2.get('model')}")
        print(f"  Two prompts  - binary acc: {binary_metrics_2.get('accuracy', 0):.4f}, cat acc: {category_metrics_2.get('accuracy', 0):.4f}, sub acc: {subcategory_metrics_2.get('accuracy', 0):.4f}")
        print(f"  One prompt   - binary acc: {binary_metrics_1.get('accuracy', 0):.4f}, cat acc: {category_metrics_1.get('accuracy', 0):.4f}, sub acc: {subcategory_metrics_1.get('accuracy', 0):.4f}")


if __name__ == "__main__":
    # Jednostavan podrazumevani poziv: koristi modele iz models/models.json ili data/models.json
    run(
        excel_path="data/single_sentence_hate_speech_labeled_samples_small.xlsx",
        # models=["llama", "qwen3"],  # ako je prazno, biće učitano iz models/models.json ili data/models.json
    )
