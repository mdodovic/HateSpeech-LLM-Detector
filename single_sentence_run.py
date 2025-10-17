"""
This is cumulative run for task 1 (yes/no hatespeech) and 3 (categorization).

First way is to combine 2 different prompts by asking if there is hatespeech and then if does what category it belongs to.

Second way is to use single prompt to do both tasks at once.

Task 2 is extraction of the hate speech sentences extraction from long texts and token coverage, which is not included here, because we sentence-by-sentence approach.

"""

import argparse
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

    for idx, rec in enumerate(records, start=1):
        text = (rec.get("text") or "").strip()
        if not text:
            continue

        # Zadatak 1: Binarna detekcija
        has_hate = detector.detect_hate_speech_binary(text)
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
            pred_cat, pred_sub_cat = detector.categorize_hate_speech(text, categories_prompt)
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
    category_metrics = evaluator.evaluate_multiclass_classification(y_true_cat, y_pred_cat, num_classes=8)
    # token_metrics = evaluator.evaluate_token_coverage(tokens_covered_list, total_tokens_list)

    # Tačnost za kategoriju i podkategoriju
    cat_accuracy = float(accuracy_score(y_true_cat, y_pred_cat)) if y_true_cat else 0.0
    sub_accuracy = float(accuracy_score(y_true_sub, y_pred_sub)) if y_true_sub else 0.0
    print("\n--- Dodatne metrike tačnosti ---")
    print(f"Category accuracy:    {cat_accuracy:.4f} ({len(y_true_cat)} samples)")
    print(f"Subcategory accuracy: {sub_accuracy:.4f} ({len(y_true_sub)} samples)")

    # Dodatni izveštaji
    # classification_report_text = evaluator.generate_classification_report(y_true_cat, y_pred_cat, CAT_NAMES)
    # cm_binary = evaluator.generate_confusion_matrix([int(v) for v in y_true_bin], [int(v) for v in y_pred_bin])
    # cm_cats = evaluator.generate_confusion_matrix(y_true_cat, y_pred_cat)

    return {
        "binary_metrics": binary_metrics,
        "category_metrics": category_metrics,
        "category_accuracy": cat_accuracy,
        "subcategory_accuracy": sub_accuracy,
        # "token_metrics": token_metrics,
        # "classification_report": classification_report_text,
        # "confusion_matrix_binary": cm_binary,
        # "confusion_matrix_categories": cm_cats,
    }


def run(excel_path: str, models: List[str]) -> None:
    """Pokreni evaluaciju za više LLM-ova i prikaži metrike."""
    print("Učitavam dataset iz Excel fajla…")
    records = load_excel_dataset(excel_path)

    # Izgradi mapiranje ime->tag (ako models nije zadan, koristi sve iz JSON-a)
    model_tags: Dict[str, str] = build_model_tags(models)

    all_results: Dict[str, Dict] = {}
    evaluator = HateSpeechEvaluator()

    for model_name, tag in model_tags.items():
        print("\n" + "=" * 70)
        print(f"Evaluacija za model: {model_name} (tag: {tag})")
        print("=" * 70)
        res = two_prompts_evaluation_model_on_records(tag, records)
        all_results[model_name] = res

        # Štampa metrika za tekući model
        evaluator.save_results(
            tag,
            res["binary_metrics"],
            res["category_metrics"],
            # res["token_metrics"],
        )
        evaluator.print_results(tag)

        # print("\nKonfuziona matrica - binarna:")
        # print(res["confusion_matrix_binary"])
        # print("\nKonfuziona matrica - kategorije 0–7:")
        # print(res["confusion_matrix_categories"])
        # print("\nIzveštaj klasifikacije po kategorijama:")
        # print(res["classification_report"])

    # Uporedni pregled (tabela)
    print("\n" + "#" * 70)
    print("Uporedni pregled metrika po modelima")
    print("#" * 70)
    compare_input = {}
    for model_name, res in all_results.items():
        compare_input[model_name] = {
            "binary_metrics": res["binary_metrics"],
            "category_metrics": res["category_metrics"],
            # "token_metrics": res["token_metrics"],
        }

    evaluator.compare_models(compare_input)
    evaluator.save_results_to_excel("results/evaluation_comparison.xlsx", verbose=True)


if __name__ == "__main__":
    # Jednostavan podrazumevani poziv: koristi modele iz models/models.json ili data/models.json
    run(
        excel_path="data/hate_speech_labeled_samples_small.xlsx",
        models=["deepseek"],  # ako je prazno, biće učitano iz models/models.json ili data/models.json
    )
