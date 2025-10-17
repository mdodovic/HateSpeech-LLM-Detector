"""
Demo skripta za detekciju i evaluaciju govora mržnje (srpski)
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict

from src.llm_detector import LLMDetector
from src.categories import get_category_prompt, HATE_SPEECH_CATEGORIES
from src.utils import load_excel_dataset, load_default_model_tags, build_model_tags
from src.evaluation import HateSpeechEvaluator


def evaluate_model_on_records(model_tag: str, records: List[Dict]) -> Dict:
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
    tokens_covered_list: List[int] = []
    total_tokens_list: List[int] = []

    for idx, rec in enumerate(records, start=1):
        text = (rec.get("text") or "").strip()
        if not text:
            continue

        # Zadatak 1: Binarna detekcija
        has_hate = detector.detect_hate_speech_binary(text)

        # Zadatak 3: Kategorizacija 0–7
        pred_cat, _ = detector.categorize_hate_speech(text, categories_prompt)

        # Zadatak 2: Izdvajanje recenica i token pokrivenost
        _sentences, tokens_covered, total_tokens = detector.extract_hate_speech_sentences(text)

        # Ground truth
        gt_has_hate = bool(rec.get("has_hate_speech", False))
        gt_cat = int(rec.get("category", 0))

        # Akumulacija
        y_true_bin.append(gt_has_hate)
        y_pred_bin.append(bool(has_hate))
        y_true_cat.append(gt_cat)
        y_pred_cat.append(int(pred_cat) if isinstance(pred_cat, int) else 0)
        tokens_covered_list.append(int(tokens_covered))
        total_tokens_list.append(int(total_tokens))

        # Kratki log za praćenje
        short_text = (text[:120] + "…") if len(text) > 120 else text
        print(f"\n[{idx}] {short_text}")
        print(f"    has_hate={has_hate}")

    # Evaluacija
    evaluator = HateSpeechEvaluator()
    binary_metrics = evaluator.evaluate_binary_classification(y_true_bin, y_pred_bin)
    category_metrics = evaluator.evaluate_multiclass_classification(y_true_cat, y_pred_cat, num_classes=8)
    token_metrics = evaluator.evaluate_token_coverage(tokens_covered_list, total_tokens_list)

    # # Dodatni izveštaji
    # from src.categories import HATE_SPEECH_CATEGORIES as CAT_NAMES
    # classification_report_text = evaluator.generate_classification_report(y_true_cat, y_pred_cat, CAT_NAMES)
    # cm_binary = evaluator.generate_confusion_matrix([int(v) for v in y_true_bin], [int(v) for v in y_pred_bin])
    # cm_cats = evaluator.generate_confusion_matrix(y_true_cat, y_pred_cat)

    return {
        "binary_metrics": binary_metrics,
        "category_metrics": category_metrics,
        "token_metrics": token_metrics,
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

    for name, tag in model_tags.items():
        print("\n" + "=" * 70)
        print(f"Evaluacija za model: {name} (tag: {tag})")
        print("=" * 70)
        res = evaluate_model_on_records(tag, records)
        all_results[name] = res

        # Štampa metrika za tekući model
        evaluator.print_results(
            tag,
            res["binary_metrics"],
            res["category_metrics"],
            res["token_metrics"],
        )
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
    for name, res in all_results.items():
        compare_input[name] = {
            "binary_metrics": res["binary_metrics"],
            "category_metrics": res["category_metrics"],
            "token_metrics": res["token_metrics"],
        }
    df = evaluator.compare_models(compare_input)
    try:
        # Lepši ispis ako je dostupan
        import pandas as _pd  # noqa: F401
        print(df)
    except Exception:
        # Fallback na običan dict
        print(compare_input)


if __name__ == "__main__":
    # Jednostavan podrazumevani poziv: koristi modele iz models/models.json ili data/models.json
    run(
        excel_path="data/hate_speech_labeled_samples_small.xlsx",
        models=[],  # ako je prazno, biće učitano iz models/models.json ili data/models.json
    )
