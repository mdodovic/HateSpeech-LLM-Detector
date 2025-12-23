"""
This is cumulative run for task 1 (yes/no hatespeech) and 3 (categorization).

First way is to combine 2 different prompts by asking if there is hatespeech and then if does what category it belongs to.

Second way is to use single prompt to do both tasks at once.

Task 2 is extraction of the hate speech sentences extraction from long texts and token coverage, which is not included here, because we sentence-by-sentence approach.

"""
# TODO: Proveriti dxa li ovo dobro sabira i gde ga sabira:


import argparse
import time
import re
from sklearn.metrics import accuracy_score
import json
from pathlib import Path
from typing import Optional, List, Dict

from src.llm_detector import LLMDetector
from src.categories import get_category_prompt, HATE_SPEECH_CATEGORIES
from src.utils import load_excel_dataset, build_model_tags, parse_category_and_subcategory
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
    # Multi-label metric accumulators (only over hate samples)
    multi_label_sub_success: List[bool] = []
    multi_label_cat_success: List[bool] = []
    multi_label_denominator = 0

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
            pred_codes = detector.categorize_hate_speech(text, categories_prompt)
            total_categorize_s += (time.perf_counter() - t1)
            n_categorize_calls += 1
            print()

            # Choose primary predicted category for metrics (first non-zero if present)
            primary_cat = 0
            primary_sub_letter = ""
            for code in (pred_codes or []):
                parsed = parse_category_and_subcategory(code)
                if int(parsed.get("category", 0)) != 0:
                    primary_cat = int(parsed.get("category", 0))
                    primary_sub_letter = str(parsed.get("subcategory", "") or "").lower()
                    break
            if primary_cat == 0 and pred_codes:
                parsed = parse_category_and_subcategory(pred_codes[0])
                primary_cat = int(parsed.get("category", 0))
                primary_sub_letter = str(parsed.get("subcategory", "") or "").lower()

            codes_str = ",".join(pred_codes) if pred_codes else ""
            gt_codes = rec.get("all_codes") or ([f"{gt_cat}{gt_subcat}".lower()] if gt_cat or gt_subcat else [])
            print(f"\tpredicted_codes={codes_str or ''}")
            if gt_codes:
                print(f"\tground_truth_codes={','.join(gt_codes)}")
            # print(f"\tprimary_category={primary_cat} {'  =  ' if primary_cat == gt_cat else '!='} ground_truth={gt_cat}")
            # Only compare subcategory when GT has hate (category != 0)
            if gt_cat != 0:
                # If one of the predicted codes matches GT category, use its letter for subcategory comparison
                pred_sub_letter = primary_sub_letter
                for code in (pred_codes or []):
                    m = re.match(r"^\s*([0-7])\s*([a-z])\s*$", code, flags=re.IGNORECASE)
                    if m and int(m.group(1)) == gt_cat:
                        pred_sub_letter = m.group(2).lower()
                        break
                # print(
                #     f"\tpredicted_subcategory={pred_sub_letter or ''} "
                #     f"{'  =  ' if (pred_sub_letter or '') == (gt_subcat or '') else '!='} "
                #     f"ground_truth_subcategory={gt_subcat or ''}"
                # )
                # Accumulate for subcategory accuracy
                y_true_sub.append(gt_subcat or "")
                y_pred_sub.append(pred_sub_letter or "")

            # Akumulacija
            y_true_cat.append(gt_cat)
            y_pred_cat.append(int(primary_cat))

            # --- Detaljna multi-label komparacija ---
            gt_codes_lower = [c.lower() for c in (gt_codes or [])]
            gt_cat_numbers = {re.match(r"^([0-7])", c).group(1) for c in gt_codes_lower if re.match(r"^([0-7])", c)}
            pred_with_sub = [c for c in pred_codes if re.match(r"^[0-7][a-z]$", c)]
            pred_without_sub = [c for c in pred_codes if re.match(r"^[0-7]$", c)]

            # 1) Prvo proveri sve predikcije sa podkategorijom za TAČNE mečeve
            exact_hits = [code for code in pred_with_sub if code in gt_codes_lower]
            for code in exact_hits:
                print(f"\tMATCH: {code} (exact)")

            if not exact_hits:
                # 2) Ako nema nijednog tačnog meča, proveri PARTIAL (kategorija poklapa, podkategorija ne)
                first_unmatched_printed = False
                for code in pred_with_sub:
                    if code in gt_codes_lower:
                        # (Ovo se ne dešava jer bi bilo u exact_hits, ali ostavljeno radi robusnosti)
                        print(f"\tMATCH: {code} (exact)")
                        continue
                    base_cat = code[0]
                    if base_cat in gt_cat_numbers:
                        print(f"\tPARTIAL: {code} (category {base_cat} present, subcategory differs)")
                    elif not first_unmatched_printed:
                        print(f"\tNO MATCH: {code} (category {base_cat} absent in ground truth)")
                        first_unmatched_printed = True

                # 3) Tek sada proveri predikcije bez podkategorije
                for code in pred_without_sub:
                    if code in gt_cat_numbers:
                        print(f"\tCATEGORY MATCH: {code} (subcategory not specified / differs)")
                    elif not first_unmatched_printed:
                        print(f"\tNO MATCH: {code} (category absent)")
                        first_unmatched_printed = True

            # --- Akumulacija za multi-label metrike ---
            # Ground-truth categories/subcategories per sample
            # success_category: at least one predicted code matches a GT category (exact or partial)
            # success_subcategory: at least one predicted code matches a GT code exactly (with subcategory)
            #   PLUS: ako GT kategorija NEMA podkategoriju (npr. '2','5','7'), svaki pogodak te kategorije
            #   računa se kao pogodak podkategorije (tretiraj kao exact na podkategoriji).
            success_sub = any(code in gt_codes_lower for code in pred_with_sub)
            success_cat = False

            # GT kategorije bez podkategorije (jednocifreni kodovi)
            gt_no_sub_cats = {code[0] for code in gt_codes_lower if re.match(r"^[0-7]$", code)}
            pred_base_cats = {code[0] for code in (pred_codes or []) if re.match(r"^[0-7]([a-z])?$", code)}
            if not success_sub and gt_no_sub_cats and (pred_base_cats & gt_no_sub_cats):
                success_sub = True

            if success_sub:
                success_cat = True  # exact (ili no-sub GT) implicira uspeh kategorije
            else:
                # Check category-only matches (with or without subcategory predicted)
                for code in pred_codes:
                    base = code[0] if code else ""
                    if base in gt_cat_numbers:
                        success_cat = True
                        break
            multi_label_sub_success.append(success_sub)
            multi_label_cat_success.append(success_cat)
            multi_label_denominator += 1  # count only hate samples

    # Evaluacija
    evaluator = HateSpeechEvaluator()
    binary_metrics = evaluator.evaluate_binary_classification(y_true_bin, y_pred_bin)
    category_metrics = evaluator.evaluate_multiclass_classification(y_true_cat, y_pred_cat)
    subcategory_metrics = evaluator.evaluate_multiclass_classification(y_true_sub, y_pred_sub)

    # Multi-label dodatne metrike (samo za uzorke sa hate govorom)
    multi_label_metrics = {}
    if multi_label_denominator > 0:
        multi_label_metrics = {
            "multi_category_accuracy": sum(1 for v in multi_label_cat_success if v) / multi_label_denominator,
            "multi_subcategory_accuracy": sum(1 for v in multi_label_sub_success if v) / multi_label_denominator,
            "multi_samples": multi_label_denominator,
        }

    # Tačnost za kategoriju i podkategoriju
    print("\n--- Dodatne metrike tačnosti ---")
    print(f"Binary accuracy:      {binary_metrics['accuracy']:.4f} ({len(y_true_bin)} samples)")
    print(f"Category accuracy:    {category_metrics['accuracy']:.4f} ({len(y_true_cat)} samples)")
    print(f"Subcategory accuracy: {subcategory_metrics['accuracy']:.4f} ({len(y_true_sub)} samples)")
    if multi_label_metrics:
        print("Multi-label category accuracy:    {:.4f} ({} hate samples)".format(multi_label_metrics['multi_category_accuracy'], multi_label_metrics['multi_samples']))
        print("Multi-label subcategory accuracy: {:.4f} ({} hate samples)".format(multi_label_metrics['multi_subcategory_accuracy'], multi_label_metrics['multi_samples']))

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
        "multi_label_metrics": multi_label_metrics,
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
        pred_codes = result.get("codes", [])

        y_true_bin.append(gt_has_hate)
        y_pred_bin.append(has_hate)
        # Kategorizaciju računamo samo za uzorke koje u ground-truth imaju hate govor (isključujemo slučajeve gde su i GT i predikcija bez hate)
        # Ovo poravnava ponašanje sa dva-prompt pristupom i uklanja true-negative uzorke iz denominatora.
        if  gt_has_hate or  has_hate:
            y_true_cat.append(gt_cat)
            y_pred_cat.append(pred_cat)
        # Subkategoriju računamo samo za uzorke gde JE hate govor i u GT i u predikciji (izbacujemo false negative slučajeve poput [6]).
        if gt_cat != 0 and has_hate and gt_has_hate:
            m = re.match(r"^\s*([0-7])\s*([a-z])\s*$", pred_sub, flags=re.IGNORECASE)
            pred_sub_letter = m.group(2).lower() if m else (pred_sub.lower() if re.match(r"^[a-z]$", pred_sub, flags=re.IGNORECASE) else "")
            y_true_sub.append(gt_subcat or "")
            y_pred_sub.append(pred_sub_letter or "")

        short_text = (text[:120] + "…") if len(text) > 120 else text
        print(f"\n[{idx}] {short_text}")
        print(f"\thas_hate={has_hate} {'  =  ' if has_hate == gt_has_hate else '!='} ground_truth={gt_has_hate}")

        # Mirror two-prompt detail only when hate is detected
        if has_hate:
            # Choose primary predicted category for display consistency (first non-zero)
            primary_cat = 0
            primary_sub_letter = ""
            for code in (pred_codes or []):
                parsed = parse_category_and_subcategory(code)
                if int(parsed.get("category", 0)) != 0:
                    primary_cat = int(parsed.get("category", 0))
                    primary_sub_letter = str(parsed.get("subcategory", "") or "").lower()
                    break
            if primary_cat == 0 and pred_codes:
                parsed = parse_category_and_subcategory(pred_codes[0])
                primary_cat = int(parsed.get("category", 0))
                primary_sub_letter = str(parsed.get("subcategory", "") or "").lower()

            codes_str = ",".join(pred_codes) if pred_codes else ""
            gt_codes = rec.get("all_codes") or ([f"{gt_cat}{gt_subcat}".lower()] if gt_cat or gt_subcat else [])
            print(f"\tpredicted_codes={codes_str or ''}")
            if gt_codes:
                print(f"\tground_truth_codes={','.join(gt_codes)}")

            # Only compare subcategory when GT has hate (category != 0)
            if gt_cat != 0:
                # If one of the predicted codes matches GT category, use its letter for subcategory comparison
                pred_sub_letter = primary_sub_letter
                for code in (pred_codes or []):
                    m = re.match(r"^\s*([0-7])\s*([a-z])\s*$", code, flags=re.IGNORECASE)
                    if m and int(m.group(1)) == gt_cat:
                        pred_sub_letter = m.group(2).lower()
                        break

            # --- Detailed multi-label comparison (same as two-prompt) ---
            gt_codes_lower = [c.lower() for c in (gt_codes or [])]
            gt_cat_numbers = {re.match(r"^([0-7])", c).group(1) for c in gt_codes_lower if re.match(r"^([0-7])", c)}
            pred_with_sub = [c for c in pred_codes if re.match(r"^[0-7][a-z]$", c)]
            pred_without_sub = [c for c in pred_codes if re.match(r"^[0-7]$", c)]

            # 1) Exact matches first
            exact_hits = [code for code in pred_with_sub if code in gt_codes_lower]
            for code in exact_hits:
                print(f"\tMATCH: {code} (exact)")

            if not exact_hits:
                # 2) Partial matches by category letter
                first_unmatched_printed = False
                for code in pred_with_sub:
                    if code in gt_codes_lower:
                        print(f"\tMATCH: {code} (exact)")
                        continue
                    base_cat = code[0]
                    if base_cat in gt_cat_numbers:
                        print(f"\tPARTIAL: {code} (category {base_cat} present, subcategory differs)")
                    elif not first_unmatched_printed:
                        print(f"\tNO MATCH: {code} (category {base_cat} absent in ground truth)")
                        first_unmatched_printed = True

                # 3) Predictions without subcategory
                for code in pred_without_sub:
                    if code in gt_cat_numbers:
                        print(f"\tCATEGORY MATCH: {code} (subcategory not specified / differs)")
                    elif not first_unmatched_printed:
                        print(f"\tNO MATCH: {code} (category absent)")
                        first_unmatched_printed = True

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

    if two_prompt_evaluator.results:
        two_prompt_evaluator.save_results_to_excel("results/single_sentence_comparison_multiple_cat.xlsx", sheet_name="Two Prompts")
    if one_prompt_evaluator.results:
        one_prompt_evaluator.save_results_to_excel("results/single_sentence_comparison_multiple_cat.xlsx", sheet_name="One Prompt")

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
        excel_path="data/single_sentence_hate_speech.xlsx",
        # models=["llama", "qwen3"],  # ako je prazno, biće učitano iz models/models.json ili data/models.json
        # models=["llama"],  # ako je prazno, biće učitano iz models/models.json ili data/models.json
    )
