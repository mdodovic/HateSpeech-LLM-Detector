"""
Main script for Hate Speech Detection using Multiple LLMs (Ollama backend, Serbian only)
"""
import os
import json
import argparse
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

from llm_detector import LLMDetector
from evaluation import HateSpeechEvaluator
from examples.src.categories import get_category_prompt, HATE_SPEECH_CATEGORIES


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load dataset from JSON file"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def evaluate_model(model_name: str, dataset: List[Dict], categories_prompt: str, output_dir: str = "results") -> Dict:
    """Evaluate a single LLM on the dataset"""
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"{'='*60}\n")

    detector = LLMDetector(model_name)

    y_true_binary: List[bool] = []
    y_pred_binary: List[bool] = []
    y_true_category: List[int] = []
    y_pred_category: List[int] = []
    tokens_covered: List[int] = []
    total_tokens: List[int] = []
    detailed_results: List[Dict] = []

    for sample in tqdm(dataset, desc=f"Processing with {model_name}"):
        text = sample["text"]
        result = detector.analyze_text_complete(text, categories_prompt)

        y_true_binary.append(sample["has_hate_speech"])
        y_pred_binary.append(result["has_hate_speech"])
        y_true_category.append(sample["category"])
        y_pred_category.append(result["category"])
        tokens_covered.append(result["tokens_covered"])
        total_tokens.append(result["total_tokens"])

        detailed_results.append({
            "text": text,
            "true_has_hate": sample["has_hate_speech"],
            "pred_has_hate": result["has_hate_speech"],
            "true_category": sample["category"],
            "pred_category": result["category"],
            "hate_sentences": result["hate_sentences"],
            "tokens_covered": result["tokens_covered"],
            "total_tokens": result["total_tokens"],
            "binary_explanation": result["binary_explanation"],
            "category_explanation": result["category_explanation"],
        })

    evaluator = HateSpeechEvaluator()
    binary_metrics = evaluator.evaluate_binary_classification(y_true_binary, y_pred_binary)
    category_metrics = evaluator.evaluate_multiclass_classification(y_true_category, y_pred_category)
    token_metrics = evaluator.evaluate_token_coverage(tokens_covered, total_tokens)

    evaluator.print_results(model_name, binary_metrics, category_metrics, token_metrics)

    model_safe_name = model_name.replace("/", "_")
    results_df = pd.DataFrame(detailed_results)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f"{output_dir}/{model_safe_name}_detailed_results.csv", index=False)

    all_metrics = {
        "model": model_name,
        "binary_metrics": binary_metrics,
        "category_metrics": category_metrics,
        "token_metrics": token_metrics,
    }
    with open(f"{output_dir}/{model_safe_name}_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    report = evaluator.generate_classification_report(y_true_category, y_pred_category, HATE_SPEECH_CATEGORIES)
    with open(f"{output_dir}/{model_safe_name}_classification_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Hate Speech Detection using Multiple LLMs (Ollama)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--models", type=str, nargs='+', required=True, help="List of Ollama model tags to evaluate")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} samples")

    categories_prompt = get_category_prompt()

    all_results: Dict[str, Dict] = {}
    for model_name in args.models:
        try:
            results = evaluate_model(model_name, dataset, categories_prompt, args.output_dir)
            all_results[model_name] = results
        except Exception as e:
            print(f"\nError evaluating {model_name}: {str(e)}")
            continue

    if len(all_results) > 1:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        # Flatten top-level metrics for quick comparison
        rows = []
        for model_name, res in all_results.items():
            row = {"model": model_name}
            row.update({f"binary_{k}": v for k, v in res["binary_metrics"].items()})
            row.update({f"category_{k}": v for k, v in res["category_metrics"].items()})
            row.update({f"token_{k}": v for k, v in res["token_metrics"].items()})
            rows.append(row)
        comparison_df = pd.DataFrame(rows)
        print("\n", comparison_df.to_string(index=False))
        comparison_df.to_csv(f"{args.output_dir}/model_comparison.csv", index=False)

    print(f"\nAll results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
