"""
Main script for Hate Speech Detection using Multiple LLMs
"""

import argparse
import json
from typing import List, Dict
from tqdm import tqdm
import pandas as pd

from llm_detector import LLMDetector
from evaluation import HateSpeechEvaluator
from categories import get_category_prompt, HATE_SPEECH_CATEGORIES


def load_dataset(dataset_path: str) -> List[Dict]:
    """
    Load dataset from JSON file
    
    Expected format:
    [
        {
            "text": "Example text...",
            "has_hate_speech": true/false,
            "category": 0-7
        },
        ...
    ]
    
    Args:
        dataset_path: Path to JSON dataset file
        
    Returns:
        List of dataset samples
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def evaluate_model(model_name: str, dataset: List[Dict], 
                  categories_prompt: str, output_dir: str = "results") -> Dict:
    """
    Evaluate a single LLM on the dataset
    
    Args:
        model_name: HuggingFace model name
        dataset: List of dataset samples
        categories_prompt: Description of categories
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"{'='*60}\n")
    
    # Initialize detector
    detector = LLMDetector(model_name)
    
    # Store predictions
    y_true_binary = []
    y_pred_binary = []
    y_true_category = []
    y_pred_category = []
    tokens_covered = []
    total_tokens = []
    detailed_results = []
    
    # Process each sample
    for sample in tqdm(dataset, desc=f"Processing with {model_name}"):
        text = sample["text"]
        
        # Get predictions
        result = detector.analyze_text_complete(text, categories_prompt)
        
        # Store for evaluation
        y_true_binary.append(sample["has_hate_speech"])
        y_pred_binary.append(result["has_hate_speech"])
        y_true_category.append(sample["category"])
        y_pred_category.append(result["category"])
        tokens_covered.append(result["tokens_covered"])
        total_tokens.append(result["total_tokens"])
        
        # Store detailed results
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
            "category_explanation": result["category_explanation"]
        })
    
    # Evaluate
    evaluator = HateSpeechEvaluator()
    
    binary_metrics = evaluator.evaluate_binary_classification(y_true_binary, y_pred_binary)
    category_metrics = evaluator.evaluate_multiclass_classification(y_true_category, y_pred_category)
    token_metrics = evaluator.evaluate_token_coverage(tokens_covered, total_tokens)
    
    # Print results
    evaluator.print_results(model_name, binary_metrics, category_metrics, token_metrics)
    
    # Save detailed results
    model_safe_name = model_name.replace("/", "_")
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{output_dir}/{model_safe_name}_detailed_results.csv", index=False)
    
    # Save metrics
    all_metrics = {
        "model": model_name,
        "binary_metrics": binary_metrics,
        "category_metrics": category_metrics,
        "token_metrics": token_metrics
    }
    
    with open(f"{output_dir}/{model_safe_name}_metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Generate and save classification report
    report = evaluator.generate_classification_report(
        y_true_category, y_pred_category, HATE_SPEECH_CATEGORIES
    )
    with open(f"{output_dir}/{model_safe_name}_classification_report.txt", 'w') as f:
        f.write(report)
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Hate Speech Detection using Multiple LLMs")
    parser.add_argument("--dataset", type=str, required=True, 
                       help="Path to dataset JSON file")
    parser.add_argument("--models", type=str, nargs='+', required=True,
                       help="List of HuggingFace model names to evaluate")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to run models on")
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} samples")
    
    # Get categories prompt
    categories_prompt = get_category_prompt()
    
    # Evaluate each model
    all_results = {}
    for model_name in args.models:
        try:
            results = evaluate_model(model_name, dataset, categories_prompt, args.output_dir)
            all_results[model_name] = results
        except Exception as e:
            print(f"\nError evaluating {model_name}: {str(e)}")
            continue
    
    # Compare models
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        evaluator = HateSpeechEvaluator()
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, results in all_results.items():
            row = {
                "Model": model_name,
                "Binary F1": results["binary_metrics"]["f1"],
                "Binary Accuracy": results["binary_metrics"]["accuracy"],
                "Category F1 (Macro)": results["category_metrics"]["f1_macro"],
                "Category Accuracy": results["category_metrics"]["accuracy"],
                "Mean Token Coverage": results["token_metrics"]["mean_coverage_ratio"]
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n", comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv(f"{args.output_dir}/model_comparison.csv", index=False)
    
    print(f"\nAll results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
