"""
Evaluation Metrics for Hate Speech Detection
"""

from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class HateSpeechEvaluator:
    """Evaluator for hate speech detection tasks"""

    def __init__(self):
        self.results = []

    def evaluate_binary_classification(self, y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
        """Evaluate binary hate speech detection (Task 1)"""
        y_true_int = [int(y) for y in y_true]
        y_pred_int = [int(y) for y in y_pred]
        return {
            "accuracy": accuracy_score(y_true_int, y_pred_int),
            "precision": precision_score(y_true_int, y_pred_int, zero_division=0),
            "recall": recall_score(y_true_int, y_pred_int, zero_division=0),
            "f1": f1_score(y_true_int, y_pred_int, zero_division=0),
        }

    def evaluate_multiclass_classification(self, y_true: List[int], y_pred: List[int], num_classes: int = 8) -> Dict[str, float]:
        """Evaluate multiclass hate speech categorization (Task 3)"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

    def evaluate_token_coverage(self, tokens_covered_list: List[int], total_tokens_list: List[int]) -> Dict[str, float]:
        """Evaluate token coverage for hate speech extraction (Task 2)"""
        coverage_ratios = [covered / total if total > 0 else 0 for covered, total in zip(tokens_covered_list, total_tokens_list)]
        return {
            "total_tokens_analyzed": int(sum(total_tokens_list)),
            "total_tokens_covered": int(sum(tokens_covered_list)),
            "mean_coverage_ratio": float(np.mean(coverage_ratios)) if coverage_ratios else 0.0,
            "median_coverage_ratio": float(np.median(coverage_ratios)) if coverage_ratios else 0.0,
            "std_coverage_ratio": float(np.std(coverage_ratios)) if coverage_ratios else 0.0,
            "min_coverage_ratio": float(np.min(coverage_ratios)) if coverage_ratios else 0.0,
            "max_coverage_ratio": float(np.max(coverage_ratios)) if coverage_ratios else 0.0,
        }

    def generate_classification_report(self, y_true: List[int], y_pred: List[int], category_names: Dict[int, str]) -> str:
        """Generate detailed classification report for categories"""
        target_names = [category_names.get(i, f"Category {i}") for i in range(len(category_names))]
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

    def generate_confusion_matrix(self, y_true: List[int], y_pred: List[int]):
        """Generate confusion matrix"""
        return confusion_matrix(y_true, y_pred)

    def compare_models(self, results_dict: Dict[str, Dict]):
        """Compare results across multiple models"""
        rows = []
        for model_name, results in results_dict.items():
            row = {"model": model_name}
            row.update(results)
            rows.append(row)
        return pd.DataFrame(rows)

    def print_results(self, model_name: str, binary_metrics: Dict, category_metrics: Dict, token_metrics: Dict):
        """Print formatted evaluation results"""
        print(f"\n{'='*60}")
        print(f"Results for: {model_name}")
        print(f"{'='*60}")

        print("\n--- Task 1: Binary Hate Speech Detection ---")
        for metric, value in binary_metrics.items():
            print(f"{metric.capitalize():15s}: {value:.4f}")

        print("\n--- Task 3: Hate Speech Categorization ---")
        for metric, value in category_metrics.items():
            print(f"{metric:20s}: {value:.4f}")

        print("\n--- Task 2: Token Coverage Statistics ---")
        for metric, value in token_metrics.items():
            if isinstance(value, float):
                print(f"{metric:20s}: {value:.4f}")
            else:
                print(f"{metric:20s}: {value}")
