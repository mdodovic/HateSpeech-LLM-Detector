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

    def evaluate_multiclass_classification(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
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
        """Generate detailed classification report for categories.

        Ensures the `labels` parameter matches `target_names` to avoid mismatches when
        some classes are absent in y_true/y_pred.
        """
        # Use explicit labels derived from the provided mapping to keep report consistent
        labels = sorted(category_names.keys())
        target_names = [category_names.get(i, f"Kategorija {i}") for i in labels]
        return classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)

    def generate_confusion_matrix(self, y_true: List[int], y_pred: List[int]):
        """Generate confusion matrix"""
        return confusion_matrix(y_true, y_pred)

    def save_results(self, model_name: str, res:Dict):
        """Save evaluation results for a model"""
        self.results.append({
            "model": model_name,
            "results":res
            }
        )


    def print_single_model_results(self, model_name: str):
        """Print results for a single model"""
        matching_results = [res for res in self.results if res.get("model") == model_name]
        if not matching_results:
            print(f"Nema rezultata za model: {model_name}")
            return
        res = matching_results[0]
        binary_metrics = res.get("binary_metrics", {})
        category_metrics = res.get("category_metrics", {})

        print(f"\nRezultati za model: {model_name}")

        if binary_metrics is not None:
            print("\n--- Zadatak 1: Binarna detekcija ---")
            for metric, value in binary_metrics.items():
                print(f"{metric.capitalize():15s}: {value:.4f}")

        if category_metrics is not None:
            print("\n--- Zadatak 2: Kategorizacija ---")
            for metric, value in category_metrics.items():
                print(f"{metric:20s}: {value:.4f}")

    def print_results(self, model_name: str = None):
        """Print formatted evaluation results"""
        if model_name is None and self.results:
            print("Rezultati za sve modele:")
            print("-" * 40)
            for res in self.results:
                self.print_single_model_results(res.get("model"))
            return
        
        self.print_single_model_results(model_name)

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten self.results into a single tidy DataFrame."""
        rows = []
        for item in self.results:
            model = item.get("model")
            res = item.get("results", {})
            row = {"model": model}
            # Flatten metrics: binary_metrics.*, category_metrics.*, subcategory_metrics.*
            for scope_key, metrics in res.items():
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        row[f"{scope_key}.{k}"] = v
            rows.append(row)
        self.df = pd.DataFrame(rows)
        return self.df

    def save_results_to_excel(self, output_path: str = None, verbose: bool = False, sheet_name: str = None):
        """Save comparison results to Excel file"""
        if not hasattr(self, 'df'):
            self.to_dataframe()        
        with pd.ExcelWriter(str(output_path), engine="openpyxl", mode="a", if_sheet_exists="new") as writer:
            self.df.to_excel(writer, index=False, sheet_name=sheet_name)
        if verbose:
            print(self.df)
        print(f"\nRezultati su sačuvani u: {output_path}")
