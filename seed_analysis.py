"""
Run and aggregate zero-shot Ollama LLM experiments across fixed generation seeds.

The script has two separate phases:
  1. run       - writes one Excel file per seed/model/prompt strategy
  2. aggregate - reads completed per-run Excel files and computes mean +/- std

This makes long experiments resumable. You can stop after any completed run and
still aggregate the partial results that already exist.

Example:
    python seed_analysis.py --mode run --models llama qwen3
    python seed_analysis.py --mode aggregate
    python seed_analysis.py --mode all --models llama mistral deepseek phi3 qwen3
"""

import argparse
import contextlib
import io
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from single_sentence_run import (
    one_prompt_evaluation_model_on_records as single_sentence_one_prompt,
    two_prompts_evaluation_model_on_records as single_sentence_two_prompts,
)
from full_text_run import one_prompt_evaluation_model_on_records as full_text_one_prompt
from full_text_run_ensemble import perform_ensemble as full_text_ensemble
from src.utils import build_model_tags, load_excel_dataset, load_excel_full_text_dataset


DEFAULT_SEEDS = [42, 123, 2024, 3407, 271828]
DEFAULT_DATASET = "data/single_sentence_hate_speech_no_offenses.xlsx"
DEFAULT_RUN_DIR = "results/seed_analysis_runs"
DEFAULT_OUTPUT = "results/seed_analysis_zero_shot_summary.xlsx"
DEFAULT_COMBINED_OUTPUT = "results/seed_analysis_all_summary.xlsx"
EXPERIMENT_DATASETS = {
    "single_sentence": "data/single_sentence_hate_speech_no_offenses.xlsx",
    "full_text": "data/paragraph_hate_speech_no_offenses.xlsx",
    "full_text_ensemble": "data/paragraph_hate_speech_no_offenses.xlsx",
}
EXPERIMENT_PROMPTS = {
    "single_sentence": ["two_prompts", "one_prompt"],
    "full_text": ["one_prompt"],
    "full_text_ensemble": ["ensemble"],
}
EXPERIMENT_RUN_DIRS = {
    "single_sentence": "results/seed_analysis_runs",
    "full_text": "results/seed_analysis_full_text_runs",
    "full_text_ensemble": "results/seed_analysis_full_text_ensemble_runs",
}
EXPERIMENT_OUTPUTS = {
    "single_sentence": "results/seed_analysis_zero_shot_summary.xlsx",
    "full_text": "results/seed_analysis_full_text_summary.xlsx",
    "full_text_ensemble": "results/seed_analysis_full_text_ensemble_summary.xlsx",
}


def _flatten_metrics(
    *,
    experiment: str,
    model_name: str,
    model_tag: str,
    seed: int,
    prompt_type: str,
    result: Dict,
) -> List[Dict]:
    rows: List[Dict] = []
    per_sample = result.get("per_sample", {})
    task_specs = [
        ("binary", "binary_metrics", "y_true_binary"),
        ("category", "category_metrics", "y_true_category"),
        ("subcategory", "subcategory_metrics", "y_true_subcategory"),
    ]

    for task, result_key, y_true_key in task_specs:
        metrics = result.get(result_key, {}) or {}
        n_samples = len(per_sample.get(y_true_key, []))
        for metric, value in metrics.items():
            rows.append({
                "experiment": experiment,
                "model": model_name,
                "model_tag": model_tag,
                "seed": seed,
                "prompt_type": prompt_type,
                "task": task,
                "metric": metric,
                "value": float(value),
                "n_samples": n_samples,
            })

    multi = result.get("multi_label_metrics", {}) or {}
    for metric in ("multi_category_accuracy", "multi_subcategory_accuracy"):
        if metric in multi:
            rows.append({
                "experiment": experiment,
                "model": model_name,
                "model_tag": model_tag,
                "seed": seed,
                "prompt_type": prompt_type,
                "task": "multi_label",
                "metric": metric,
                "value": float(multi[metric]),
                "n_samples": int(multi.get("multi_samples", 0)),
            })

    return rows


def _flatten_predictions(
    *,
    experiment: str,
    model_name: str,
    model_tag: str,
    seed: int,
    prompt_type: str,
    result: Dict,
) -> List[Dict]:
    rows: List[Dict] = []
    per_sample = result.get("per_sample", {})
    task_specs = [
        ("binary", "y_true_binary", "y_pred_binary"),
        ("category", "y_true_category", "y_pred_category"),
        ("subcategory", "y_true_subcategory", "y_pred_subcategory"),
    ]

    for task, y_true_key, y_pred_key in task_specs:
        y_true = per_sample.get(y_true_key, [])
        y_pred = per_sample.get(y_pred_key, [])
        for idx, (yt, yp) in enumerate(zip(y_true, y_pred), start=1):
            rows.append({
                "experiment": experiment,
                "model": model_name,
                "model_tag": model_tag,
                "seed": seed,
                "prompt_type": prompt_type,
                "task": task,
                "sample_index": idx,
                "y_true": yt,
                "y_pred": yp,
            })

    return rows


def _mean_std_summary(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["experiment", "model", "model_tag", "prompt_type", "task", "metric"]
    summary = (
        per_seed_df
        .groupby(group_cols, as_index=False)
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            min=("value", "min"),
            max=("value", "max"),
            n_seeds=("seed", "nunique"),
            n_samples=("n_samples", "max"),
        )
    )
    summary["std"] = summary["std"].fillna(0.0)
    for col in ("mean", "std", "min", "max"):
        summary[col] = summary[col].astype(float)
    summary["mean_pm_std"] = summary.apply(
        lambda row: f"{row['mean']:.4f} +/- {row['std']:.4f}",
        axis=1,
    )
    return summary


def _paper_table(summary: pd.DataFrame, task: str) -> pd.DataFrame:
    subset = summary[
        (summary["task"] == task)
        & (summary["metric"].isin(["accuracy", "f1"]))
    ].copy()
    if subset.empty:
        return subset

    rows = []
    for _, row in subset.iterrows():
        rows.append({
            "model": row["model"],
            "experiment": row["experiment"],
            "model_tag": row["model_tag"],
            "prompt_type": row["prompt_type"],
            "metric": row["metric"],
            "mean": row["mean"],
            "std": row["std"],
            "mean +/- std": row["mean_pm_std"],
            "n_seeds": row["n_seeds"],
            "n_samples": row["n_samples"],
        })

    return pd.DataFrame(rows).sort_values(["experiment", "prompt_type", "model", "metric"])


def _parse_models(models: List[str] | None) -> Dict[str, str]:
    requested = models if models else None
    return build_model_tags(requested)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def _run_file_path(run_dir: Path, experiment: str, model_name: str, model_tag: str, prompt_type: str, seed: int) -> Path:
    file_name = (
        f"experiment_{_safe_name(experiment)}__seed_{int(seed)}__model_{_safe_name(model_name)}"
        f"__tag_{_safe_name(model_tag)}__prompt_{_safe_name(prompt_type)}.xlsx"
    )
    return run_dir / file_name


def _write_run_workbook(
    *,
    path: Path,
    metadata: Dict,
    metric_rows: List[Dict] | None = None,
    prediction_rows: List[Dict] | None = None,
    run_log: str | None = None,
    error: Exception | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        pd.DataFrame([metadata]).to_excel(writer, sheet_name="Metadata", index=False)
        if metric_rows:
            pd.DataFrame(metric_rows).to_excel(writer, sheet_name="Metrics", index=False)
        if prediction_rows:
            pd.DataFrame(prediction_rows).to_excel(writer, sheet_name="Predictions", index=False)
        if run_log:
            chunks = [run_log[i:i + 30000] for i in range(0, len(run_log), 30000)]
            pd.DataFrame({
                "chunk": list(range(1, len(chunks) + 1)),
                "text": chunks,
            }).to_excel(writer, sheet_name="RunLog", index=False)
        if error is not None:
            pd.DataFrame([{
                **metadata,
                "error": repr(error),
            }]).to_excel(writer, sheet_name="Error", index=False)


class _RunHeartbeat:
    def __init__(self, label: str, interval_s: int = 30):
        self.label = label
        self.interval_s = interval_s
        self.started = time.perf_counter()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        print(f"Started: {self.label}")
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=1)

    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            elapsed = self.elapsed_s()
            print(f"Still running ({elapsed/60:.1f} min): {self.label}", flush=True)


def run_seed_calls(
    *,
    experiment: str,
    dataset_path: str,
    models: List[str] | None,
    seeds: Iterable[int],
    debug: int,
    run_dir: str,
    prompt_types: List[str],
    overwrite: bool,
) -> None:
    if experiment == "single_sentence":
        records = load_excel_dataset(dataset_path)
    elif experiment in {"full_text", "full_text_ensemble"}:
        records = load_excel_full_text_dataset(dataset_path)
    else:
        raise ValueError(f"Unsupported experiment: {experiment}")

    if debug > 0 and len(records) > debug:
        records = records[:debug]

    model_tags = _parse_models(models)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in seeds]

    print(f"Experiment: {experiment}")
    print(f"Dataset   : {dataset_path} ({len(records)} records)")
    print(f"Models : {', '.join(f'{name}={tag}' for name, tag in model_tags.items())}")
    print(f"Seeds  : {', '.join(str(seed) for seed in seeds)}")
    print(f"Run dir: {run_path}")

    if experiment == "full_text_ensemble":
        run_specs = [("ensemble", "+".join(model_tags.values()), "ensemble")]
    else:
        run_specs = [
            (model_name, model_tag, prompt_type)
            for model_name, model_tag in model_tags.items()
            for prompt_type in prompt_types
        ]

    for seed in seeds:
        for model_name, model_tag, prompt_type in run_specs:
            run_file = _run_file_path(run_path, experiment, model_name, model_tag, prompt_type, seed)
            metadata = {
                "experiment": experiment,
                "dataset": dataset_path,
                "debug": debug,
                "n_records": len(records),
                "model": model_name,
                "model_tag": model_tag,
                "seed": seed,
                "prompt_type": prompt_type,
                "status": "pending",
            }
            if run_file.exists() and not overwrite:
                print(f"\nSkipping existing: {run_file}")
                continue

            print(f"\nRunning experiment={experiment} seed={seed} model={model_name} ({model_tag}) prompt={prompt_type}")
            print(f"Writing per-run workbook: {run_file}")
            log_buffer = io.StringIO()
            try:
                label = f"{experiment} | seed={seed} | model={model_name} | prompt={prompt_type}"
                with _RunHeartbeat(label) as heartbeat:
                    with contextlib.redirect_stdout(log_buffer):
                        if experiment == "single_sentence" and prompt_type == "two_prompts":
                            result = single_sentence_two_prompts(model_tag, records, seed=seed)
                        elif experiment == "single_sentence" and prompt_type == "one_prompt":
                            result = single_sentence_one_prompt(model_tag, records, seed=seed)
                        elif experiment == "full_text" and prompt_type == "one_prompt":
                            result = full_text_one_prompt(model_tag, records, seed=seed)
                        elif experiment == "full_text_ensemble" and prompt_type == "ensemble":
                            result = full_text_ensemble(
                                excel_path=dataset_path,
                                models=models or [],
                                debug=debug,
                                output_path=None,
                                seed=seed,
                            )
                        else:
                            raise ValueError(f"Unsupported experiment/prompt_type: {experiment}/{prompt_type}")
                    elapsed_s = heartbeat.elapsed_s()

                metric_rows = _flatten_metrics(
                    experiment=experiment,
                    model_name=model_name,
                    model_tag=model_tag,
                    seed=seed,
                    prompt_type=prompt_type,
                    result=result,
                )
                prediction_rows = _flatten_predictions(
                    experiment=experiment,
                    model_name=model_name,
                    model_tag=model_tag,
                    seed=seed,
                    prompt_type=prompt_type,
                    result=result,
                )
                metadata["status"] = "ok"
                _write_run_workbook(
                    path=run_file,
                    metadata=metadata,
                    metric_rows=metric_rows,
                    prediction_rows=prediction_rows,
                    run_log=log_buffer.getvalue(),
                )
                print(f"Completed in {elapsed_s/60:.1f} min: {run_file}", flush=True)
            except Exception as exc:
                metadata["status"] = "error"
                _write_run_workbook(path=run_file, metadata=metadata, run_log=log_buffer.getvalue(), error=exc)
                print(f"  ERROR: {exc}")


def _read_completed_runs(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    prediction_frames = []
    metadata_frames = []
    error_frames = []

    run_path = Path(run_dir)
    for path in sorted(run_path.glob("*.xlsx")):
        try:
            xls = pd.ExcelFile(path)
            meta = None
            if "Metadata" in xls.sheet_names:
                meta = pd.read_excel(path, sheet_name="Metadata")
                meta["run_file"] = str(path)
                metadata_frames.append(meta)
            if "Metrics" in xls.sheet_names:
                metrics = pd.read_excel(path, sheet_name="Metrics")
                if "experiment" not in metrics.columns:
                    metrics["experiment"] = (
                        str(meta["experiment"].iloc[0])
                        if meta is not None and "experiment" in meta.columns and not meta.empty
                        else "single_sentence"
                    )
                metrics["run_file"] = str(path)
                metric_frames.append(metrics)
            if "Predictions" in xls.sheet_names:
                preds = pd.read_excel(path, sheet_name="Predictions")
                if "experiment" not in preds.columns:
                    preds["experiment"] = (
                        str(meta["experiment"].iloc[0])
                        if meta is not None and "experiment" in meta.columns and not meta.empty
                        else "single_sentence"
                    )
                preds["run_file"] = str(path)
                prediction_frames.append(preds)
            if "Error" in xls.sheet_names:
                err = pd.read_excel(path, sheet_name="Error")
                err["run_file"] = str(path)
                error_frames.append(err)
        except Exception as exc:
            error_frames.append(pd.DataFrame([{
                "run_file": str(path),
                "error": repr(exc),
            }]))

    metrics_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    metadata_df = pd.concat(metadata_frames, ignore_index=True) if metadata_frames else pd.DataFrame()
    errors_df = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()
    return metrics_df, predictions_df, metadata_df, errors_df


def aggregate_seed_results(
    *,
    run_dir: str,
    output_path: str,
) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_seed_df, per_sample_df, metadata_df, errors_df = _read_completed_runs(run_dir)

    if per_seed_df.empty:
        with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
            metadata_df.to_excel(writer, sheet_name="RunMetadata", index=False)
            if not errors_df.empty:
                errors_df.to_excel(writer, sheet_name="Errors", index=False)
        raise RuntimeError(f"No successful Metrics sheets found under {run_dir}; wrote available metadata/errors to {out_path}.")

    summary_df = _mean_std_summary(per_seed_df)
    table_binary = _paper_table(summary_df, "binary")
    table_category = _paper_table(summary_df, "category")
    table_subcategory = _paper_table(summary_df, "subcategory")

    with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
        metadata_df.to_excel(writer, sheet_name="RunMetadata", index=False)
        per_seed_df.to_excel(writer, sheet_name="PerSeedMetrics", index=False)
        summary_df.to_excel(writer, sheet_name="MeanStdSummary", index=False)
        table_binary.to_excel(writer, sheet_name="XI_Binary", index=False)
        table_category.to_excel(writer, sheet_name="XII_Category", index=False)
        table_subcategory.to_excel(writer, sheet_name="XIII_Subcategory", index=False)
        per_sample_df.to_excel(writer, sheet_name="PerSeedPredictions", index=False)
        if not errors_df.empty:
            errors_df.to_excel(writer, sheet_name="Errors", index=False)

    print(f"\nAggregated {per_seed_df['run_file'].nunique()} completed run file(s).")
    print(f"Saved summary workbook to: {out_path}")
    if not errors_df.empty:
        print(f"Found {len(errors_df)} error row(s); see the Errors sheet.")


def aggregate_multiple_seed_results(
    *,
    run_dirs: List[str],
    output_path: str,
) -> None:
    metric_frames = []
    prediction_frames = []
    metadata_frames = []
    error_frames = []

    for run_dir in run_dirs:
        metrics_df, predictions_df, metadata_df, errors_df = _read_completed_runs(run_dir)
        if not metrics_df.empty:
            metric_frames.append(metrics_df)
        if not predictions_df.empty:
            prediction_frames.append(predictions_df)
        if not metadata_df.empty:
            metadata_frames.append(metadata_df)
        if not errors_df.empty:
            error_frames.append(errors_df)

    per_seed_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    per_sample_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    metadata_df = pd.concat(metadata_frames, ignore_index=True) if metadata_frames else pd.DataFrame()
    errors_df = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if per_seed_df.empty:
        with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
            metadata_df.to_excel(writer, sheet_name="RunMetadata", index=False)
            if not errors_df.empty:
                errors_df.to_excel(writer, sheet_name="Errors", index=False)
        raise RuntimeError(f"No successful Metrics sheets found under: {', '.join(run_dirs)}")

    summary_df = _mean_std_summary(per_seed_df)
    table_binary = _paper_table(summary_df, "binary")
    table_category = _paper_table(summary_df, "category")
    table_subcategory = _paper_table(summary_df, "subcategory")

    with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
        metadata_df.to_excel(writer, sheet_name="RunMetadata", index=False)
        per_seed_df.to_excel(writer, sheet_name="PerSeedMetrics", index=False)
        summary_df.to_excel(writer, sheet_name="MeanStdSummary", index=False)
        table_binary.to_excel(writer, sheet_name="XI_Binary", index=False)
        table_category.to_excel(writer, sheet_name="XII_Category", index=False)
        table_subcategory.to_excel(writer, sheet_name="XIII_Subcategory", index=False)
        per_sample_df.to_excel(writer, sheet_name="PerSeedPredictions", index=False)
        if not errors_df.empty:
            errors_df.to_excel(writer, sheet_name="Errors", index=False)

    print(f"\nAggregated combined results from {len(run_dirs)} run dir(s).")
    print(f"Saved combined summary workbook to: {out_path}")


def run_all_experiments(
    *,
    mode: str,
    models: List[str] | None,
    seeds: Iterable[int],
    debug: int,
    output_path: str,
    overwrite: bool,
) -> None:
    experiments = ["single_sentence", "full_text", "full_text_ensemble"]

    for experiment in experiments:
        dataset = EXPERIMENT_DATASETS[experiment]
        run_dir = EXPERIMENT_RUN_DIRS[experiment]
        output = EXPERIMENT_OUTPUTS[experiment]
        prompt_types = EXPERIMENT_PROMPTS[experiment]

        print("\n" + "#" * 70)
        print(f"Experiment batch: {experiment}")
        print("#" * 70)

        if mode in ("run", "all"):
            run_seed_calls(
                experiment=experiment,
                dataset_path=dataset,
                models=models,
                seeds=seeds,
                debug=debug,
                run_dir=run_dir,
                prompt_types=prompt_types,
                overwrite=overwrite,
            )

        if mode in ("aggregate", "all"):
            aggregate_seed_results(
                run_dir=run_dir,
                output_path=output,
            )

    if mode in ("aggregate", "all"):
        aggregate_multiple_seed_results(
            run_dirs=[EXPERIMENT_RUN_DIRS[experiment] for experiment in experiments],
            output_path=output_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeat zero-shot LLM experiments across Ollama generation seeds.",
    )
    parser.add_argument("--mode", choices=["run", "aggregate", "all"], default="all",
                        help="run = write per-call files; aggregate = summarize existing files; all = run then aggregate.")
    parser.add_argument("--experiment", choices=["single_sentence", "full_text", "full_text_ensemble", "all"], default="single_sentence")
    parser.add_argument("--dataset", default=None,
                        help="Dataset path. Default depends on --experiment.")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model names from models/models.json. Default: all configured models.")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--debug", type=int, default=548,
                        help="Limit number of records; use -1 or 0 for the full loaded dataset.")
    parser.add_argument("--run_dir", default=DEFAULT_RUN_DIR,
                        help="Directory containing one Excel workbook per seed/model/prompt run.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt_types", nargs="*", default=None,
                        choices=["two_prompts", "one_prompt", "ensemble"],
                        help="Prompt strategies. Default depends on --experiment.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run calls even if their per-run Excel workbook already exists.")
    args = parser.parse_args()
    if args.experiment == "all":
        if args.dataset is not None:
            raise ValueError("--dataset can only be used with one specific --experiment, not --experiment all.")
        if args.prompt_types is not None:
            raise ValueError("--prompt_types can only be used with one specific --experiment, not --experiment all.")
        output_path = (
            args.output
            if args.output != DEFAULT_OUTPUT
            else DEFAULT_COMBINED_OUTPUT
        )
        run_all_experiments(
            mode=args.mode,
            models=args.models,
            seeds=args.seeds,
            debug=args.debug,
            output_path=output_path,
            overwrite=args.overwrite,
        )
        return

    dataset = args.dataset or EXPERIMENT_DATASETS[args.experiment]
    prompt_types = args.prompt_types or EXPERIMENT_PROMPTS[args.experiment]

    if args.mode in ("run", "all"):
        run_seed_calls(
            experiment=args.experiment,
            dataset_path=dataset,
            models=args.models,
            seeds=args.seeds,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_types=prompt_types,
            overwrite=args.overwrite,
        )

    if args.mode in ("aggregate", "all"):
        aggregate_seed_results(
            run_dir=args.run_dir,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
