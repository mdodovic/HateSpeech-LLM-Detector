"""
Calculate mean +/- std from completed seed_analysis per-run Excel files.

This script does not call any LLMs. It only reads *.xlsx files created by
seed_analysis.py under the run directories and writes summary workbooks.

Examples:
    python calculate_seed_stats.py
    python calculate_seed_stats.py --run_dirs results/seed_analysis_runs
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_RUN_DIRS = [
    "results/seed_analysis_runs",
    "results/seed_analysis_full_text_runs",
    "results/seed_analysis_full_text_ensemble_runs",
]
DEFAULT_OUTPUT = "results/seed_analysis_all_summary.xlsx"
MODEL_ORDER = {
    "llama": 0,
    "mistral": 1,
    "deepseek": 2,
    "phi": 3,
    "phi3": 3,
    "qwen": 4,
    "qwen3": 4,
    "qwen35": 5,
    "ensemble": 99,
}


def read_completed_runs(run_dirs: List[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    prediction_frames = []
    metadata_frames = []
    error_frames = []

    for run_dir in run_dirs:
        run_path = Path(run_dir)
        if not run_path.exists():
            print(f"[skip] Missing run directory: {run_path}")
            continue

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


def mean_std_summary(per_seed_df: pd.DataFrame) -> pd.DataFrame:
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


def paper_table(summary: pd.DataFrame, task: str) -> pd.DataFrame:
    subset = summary[
        (summary["task"] == task)
        & (summary["metric"].isin(["accuracy", "f1"]))
    ].copy()
    if subset.empty:
        return subset

    rows = []
    for _, row in subset.iterrows():
        rows.append({
            "experiment": row["experiment"],
            "model": row["model"],
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


def _metric_cell(rows: pd.DataFrame, task: str, metric: str, precision: int) -> str:
    sub = rows[(rows["task"] == task) & (rows["metric"] == metric)]
    if sub.empty:
        return "-"
    row = sub.iloc[0]
    if pd.isna(row["mean"]):
        return "-"
    return f"{row['mean']:.{precision}f} +/- {row['std']:.{precision}f}"


def _sort_models(models: List[str]) -> List[str]:
    return sorted(models, key=lambda m: (MODEL_ORDER.get(str(m).lower(), 50), str(m).lower()))


def _print_table_section(
    summary: pd.DataFrame,
    *,
    title: str,
    filters: List[tuple[str, str]],
    precision: int,
) -> None:
    parts = []
    for experiment, prompt_type in filters:
        parts.append(summary[
            (summary["experiment"] == experiment)
            & (summary["prompt_type"] == prompt_type)
        ])
    table_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    if table_df.empty:
        print("No completed rows found for this table.")
        return

    headers = [
        "Model",
        "Detection Acc",
        "Detection F1",
        "Category Acc",
        "Category F1",
        "Subcat Acc",
        "Subcat F1",
        "Seeds",
        "Experiment",
    ]
    print(
        f"{headers[0]:<12} {headers[1]:<17} {headers[2]:<17} "
        f"{headers[3]:<17} {headers[4]:<17} {headers[5]:<17} "
        f"{headers[6]:<17} {headers[7]:<7} {headers[8]}"
    )
    print("-" * 120)

    group_cols = ["experiment", "prompt_type", "model"]
    groups = {
        key: group
        for key, group in table_df.groupby(group_cols, sort=False)
    }
    ordered_keys = sorted(
        groups.keys(),
        key=lambda key: (
            [f for f in filters].index((key[0], key[1])) if (key[0], key[1]) in filters else 99,
            MODEL_ORDER.get(str(key[2]).lower(), 50),
            str(key[2]).lower(),
        ),
    )

    for experiment, prompt_type, model in ordered_keys:
        rows = groups[(experiment, prompt_type, model)]
        seed_counts = rows["n_seeds"].dropna().unique().tolist()
        seeds = int(seed_counts[0]) if seed_counts else "-"
        label = str(model)
        if experiment.endswith("ensemble") or prompt_type == "ensemble":
            label = "ensemble*"

        print(
            f"{label:<12} "
            f"{_metric_cell(rows, 'binary', 'accuracy', precision):<17} "
            f"{_metric_cell(rows, 'binary', 'f1', precision):<17} "
            f"{_metric_cell(rows, 'category', 'accuracy', precision):<17} "
            f"{_metric_cell(rows, 'category', 'f1', precision):<17} "
            f"{_metric_cell(rows, 'subcategory', 'accuracy', precision):<17} "
            f"{_metric_cell(rows, 'subcategory', 'f1', precision):<17} "
            f"{seeds!s:<7} "
            f"{experiment}/{prompt_type}"
        )


def print_paper_ready_tables(summary: pd.DataFrame, precision: int = 3) -> None:
    print("\n\nPAPER-READY MEAN +/- STD TABLES")
    print("Use these values to replace the metric cells in Tables XI-XIII.")

    _print_table_section(
        summary,
        title="TABLE XI - Single-stage sentence-level results (single_sentence / one_prompt)",
        filters=[("single_sentence", "one_prompt")],
        precision=precision,
    )
    _print_table_section(
        summary,
        title="TABLE XII - Two-stage sentence-level results (single_sentence / two_prompts)",
        filters=[("single_sentence", "two_prompts")],
        precision=precision,
    )
    _print_table_section(
        summary,
        title="TABLE XIII - Paragraph-level results (full_text + full_text_ensemble)",
        filters=[("full_text", "one_prompt"), ("full_text_ensemble", "ensemble")],
        precision=precision,
    )

    print("\nNote: ensemble* rows are printed only when completed ensemble run files are present.")


def write_summary(output_path: str, run_dirs: List[str], print_tables: bool = True, precision: int = 3) -> None:
    per_seed_df, per_sample_df, metadata_df, errors_df = read_completed_runs(run_dirs)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if per_seed_df.empty:
        with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
            metadata_df.to_excel(writer, sheet_name="RunMetadata", index=False)
            if not errors_df.empty:
                errors_df.to_excel(writer, sheet_name="Errors", index=False)
        raise RuntimeError(f"No completed Metrics sheets found. Wrote available metadata/errors to {out_path}")

    summary_df = mean_std_summary(per_seed_df)
    table_binary = paper_table(summary_df, "binary")
    table_category = paper_table(summary_df, "category")
    table_subcategory = paper_table(summary_df, "subcategory")

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

    print(f"Read {per_seed_df['run_file'].nunique()} completed run file(s).")
    print(f"Saved summary workbook to: {out_path}")
    if not errors_df.empty:
        print(f"Found {len(errors_df)} error row(s); see the Errors sheet.")
    if print_tables:
        print_paper_ready_tables(summary_df, precision=precision)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate mean +/- std from completed seed_analysis per-run Excel files.",
    )
    parser.add_argument("--run_dirs", nargs="*", default=DEFAULT_RUN_DIRS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--no_print", action="store_true",
                        help="Only write the Excel workbook; do not print paper-ready tables.")
    parser.add_argument("--precision", type=int, default=3,
                        help="Decimal places for printed mean +/- std values.")
    args = parser.parse_args()

    write_summary(args.output, args.run_dirs, print_tables=not args.no_print, precision=args.precision)


if __name__ == "__main__":
    main()
