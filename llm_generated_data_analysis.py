"""
Compare synthetic (Source == "LLM") and authentic paragraphs in the Serbian
hate-speech annotator dataset.

The script is read-only with respect to dataset files. It writes a Markdown
summary to synthetic_analysis.md by default.

Default input:
  data/access_paragraph_hate_speech_with_offenses.xlsx

Example:
  python llm_generated_data_analysis.py
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import pandas as pd


DEFAULT_INPUT = Path("data/access_paragraph_hate_speech_with_offenses.xlsx")
DEFAULT_OUTPUT = Path("synthetic_analysis.md")

SOURCE_SYNTHETIC = "LLM"
TEST_MAX_ID = 101

MAIN_CATEGORIES = [str(i) for i in range(1, 8)]
SUBCATEGORY_ORDER = [
    "1a",
    "1b",
    "1c",
    "2",
    "3a",
    "3b",
    "4a",
    "4b",
    "5",
    "6a",
    "6b",
    "6c",
    "7",
]


def resolve_input_path(path: Path) -> Path:
    """Resolve the input path and tolerate the common 'pffenses' typo."""
    if path.exists():
        return path

    fixed = Path(str(path).replace("pffenses", "offenses"))
    if fixed.exists():
        return fixed

    if DEFAULT_INPUT.exists():
        return DEFAULT_INPUT

    raise FileNotFoundError(f"Dataset not found: {path}")


def is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value)


def split_label_entries(raw: object) -> list[str]:
    """Split a paragraph annotation into one label entry per sentence.

    Commas separate sentences, except inside parentheses. Semicolons inside
    parentheses separate multiple labels for the same sentence.
    """
    if is_missing(raw):
        return []

    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []

    entries: list[str] = []
    buf: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
            buf.append(char)
        elif char == ")":
            depth = max(0, depth - 1)
            buf.append(char)
        elif char == "," and depth == 0:
            token = "".join(buf).strip()
            if token:
                entries.append(token)
            buf = []
        else:
            buf.append(char)

    token = "".join(buf).strip()
    if token:
        entries.append(token)
    return entries


def parse_codes(entry: object) -> list[str]:
    """Parse one sentence label entry into normalized codes."""
    if is_missing(entry):
        return ["0"]

    text = str(entry).strip().lower()
    if not text or text == "nan":
        return ["0"]

    if text.startswith("(") and text.endswith(")"):
        parts = [part.strip() for part in text[1:-1].split(";") if part.strip()]
    else:
        parts = [part.strip() for part in re.split(r"[;:]", text) if part.strip()]

    codes: list[str] = []
    for part in parts:
        if part == "u":
            codes.append("U")
            continue
        if part == "0":
            codes.append("0")
            continue
        match = re.match(r"^([1-7])\s*([a-z])?$", part)
        if match:
            codes.append(match.group(1) + (match.group(2) or ""))

    return codes or ["0"]


def main_category(code: str) -> str | None:
    match = re.match(r"^([1-7])", str(code).lower())
    return match.group(1) if match else None


def sentence_class(codes: Iterable[str]) -> str:
    """Map a possibly multi-label sentence to neutral/offensive/hate."""
    code_set = {str(code) for code in codes}
    if any(main_category(code) for code in code_set):
        return "hate"
    if "U" in code_set:
        return "offensive"
    return "neutral"


def final_entries(row: pd.Series, a1_col: str, a3_col: str | None) -> list[str]:
    """Use senior labels when present, otherwise fall back to Annotator1."""
    a1 = split_label_entries(row.get(a1_col, ""))
    a3 = split_label_entries(row.get(a3_col, "")) if a3_col else []
    n = max(len(a1), len(a3))
    result: list[str] = []
    for i in range(n):
        senior = a3[i].strip() if i < len(a3) else ""
        fallback = a1[i].strip() if i < len(a1) else "0"
        result.append(senior if senior else fallback)
    return result


def resolve_columns(df: pd.DataFrame) -> tuple[str, str, str, str | None, str | None, str]:
    columns = {str(col).strip().lower(): str(col) for col in df.columns}

    id_col = columns.get("id", "ID")
    text_col = columns.get("text", "Text")
    source_col = columns.get("source", "Source")

    a1_col = None
    a2_col = None
    a3_col = None
    for col in df.columns:
        lower = str(col).strip().lower()
        if "annotator1" in lower or "anotator1" in lower:
            a1_col = str(col)
        elif "annotator2" in lower or "anotator2" in lower:
            a2_col = str(col)
        elif "annotator3" in lower or "senior" in lower:
            a3_col = str(col)

    final_col = columns.get("category")
    if a1_col is None and final_col is None:
        raise ValueError("Could not find Annotator1 or Category labels.")
    if source_col not in df.columns:
        raise ValueError("Could not find Source column.")

    return id_col, text_col, source_col, a1_col or final_col, a2_col, a3_col, final_col or a1_col


def build_sentence_records(df: pd.DataFrame, id_col: str, source_col: str, a1_col: str, a3_col: str | None) -> list[dict]:
    records: list[dict] = []
    for _, row in df.iterrows():
        entries = final_entries(row, a1_col, a3_col)
        source = str(row.get(source_col, "")).strip()
        is_synthetic = source.upper() == SOURCE_SYNTHETIC
        paragraph_id = row.get(id_col)

        for sentence_index, entry in enumerate(entries, start=1):
            codes = parse_codes(entry)
            records.append(
                {
                    "paragraph_id": paragraph_id,
                    "source": source,
                    "subset": "Synthetic" if is_synthetic else "Authentic",
                    "sentence_index": sentence_index,
                    "entry": entry,
                    "codes": codes,
                    "class": sentence_class(codes),
                }
            )
    return records


def paragraph_stats(lengths: list[int]) -> dict[str, float | int]:
    return {
        "Paragraphs": len(lengths),
        "Sentences": int(sum(lengths)),
        "Mean": mean(lengths) if lengths else 0.0,
        "Median": median(lengths) if lengths else 0.0,
        "Min": min(lengths) if lengths else 0,
        "Max": max(lengths) if lengths else 0,
    }


def format_num(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def table(headers: list[str], rows: list[list[object]]) -> str:
    string_rows = [[format_num(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[i]) for row in string_rows))
        for i, header in enumerate(headers)
    ]
    header_line = " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers))
    sep_line = "-+-".join("-" * width for width in widths)
    body = [" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in string_rows]
    return "\n".join([header_line, sep_line, *body])


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(format_num(cell) for cell in row) + " |")
    return "\n".join(out)


def pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def class_distribution(records: list[dict], subset: str) -> list[list[object]]:
    subset_records = [record for record in records if record["subset"] == subset]
    total = len(subset_records)
    counts = Counter(record["class"] for record in subset_records)
    return [
        [subset, "neutral", counts["neutral"], pct(counts["neutral"], total)],
        [subset, "offensive", counts["offensive"], pct(counts["offensive"], total)],
        [subset, "hate", counts["hate"], pct(counts["hate"], total)],
    ]


def category_distribution(records: list[dict], subset: str) -> tuple[list[list[object]], int]:
    counts: Counter[str] = Counter()
    for record in records:
        if record["subset"] != subset:
            continue
        for code in record["codes"]:
            category = main_category(code)
            if category:
                counts[category] += 1

    total = sum(counts.values())
    rows = [[subset, category, counts[category], pct(counts[category], total)] for category in MAIN_CATEGORIES]
    return rows, total


def synthetic_subcategory_distribution(records: list[dict]) -> list[list[object]]:
    counts: Counter[str] = Counter()
    for record in records:
        if record["subset"] != "Synthetic":
            continue
        for code in record["codes"]:
            if main_category(code):
                counts[str(code).lower()] += 1

    total = sum(counts.values())
    return [[code, counts[code], pct(counts[code], total)] for code in SUBCATEGORY_ORDER]


def disagreement_rows(
    df: pd.DataFrame,
    id_col: str,
    source_col: str,
    a1_col: str | None,
    a2_col: str | None,
) -> list[list[object]] | None:
    if not a1_col or not a2_col:
        return None

    stats = {
        "Synthetic": {"total": 0, "different": 0},
        "Authentic": {"total": 0, "different": 0},
    }
    for _, row in df.iterrows():
        subset = (
            "Synthetic"
            if str(row.get(source_col, "")).strip().upper() == SOURCE_SYNTHETIC
            else "Authentic"
        )
        entries1 = split_label_entries(row.get(a1_col, ""))
        entries2 = split_label_entries(row.get(a2_col, ""))
        n = min(len(entries1), len(entries2))
        for i in range(n):
            codes1 = sorted(set(parse_codes(entries1[i])))
            codes2 = sorted(set(parse_codes(entries2[i])))
            stats[subset]["total"] += 1
            stats[subset]["different"] += int(codes1 != codes2)

    rows = []
    for subset in ["Synthetic", "Authentic"]:
        total = stats[subset]["total"]
        different = stats[subset]["different"]
        rows.append([subset, total, different, pct(different, total)])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Source == LLM synthetic paragraphs against authentic paragraphs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    df = pd.read_excel(input_path)
    id_col, _text_col, source_col, a1_col, a2_col, a3_col, _final_col = resolve_columns(df)

    is_synthetic = df[source_col].astype(str).str.strip().str.upper().eq(SOURCE_SYNTHETIC)
    records = build_sentence_records(df, id_col, source_col, a1_col, a3_col)

    paragraph_lengths = {}
    for subset_name, mask in [("Synthetic", is_synthetic), ("Authentic", ~is_synthetic)]:
        lengths = [
            len(final_entries(row, a1_col, a3_col))
            for _, row in df.loc[mask].iterrows()
        ]
        paragraph_lengths[subset_name] = lengths

    synthetic_in_test = int((is_synthetic & (pd.to_numeric(df[id_col], errors="coerce") <= TEST_MAX_ID)).sum())
    split_status = "PASS" if synthetic_in_test == 0 else "FAIL"

    size_headers = ["Subset", "Paragraphs", "Sentences"]
    size_rows = [
        ["Synthetic", int(is_synthetic.sum()), sum(paragraph_lengths["Synthetic"])],
        ["Authentic", int((~is_synthetic).sum()), sum(paragraph_lengths["Authentic"])],
    ]

    length_headers = ["Subset", "Mean", "Median", "Min", "Max"]
    length_rows = []
    for subset in ["Synthetic", "Authentic"]:
        stats = paragraph_stats(paragraph_lengths[subset])
        length_rows.append([subset, stats["Mean"], stats["Median"], stats["Min"], stats["Max"]])

    class_headers = ["Subset", "Class", "Count", "Percent"]
    class_rows = class_distribution(records, "Synthetic") + class_distribution(records, "Authentic")

    category_headers = ["Subset", "Category", "Label count", "Percent"]
    synthetic_cat_rows, synthetic_hate_labels = category_distribution(records, "Synthetic")
    authentic_cat_rows, authentic_hate_labels = category_distribution(records, "Authentic")
    category_rows = synthetic_cat_rows + authentic_cat_rows

    subcategory_headers = ["Subcategory", "Label count", "Percent"]
    subcategory_rows = synthetic_subcategory_distribution(records)

    disagreement = disagreement_rows(df, id_col, source_col, a1_col, a2_col)
    disagreement_headers = ["Subset", "Compared sentences", "Disagreements", "Disagreement percent"]

    lines: list[str] = []
    lines.append("# Synthetic vs. Authentic Data Analysis")
    lines.append("")
    lines.append(f"Input file: `{input_path}`")
    lines.append(f"Final labels: `{a3_col}` per sentence when present, otherwise `{a1_col}`.")
    lines.append("Sentence counts are based on annotation entries, matching the 8,029 labeled sentences.")
    lines.append("")
    lines.append("## Subset Size")
    lines.append(markdown_table(size_headers, size_rows))
    lines.append("")
    lines.append("## Split Placement")
    lines.append(
        f"{split_status}: synthetic paragraphs in held-out test split (ID <= {TEST_MAX_ID}) = {synthetic_in_test}"
    )
    lines.append("")
    lines.append("## Paragraph Length")
    lines.append(markdown_table(length_headers, length_rows))
    lines.append("")
    lines.append("## Sentence-Level Class Distribution")
    lines.append("Priority: hate if any 1-7 label is present, otherwise offensive if U is present, otherwise neutral.")
    lines.append(markdown_table(class_headers, class_rows))
    lines.append("")
    lines.append("## Hate Category Distribution")
    lines.append(
        "Counts are label-level: each hate label in a multi-label sentence is counted separately."
    )
    lines.append(
        f"Synthetic hate-label denominator = {synthetic_hate_labels}; authentic hate-label denominator = {authentic_hate_labels}."
    )
    lines.append(markdown_table(category_headers, category_rows))
    lines.append("")
    lines.append("## Synthetic Subcategory Distribution")
    lines.append(markdown_table(subcategory_headers, subcategory_rows))
    lines.append("")
    lines.append("## Annotator1 vs Annotator2 Disagreement")
    if disagreement is None:
        lines.append("Skipped: independent annotator columns were not found.")
    else:
        lines.append(markdown_table(disagreement_headers, disagreement))

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Synthetic vs. Authentic Data Analysis")
    print(f"Input file: {input_path}")
    print(f"Final labels: {a3_col} per sentence when present, otherwise {a1_col}.")
    print("Sentence counts are based on annotation entries, matching the 8,029 labeled sentences.")
    print()
    print("Subset size")
    print(table(size_headers, size_rows))
    print()
    print(f"Split placement: {split_status} - synthetic paragraphs in ID <= {TEST_MAX_ID}: {synthetic_in_test}")
    print()
    print("Paragraph length")
    print(table(length_headers, length_rows))
    print()
    print("Sentence-level class distribution")
    print(table(class_headers, class_rows))
    print()
    print("Hate category distribution")
    print(table(category_headers, category_rows))
    print()
    print("Synthetic subcategory distribution")
    print(table(subcategory_headers, subcategory_rows))
    print()
    print("Annotator1 vs Annotator2 disagreement")
    if disagreement is None:
        print("Skipped: independent annotator columns were not found.")
    else:
        print(table(disagreement_headers, disagreement))
    print()
    print(f"Markdown summary written to: {args.output}")


if __name__ == "__main__":
    main()
