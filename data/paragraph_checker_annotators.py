import argparse
import pandas as pd
import re
from pathlib import Path

DEFAULT_FILE = "access_paragraph_hate_speech_with_offenses.xlsx"

def split_sentences(text: str) -> list[str]:
    """Split text into sentences without dropping punctuation."""
    if pd.isna(text) or not str(text).strip():
        return []

    s = str(text).strip()
    # Split on sentence-ending punctuation (.!?…) optionally followed by more dots,
    # optional trailing comma/semicolon, and optional whitespace, before the start
    # of a new sentence. \s* (not \s+) so "word.Next" with no space still splits.
    # [,;\s]* handles cases like "..., NextSentence"
    pattern = "(?<=[.!?…])\\.*[,;\\s]*(?=(?:[A-Za-zČĆŠĐŽčćšđž\\u0400-\\u04FF0-9@#:'\"""„''()]|[\\u2600-\\u26FF\\u2700-\\u27BF\\U0001F1E6-\\U0001F1FF\\U0001F300-\\U0001F5FF\\U0001F600-\\U0001F64F\\U0001F680-\\U0001F6FF\\U0001F900-\\U0001F9FF\\U0001FA70-\\U0001FAFF]))"
    parts = re.split(pattern, s)
    return [p.strip() for p in parts if p and p.strip()]


def split_categories(categories: str) -> list[str]:
    """Split categories string by comma, respecting {…} groups as atomic tokens."""
    if pd.isna(categories) or categories is None:
        return []
    raw = str(categories).strip()
    if not raw:
        return []

    # Tokenize: curly-brace groups stay together, bare values split on comma
    tokens = []
    i = 0
    while i < len(raw):
        if raw[i] == '{':
            j = raw.index('}', i)
            tokens.append(raw[i:j + 1].strip())
            i = j + 1
            # skip trailing comma / whitespace
            while i < len(raw) and raw[i] in (',', ' '):
                i += 1
        elif raw[i] == ',':
            i += 1
            # skip whitespace after comma
            while i < len(raw) and raw[i] == ' ':
                i += 1
        else:
            j = i
            while j < len(raw) and raw[j] not in (',', '{'):
                j += 1
            token = raw[i:j].strip()
            if token or token == '0':
                tokens.append(token)
            i = j

    return tokens


def resolve_columns(df: pd.DataFrame) -> tuple:
    """Resolve ID, Text, and annotator columns from the DataFrame.

    Returns (id_col, text_col, annotator_cols_dict) where
    annotator_cols_dict maps display-name -> column-name.
    """
    id_col = None
    text_col = None
    annotator_cols = {}

    for c in df.columns:
        cl = str(c).lower()
        if id_col is None and (cl == 'id' or cl.startswith('id')):
            id_col = c
        elif text_col is None and ('text' in cl or 'content' in cl):
            text_col = c
        elif 'source' in cl:
            continue  # skip source column
        elif cl.startswith('unnamed'):
            continue  # skip empty/unnamed columns
        else:
            # Treat remaining columns as annotator columns
            annotator_cols[str(c)] = c

    # Fallbacks
    if id_col is None:
        id_col = df.columns[0]
    if text_col is None:
        text_col = df.columns[1]

    return id_col, text_col, annotator_cols


def process_excel(file_path: Path, sample_filter: str | int | None = None, only_mismatch: bool = False, ignore_empty: bool = False) -> None:
    """Process Excel file and print sentence:category pairs per sample for every annotator."""
    print(f"Processing: {file_path}\n")

    df = pd.read_excel(file_path)
    id_col, text_col, annotator_cols = resolve_columns(df)

    if not annotator_cols:
        print("ERROR: No annotator columns found.")
        return

    print(f"Annotators found: {', '.join(annotator_cols.keys())}\n")

    for _, row in df.iterrows():
        sample_id = row.get(id_col, _)
        if sample_filter is not None and str(sample_id) != str(sample_filter):
            continue

        text = row.get(text_col, '')
        sentences = split_sentences(text)
        n_sent = len(sentences)

        # Parse categories for each annotator
        ann_cats = {}
        for name, col in annotator_cols.items():
            ann_cats[name] = split_categories(row.get(col, ''))

        # Check for mismatches (skip completely empty annotators if ignore_empty)
        mismatches = {}
        for name, cats in ann_cats.items():
            if ignore_empty and len(cats) == 0:
                mismatches[name] = False
            else:
                mismatches[name] = len(cats) != n_sent
        any_mismatch = any(mismatches.values())

        if only_mismatch and not any_mismatch:
            continue

        # Header
        print(f"sample: {sample_id}")
        counts_str = "  ".join(f"{name}={len(cats)}" for name, cats in ann_cats.items())
        print(f"sentences={n_sent}  {counts_str}")

        if any_mismatch:
            bad = [name for name, m in mismatches.items() if m]
            print(f"WARNING: count mismatch for: {', '.join(bad)}")

        # Determine column widths for alignment
        max_len = max(n_sent, *(len(c) for c in ann_cats.values()))
        ann_names = list(ann_cats.keys())
        col_widths = {name: max(len(name), *(len(c) for c in cats), 7) if cats else max(len(name), 7) for name, cats in ann_cats.items()}

        # Header row
        header = "     ".join(f"{name:>{col_widths[name]}}" for name in ann_names)
        print(f"{'':>4}  {header}   sentence")

        for i in range(max_len):
            idx_str = f"{i + 1:>3}."
            parts = []
            for name in ann_names:
                cats = ann_cats[name]
                val = cats[i] if i < len(cats) else 'MISSING'
                parts.append(f"{val:>{col_widths[name]}}")
            sent = sentences[i] if i < n_sent else '<NO_SENTENCE>'
            print(f"{idx_str}  {'     '.join(parts)}   {sent}")

        print("-" * 100)

    # Summary of mismatches
    print("\nSummary of mismatches:")
    for name in annotator_cols.keys():
        mismatch_count = 0
        for _, row in df.iterrows():
            cats = split_categories(str(row.get(annotator_cols[name], '')))
            if ignore_empty and len(cats) == 0:
                continue
            if len(split_sentences(str(row.get(text_col, '')))) != len(cats):
                mismatch_count += 1
        print(f"{name}: {mismatch_count} mismatches")

    # Tiebreaker check: when annotator1 and annotator2 disagree, is annotator3 present?
    ann_names = list(annotator_cols.keys())
    if len(ann_names) >= 3:
        a1_name, a2_name, a3_name = ann_names[0], ann_names[1], ann_names[2]
        a1_col, a2_col, a3_col = annotator_cols[a1_name], annotator_cols[a2_name], annotator_cols[a3_name]

        disagree_total = 0
        tiebreaker_missing = 0
        missing_samples = []

        for _, row in df.iterrows():
            sample_id = row.get(id_col, _)
            if sample_filter is not None and str(sample_id) != str(sample_filter):
                continue

            sentences = split_sentences(str(row.get(text_col, '')))
            cats1 = split_categories(str(row.get(a1_col, '')))
            cats2 = split_categories(str(row.get(a2_col, '')))
            cats3 = split_categories(str(row.get(a3_col, '')))

            n = min(len(sentences), len(cats1), len(cats2))
            has_missing = False
            for i in range(n):
                if cats1[i] != cats2[i]:
                    disagree_total += 1
                    c3 = cats3[i] if i < len(cats3) else None
                    if c3 is None or c3 == '' or c3 == 'MISSING':
                        tiebreaker_missing += 1
                        has_missing = True
            if has_missing:
                missing_samples.append(sample_id)

        print(f"\nTiebreaker check ({a1_name} vs {a2_name}, tiebreaker: {a3_name}):")
        print(f"  Sentences where {a1_name} != {a2_name}: {disagree_total}")
        print(f"  Of those, {a3_name} missing: {tiebreaker_missing}")
        if missing_samples:
            print(f"  Samples with missing tiebreaker ({len(missing_samples)}): {missing_samples}")

def gather_excel_files(arg_path: str | None) -> list[Path]:
    """Return a list of Excel files based on CLI argument or fallback."""
    data_dir = Path(__file__).parent

    if arg_path:
        p = Path(arg_path)
        if not p.exists():
            p = data_dir / arg_path
        if p.exists():
            if p.is_file() and p.suffix.lower() == ".xlsx" and not p.name.startswith("~$"):
                return [p]
            if p.is_dir():
                return sorted([f for f in p.glob("*.xlsx") if not f.name.startswith("~$")])

    # Default to the annotator dataset
    default = data_dir / DEFAULT_FILE
    if default.exists():
        return [default]

    return sorted([f for f in data_dir.glob("*.xlsx") if not f.name.startswith("~$")])


def main():
    parser = argparse.ArgumentParser(description="Check sentence/category alignment for multiple annotators")
    parser.add_argument("-f", "--file", dest="file", help=f"Excel file or directory (default: {DEFAULT_FILE})", default=None)
    parser.add_argument("-s", "--sample", dest="sample", help="Filter to a single sample ID", default=None)
    parser.add_argument("-only_mismatch", "--only_mismatch", "-m", dest="only_mismatch", action="store_true", help="Print only samples with sentence/category count mismatches")
    parser.add_argument("-ignore_empty", "--ignore_empty", "-ie", dest="ignore_empty", action="store_true", help="Ignore annotators with completely empty categories (don't count as mismatch)")
    args = parser.parse_args()

    excel_files = gather_excel_files(args.file)

    if not excel_files:
        print("No Excel files found to process.")
        return

    for excel_file in excel_files:
        try:
            process_excel(excel_file, sample_filter=args.sample, only_mismatch=args.only_mismatch, ignore_empty=args.ignore_empty)
        except Exception as e:
            print(f"Error processing {excel_file}: {e}")


if __name__ == "__main__":
    main()
