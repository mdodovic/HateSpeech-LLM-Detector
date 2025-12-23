import argparse
import pandas as pd
import re
from pathlib import Path

def split_sentences(text: str) -> list[str]:
    """Split text into sentences without dropping punctuation.

    Uses a regex that looks for whitespace after a sentence ender
    . ! ? and before the start of a new sentence (capital letter,
    digit or opening quote). Works well for SR/HR text too.
    """
    if pd.isna(text) or not str(text).strip():
        return []

    s = str(text).strip()
    # Keep punctuation with the sentence; split on boundary between sentences
    # Supports Latin (incl. Serbian diacritics) and Cyrillic blocks
    # Treat emojis as valid "next characters" after punctuation so we split even when
    # a sentence ends with e.g. "... ! <emoji> Next sentence". We include common emoji
    # Unicode ranges alongside letters, digits, and quotes in the lookahead.
    #
    # Note: Python's built-in `re` doesn't support Unicode properties like \p{Emoji},
    # so we explicitly list the primary emoji blocks:
    # - U+2600–U+26FF  Misc Symbols
    # - U+2700–U+27BF  Dingbats
    # - U+1F1E6–U+1F1FF Regional Indicator Symbols (flags)
    # - U+1F300–U+1F5FF Misc Symbols and Pictographs
    # - U+1F600–U+1F64F Emoticons
    # - U+1F680–U+1F6FF Transport and Map Symbols
    # - U+1F900–U+1F9FF Supplemental Symbols and Pictographs
    # - U+1FA70–U+1FAFF Symbols & Pictographs Extended-A
    pattern = "(?<=[.!?…])\\s+(?=(?:[A-Za-zČĆŠĐŽčćšđž\\u0400-\\u04FF0-9'\"“”„‘’()]|[\\u2600-\\u26FF\\u2700-\\u27BF\\U0001F1E6-\\U0001F1FF\\U0001F300-\\U0001F5FF\\U0001F600-\\U0001F64F\\U0001F680-\\U0001F6FF\\U0001F900-\\U0001F9FF\\U0001FA70-\\U0001FAFF]))"
    parts = re.split(pattern, s)
    # Clean up extra whitespace
    return [p.strip() for p in parts if p and p.strip()]

def split_categories(categories: str) -> list[str]:
    """Split categories string by comma and trim blanks."""
    if pd.isna(categories) or categories is None:
        return []
    raw = str(categories).strip()
    if not raw:
        return []
    # Only comma-separated; ignore empty trailing commas
    return [c.strip() for c in raw.split(',') if c.strip() or c.strip() == '0']

def process_excel(file_path: Path, sample_filter: str | int | None = None, only_mismatch: bool = False) -> None:
    """Process Excel file and print sentence:category pairs per sample.

    Expected columns (case-insensitive): ID, Text, Category
    """
    print(f"Processing: {file_path}\n")

    df = pd.read_excel(file_path)

    # Resolve column names flexibly
    cols_lower = {str(c).lower(): c for c in df.columns}
    id_col = None
    text_col = None
    cat_col = None

    for c in df.columns:
        cl = str(c).lower()
        if id_col is None and (cl == 'id' or cl.endswith(' id') or cl.startswith('id')):
            id_col = c
        if text_col is None and ('text' in cl or 'content' in cl):
            text_col = c
        if cat_col is None and ('categor' in cl or 'label' in cl or 'class' in cl):
            cat_col = c

    # Fallbacks if headers differ
    if text_col is None and len(df.columns) >= 2:
        text_col = df.columns[1]
    if cat_col is None and len(df.columns) >= 3:
        cat_col = df.columns[2]
    if id_col is None:
        id_col = df.columns[0]

    for _, row in df.iterrows():
        sample_id = row.get(id_col, _)
        # If sample filter provided, skip non-matching rows
        if sample_filter is not None and str(sample_id) != str(sample_filter):
            continue
        text = row.get(text_col, '')
        categories = row.get(cat_col, '')

        sentences = split_sentences(text)
        cats = split_categories(categories)

        mismatch = (len(sentences) != len(cats))
        if only_mismatch and not mismatch:
            # Skip clean samples when only mismatches are requested
            continue

        print(f"sample: {sample_id}")
        print(f"sentences={len(sentences)} categories={len(cats)}")
        if mismatch:
            print("WARNING: counts mismatch; showing aligned pairs and extras below.")

        # Print aligned pairs
        max_len = max(len(sentences), len(cats))
        for i in range(max_len):
            sent = sentences[i] if i < len(sentences) else '<NO_SENTENCE>'
            cat = cats[i] if i < len(cats) else 'MISSING'
            print(f"{cat:2} : {sent}")
        print("-" * 80)

def gather_excel_files(arg_path: str | None) -> list[Path]:
    """Return a list of Excel files based on CLI argument or fallback.

    - If `arg_path` is a file: return just that file.
    - If `arg_path` is a directory: return all .xlsx inside it.
    - If missing or not found: return all .xlsx in this script's folder.
    """
    data_dir = Path(__file__).parent

    if arg_path:
        p = Path(arg_path)
        # If a relative path was provided, try relative to data_dir first
        if not p.exists():
            p = (data_dir / arg_path)
        if p.exists():
            if p.is_file() and p.suffix.lower() == ".xlsx" and not p.name.startswith("~$"):
                return [p]
            if p.is_dir():
                return sorted([f for f in p.glob("*.xlsx") if not f.name.startswith("~$")])

    # Fallback to all .xlsx in data directory
    return sorted([f for f in data_dir.glob("*.xlsx") if not f.name.startswith("~$")])

def main():
    parser = argparse.ArgumentParser(description="Print cat:sentence pairs from Excel data")
    parser.add_argument("-f", "--file", dest="file", help="Excel file or directory to process; defaults to all .xlsx in data/", default=None)
    parser.add_argument("-s", "--sample", dest="sample", help="Filter to a single sample ID", default=None)
    parser.add_argument("-only_mismatch", "--only_mismatch", "-m", dest="only_mismatch", action="store_true", help="Print only samples where sentence/category counts mismatch")
    args = parser.parse_args()

    excel_files = gather_excel_files(args.file)

    if not excel_files:
        print("No Excel files found to process.")
        return

    for excel_file in excel_files:
        try:
            process_excel(excel_file, sample_filter=args.sample, only_mismatch=args.only_mismatch)
        except Exception as e:
            print(f"Error processing {excel_file}: {e}")

if __name__ == "__main__":
    main()
