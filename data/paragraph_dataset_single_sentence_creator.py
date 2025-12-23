import argparse
import pandas as pd
import re
from pathlib import Path
from typing import List

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

def resolve_columns(df: pd.DataFrame):
    """Resolve ID, Text, Category column names flexibly.

    Returns tuple: (id_col, text_col, cat_col)
    """
    id_col = None
    text_col = None
    cat_col = None

    for c in df.columns:
        cl = str(c).lower()
        if id_col is None and (cl == 'id' or cl.endswith(' id') or cl.startswith('id')):
            id_col = c
        if text_col is None and ('text' in cl or 'content' in cl or 'tekst' in cl):
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

    return id_col, text_col, cat_col


def expand_to_sentences(df: pd.DataFrame, id_col, text_col, cat_col) -> pd.DataFrame:
    """Expand each sample into multiple rows: one per sentence.

    Uses `split_sentences(text)` and `split_categories(category)`.
    If counts mismatch, still outputs all sentences; missing categories set to empty string.

    Output columns: ID, SentenceIndex, Sentence, Category
    """
    rows: List[dict] = []

    for _, row in df.iterrows():
        sample_id = row.get(id_col, _)
        text = row.get(text_col, '')
        categories = row.get(cat_col, '')

        sentences = split_sentences(text)
        cats = split_categories(categories)

        for i, sent in enumerate(sentences):
            cat_val = cats[i] if i < len(cats) else ''
            rows.append({
                'Sentence': sent,
                'Category': cat_val,
            })

    return pd.DataFrame(rows)

def gather_excel_file(arg_path: str | None) -> Path | None:
    """Return a single Excel file path based on CLI argument or fallback.

    - If `arg_path` is a file: return that file.
    - If missing or not found: return data/paragraph_hate_speech.xlsx if present,
      else the first .xlsx in this script's folder.
    """
    data_dir = Path(__file__).parent

    if arg_path:
        p = Path(arg_path)
        if not p.exists():
            p = (data_dir / arg_path)
        if p.exists() and p.is_file() and p.suffix.lower() == ".xlsx" and not p.name.startswith("~$"):
            return p

    default = data_dir / "paragraph_hate_speech.xlsx"
    if default.exists():
        return default

    candidates = sorted([f for f in data_dir.glob("*.xlsx") if not f.name.startswith("~$")])
    return candidates[0] if candidates else None

def main():
    parser = argparse.ArgumentParser(description="Expand full-text dataset to per-sentence Excel")
    parser.add_argument("-f", "--file", dest="file", help="Input Excel file (default: data/paragraph_hate_speech.xlsx)", default=None)
    parser.add_argument("-o", "--output", dest="output", help="Output Excel path (default: data/paragraph_new_line.xlsx)", default="data/paragraph_new_line.xlsx")
    args = parser.parse_args()

    excel_file = gather_excel_file(args.file)

    if not excel_file:
        print("No Excel file found to process.")
        return

    try:
        df = pd.read_excel(excel_file)
        id_col, text_col, cat_col = resolve_columns(df)
        out_df = expand_to_sentences(df, id_col, text_col, cat_col)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            out_df.to_excel(writer, index=False, sheet_name="Sentences")
        print(f"Wrote per-sentence dataset: {out_path}")
    except Exception as e:
        print(f"Error processing {excel_file}: {e}")

if __name__ == "__main__":
    main()
