import argparse
import pandas as pd
import re
from pathlib import Path
from typing import List


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without dropping punctuation."""
    if pd.isna(text) or not str(text).strip():
        return []

    s = str(text).strip()
    pattern = "(?<=[.!?…])\\.*[,;\\s]*(?=(?:[A-Za-zČĆŠĐŽčćšđž\\u0400-\\u04FF0-9@#:'\"""„''()]|[\\u2600-\\u26FF\\u2700-\\u27BF\\U0001F1E6-\\U0001F1FF\\U0001F300-\\U0001F5FF\\U0001F600-\\U0001F64F\\U0001F680-\\U0001F6FF\\U0001F900-\\U0001F9FF\\U0001FA70-\\U0001FAFF]))"
    parts = re.split(pattern, s)
    return [p.strip() for p in parts if p and p.strip()]


def split_categories(categories: str) -> list[str]:
    """Split categories string by comma, respecting {…} and (…) groups as atomic tokens."""
    if pd.isna(categories) or categories is None:
        return []
    raw = str(categories).strip()
    if not raw:
        return []

    tokens = []
    i = 0
    while i < len(raw):
        if raw[i] in ('{', '('):
            close = '}' if raw[i] == '{' else ')'
            j = raw.index(close, i)
            tokens.append(raw[i:j + 1].strip())
            i = j + 1
            while i < len(raw) and raw[i] in (',', ' '):
                i += 1
        elif raw[i] == ',':
            i += 1
            while i < len(raw) and raw[i] == ' ':
                i += 1
        else:
            j = i
            while j < len(raw) and raw[j] not in (',', '{', '('):
                j += 1
            token = raw[i:j].strip()
            if token or token == '0':
                tokens.append(token)
            i = j

    return tokens


def normalize_category(cat: str, remove_offense: bool = False) -> str:
    """Normalize a category token: strip {}/(), replace ; with comma, optionally replace U with 0."""
    result = cat.replace('{', '').replace('}', '').replace('(', '').replace(')', '').replace(';', ',')
    if remove_offense:
        parts = [p.strip() for p in result.split(',')]
        parts = ['0' if p == 'U' else p for p in parts]
        result = ','.join(parts)
    return result


def resolve_columns(df: pd.DataFrame):
    """Resolve ID, Text, and LLM column from the DataFrame.

    Returns tuple: (id_col, text_col, llm_col)
    """
    id_col = None
    text_col = None
    llm_col = None

    for c in df.columns:
        cl = str(c).lower()
        if id_col is None and (cl == 'id' or cl.startswith('id')):
            id_col = c
        elif text_col is None and ('text' in cl or 'content' in cl or 'tekst' in cl):
            text_col = c
        elif llm_col is None and 'llm' in cl:
            llm_col = c

    if id_col is None:
        id_col = df.columns[0]
    if text_col is None:
        id_col = df.columns[1]
    if llm_col is None:
        raise ValueError("No LLM column found in the Excel file. Expected a column with 'LLM' in its name.")

    return id_col, text_col, llm_col


def build_paragraph_dataset(df: pd.DataFrame, id_col, text_col, llm_col, remove_offense: bool = False) -> pd.DataFrame:
    """Build paragraph-level dataset with ID, Text, Category from LLM column."""
    rows: List[dict] = []
    for _, row in df.iterrows():
        sample_id = row.get(id_col, _)
        text = row.get(text_col, '')
        raw = str(row.get(llm_col, '')).strip()

        if pd.isna(row.get(llm_col, None)) or raw == '' or raw == 'nan':
            category = ''
        else:
            category = raw

        if remove_offense:
            category = re.sub(r'(?<![A-Za-z])U(?![A-Za-z])', '0', category)

        rows.append({
            'ID': sample_id,
            'Text': text,
            'Category': category,
        })
    return pd.DataFrame(rows)


def expand_to_sentences(df: pd.DataFrame, id_col, text_col, llm_col, remove_offense: bool = False) -> pd.DataFrame:
    """Expand each sample into one row per sentence using LLM column for categories."""
    rows: List[dict] = []

    for _, row in df.iterrows():
        sample_id = row.get(id_col, _)
        text = row.get(text_col, '')

        sentences = split_sentences(text)
        cats = split_categories(row.get(llm_col, ''))

        for i, sent in enumerate(sentences):
            cat_val = cats[i] if i < len(cats) else ''
            if cat_val and remove_offense:
                cat_val = re.sub(r'(?<![A-Za-z])U(?![A-Za-z])', '0', cat_val)
            rows.append({
                'ID': sample_id,
                'Text': sent,
                'Category': cat_val,
            })

    return pd.DataFrame(rows)


def gather_excel_file(arg_path: str | None) -> Path | None:
    """Return a single Excel file path based on CLI argument or fallback."""
    data_dir = Path(__file__).parent

    if arg_path:
        p = Path(arg_path)
        if not p.exists():
            p = (data_dir / arg_path)
        if p.exists() and p.is_file() and p.suffix.lower() == ".xlsx" and not p.name.startswith("~$"):
            return p

    default = data_dir / "access_paragraph_hate_speech_with_offenses.xlsx"
    if default.exists():
        return default

    candidates = sorted([f for f in data_dir.glob("*.xlsx") if not f.name.startswith("~$")])
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Extract ID, Text, and LLM column from dataset")
    parser.add_argument("-f", "--file", dest="file", help="Input Excel file (default: data/access_paragraph_hate_speech_with_offenses.xlsx)", default=None)
    parser.add_argument("-o", "--output", dest="output", help="Output per-sentence Excel path (default: data/single_sentence_hate_speech_llm.xlsx)", default="data/single_sentence_hate_speech_llm.xlsx")
    parser.add_argument("-p", "--paragraph_output", dest="paragraph_output", help="Output paragraph Excel path (default: data/paragraph_hate_speech_llm.xlsx)", default="data/paragraph_hate_speech_llm.xlsx")
    parser.add_argument("--remove_offense", "-ro", dest="remove_offense", action="store_true", help="Replace U (offense) category with 0 (no hate speech)")
    args = parser.parse_args()

    excel_file = gather_excel_file(args.file)

    if not excel_file:
        print("No Excel file found to process.")
        return

    try:
        df = pd.read_excel(excel_file)
        id_col, text_col, llm_col = resolve_columns(df)
        print(f"LLM column found: {llm_col}")
        if args.remove_offense:
            print("Replacing U (offense) with 0")

        # Paragraph-level output
        para_df = build_paragraph_dataset(df, id_col, text_col, llm_col, remove_offense=args.remove_offense)
        para_path = Path(args.paragraph_output)
        para_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(para_path, engine="openpyxl") as writer:
            para_df.to_excel(writer, index=False, sheet_name="Paragraphs")
        print(f"Wrote paragraph dataset: {para_path} ({len(para_df)} rows)")

        # Per-sentence output
        out_df = expand_to_sentences(df, id_col, text_col, llm_col, remove_offense=args.remove_offense)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            out_df.to_excel(writer, index=False, sheet_name="Sentences")
        print(f"Wrote per-sentence dataset: {out_path} ({len(out_df)} rows)")
    except Exception as e:
        print(f"Error processing {excel_file}: {e}")


if __name__ == "__main__":
    main()
