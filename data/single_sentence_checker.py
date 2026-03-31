import argparse
import pandas as pd
import re
from pathlib import Path


DEFAULT_FILE = "single_sentence_hate_speech.xlsx"


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without dropping punctuation."""
    if pd.isna(text) or not str(text).strip():
        return []
    s = str(text).strip()
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

    tokens = []
    i = 0
    while i < len(raw):
        if raw[i] == '{':
            j = raw.index('}', i)
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
            while j < len(raw) and raw[j] not in (',', '{'):
                j += 1
            token = raw[i:j].strip()
            if token or token == '0':
                tokens.append(token)
            i = j

    return tokens


def process_excel(file_path: Path, sample_filter: str | int | None = None, only_not_ok: bool = False) -> None:
    print(f"Processing: {file_path}\n")
    df = pd.read_excel(file_path)

    # Detect columns
    id_col = None
    text_col = None
    cat_col = None
    for c in df.columns:
        cl = str(c).lower()
        if id_col is None and (cl == 'id' or cl.startswith('id') or cl.endswith(' id')):
            id_col = c
        if text_col is None and ('text' in cl or 'sentence' in cl or 'content' in cl):
            text_col = c
        if cat_col is None and ('categor' in cl or 'label' in cl or 'class' in cl):
            cat_col = c

    # Fallback guesses
    if id_col is None and len(df.columns) >= 1:
        id_col = df.columns[0]
    if text_col is None and len(df.columns) >= 2:
        text_col = df.columns[1]
    if cat_col is None and len(df.columns) >= 3:
        cat_col = df.columns[2]

    total = 0
    ok_count = 0
    multi_count = 0
    empty_count = 0

    for _, row in df.iterrows():
        sample_id = row.get(id_col, _)
        if sample_filter is not None and str(sample_id) != str(sample_filter):
            continue
        text = row.get(text_col, '')
        categories_raw = row.get(cat_col, '')

        sentences = split_sentences(text)
        cats = split_categories(categories_raw)

        status = 'OK' if len(sentences) == 1 else f'MULTI({len(sentences)})'
        if len(sentences) == 0:
            status = 'EMPTY'

        total += 1
        if status == 'OK':
            ok_count += 1
        elif status == 'EMPTY':
            empty_count += 1
        else:
            multi_count += 1

        if only_not_ok and status == 'OK':
            continue

        print(f"sample: {sample_id}")
        print(f"category_count={len(cats)} sentence_count={len(sentences)} status={status}")
        if cats:
            print("category:", ', '.join(cats))
        else:
            print("category: NONE")

        if len(sentences) == 1:
            print(f"sentence: {sentences[0]}")
        elif len(sentences) > 1:
            print("WARNING: Expected single sentence, found multiple:")
            for i, sent in enumerate(sentences, 1):
                print(f"  [{i}] {sent}")
        else:
            print("No sentence text present.")
        print('-' * 80)

    print(f"\nDone. Total: {total} | OK: {ok_count} | MULTI: {multi_count} | EMPTY: {empty_count}")


def gather_excel_files(arg_path: str | None) -> list[Path]:
    data_dir = Path(__file__).parent
    if arg_path:
        p = Path(arg_path)
        if not p.exists():
            p = data_dir / arg_path
        if p.exists():
            if p.is_file() and p.suffix.lower() == '.xlsx' and not p.name.startswith('~$'):
                return [p]
            if p.is_dir():
                return sorted([f for f in p.glob('*.xlsx') if not f.name.startswith('~$')])

    default = data_dir / DEFAULT_FILE
    if default.exists():
        return [default]

    return sorted([f for f in data_dir.glob('*.xlsx') if not f.name.startswith('~$')])


def main():
    parser = argparse.ArgumentParser(description='Check single-sentence samples and flag multi-sentence rows.')
    parser.add_argument('-f', '--file', dest='file', help='Excel file or directory; defaults to all .xlsx in data/', default=None)
    parser.add_argument('-s', '--sample', dest='sample', help='Filter to a single sample ID', default=None)
    parser.add_argument('-n', '--not_ok', dest='not_ok', action='store_true', help='Print only not-OK samples (MULTI or EMPTY)')
    args = parser.parse_args()

    excel_files = gather_excel_files(args.file)
    if not excel_files:
        print('No Excel files found to process.')
        return

    for excel_file in excel_files:
        try:
            process_excel(excel_file, sample_filter=args.sample, only_not_ok=args.not_ok)
        except Exception as e:
            print(f'Error processing {excel_file}: {e}')


if __name__ == '__main__':
    main()
