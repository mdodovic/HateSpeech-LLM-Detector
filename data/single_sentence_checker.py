import argparse
import pandas as pd
import re
from pathlib import Path


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (Latin + Cyrillic) keeping punctuation."""
    if pd.isna(text) or not str(text).strip():
        return []
    s = str(text).strip()
    pattern = r"(?<=[.!?…])\s+(?=[A-Za-zČĆŠĐŽčćšđž\u0400-\u04FF0-9\"\“\”\„])"
    parts = re.split(pattern, s)
    return [p.strip() for p in parts if p and p.strip()]


def split_categories(categories: str) -> list[str]:
    if pd.isna(categories) or categories is None:
        return []
    raw = str(categories).strip()
    if not raw:
        return []
    # Comma separated; keep zeros
    return [c.strip() for c in raw.split(',') if c.strip() or c.strip() == '0']


def process_excel(file_path: Path, sample_filter: str | int | None = None) -> None:
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
    return sorted([f for f in Path(__file__).parent.glob('*.xlsx') if not f.name.startswith('~$')])


def main():
    parser = argparse.ArgumentParser(description='Check single-sentence samples and flag multi-sentence rows.')
    parser.add_argument('-f', '--file', dest='file', help='Excel file or directory; defaults to all .xlsx in data/', default=None)
    parser.add_argument('-s', '--sample', dest='sample', help='Filter to a single sample ID', default=None)
    args = parser.parse_args()

    excel_files = gather_excel_files(args.file)
    if not excel_files:
        print('No Excel files found to process.')
        return

    for excel_file in excel_files:
        try:
            process_excel(excel_file, sample_filter=args.sample)
        except Exception as e:
            print(f'Error processing {excel_file}: {e}')


if __name__ == '__main__':
    main()
