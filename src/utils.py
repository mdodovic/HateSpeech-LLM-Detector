"""
Utility functions for HateSpeech-LLM-Detector (Serbian)
"""
from typing import List, Dict, Any
import sys
import json
import csv
import re
import pandas as pd

# Constants
NO_HATE_SPEECH_CATEGORY = 0


def _map_serbian_category_code(code: str) -> int:
    """Map codes like '1a','3b','6a' to top-level category 1..7; '0' to 0."""
    if not code:
        return NO_HATE_SPEECH_CATEGORY
    code = str(code).strip().lower()
    if code == '0':
        return 0
    m = re.match(r"^(\d)", code)
    if m:
        num = int(m.group(1))
        if 0 <= num <= 7:
            return num
    # Try extracting a leading integer from strings like '2 homofobija'
    m2 = re.match(r"^(\d)\b", code)
    if m2:
        num = int(m2.group(1))
        if 0 <= num <= 7:
            return num
    return NO_HATE_SPEECH_CATEGORY


def load_excel_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load dataset from an Excel file like data/hate_speech_labeled_samples.xlsx.

    Expected columns (robust):
    - 'Text' (or 'text') for the sentence
    - 'Labelar1' for binary (0 means no hate; non-zero/filled means hate)
    - 'Id category' codes like 1a,3b,6a... or 'Category' like '2 homofobija'

    Returns list of dicts: {text, has_hate_speech: bool, category: int}
    """
    df = pd.read_excel(filepath)
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        text = str(row.get('Text') or row.get('text') or '').strip()
        if not text:
            continue
        # Binary label
        lab = row.get('Labelar1')
        has_hate = None
        if lab is not None and str(lab).strip() != '':
            try:
                has_hate = int(lab) != 0
            except Exception:
                has_hate = str(lab).strip().lower() not in {'0', 'false', 'ne', ''}
        # Category id
        cat_code = row.get('Id category')
        if cat_code is None or str(cat_code).strip() == '':
            cat_code = row.get('Category')
        category = _map_serbian_category_code(str(cat_code) if cat_code is not None else '')
        if has_hate is None:
            has_hate = category != NO_HATE_SPEECH_CATEGORY
        records.append({
            'text': text,
            'has_hate_speech': bool(has_hate),
            'category': int(category),
        })
    return records


def validate_dataset(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate dataset structure and return statistics"""
    errors: List[str] = []
    warnings: List[str] = []

    if not data:
        errors.append("Dataset is empty")
        return {"valid": False, "errors": errors, "warnings": warnings, "statistics": {"total_samples": 0}}

    required_fields = ['text', 'has_hate_speech', 'category']

    for i, sample in enumerate(data):
        for field in required_fields:
            if field not in sample:
                errors.append(f"Sample {i} missing field: {field}")
        if 'text' in sample and not isinstance(sample['text'], str):
            errors.append(f"Sample {i} 'text' is not a string")
        if 'has_hate_speech' in sample and not isinstance(sample['has_hate_speech'], bool):
            warnings.append(f"Sample {i} 'has_hate_speech' should be bool")
        if 'category' in sample:
            try:
                cat = int(sample['category'])
                if cat < 0 or cat > 7:
                    warnings.append(f"Sample {i} 'category' should be 0-7")
            except Exception:
                warnings.append(f"Sample {i} 'category' is not an int")

    total_samples = len(data)
    stats = {
        "total_samples": total_samples,
        "hate_speech_count": sum(1 for s in data if s.get('has_hate_speech', False)),
        "no_hate_count": sum(1 for s in data if not s.get('has_hate_speech', False)),
        "category_distribution": {},
    }
    for cid in range(8):
        count = sum(1 for s in data if s.get('category') == cid)
        if count > 0:
            stats['category_distribution'][cid] = count

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings, "statistics": stats}


if __name__ == "__main__":
    # Minimal CLI to quickly show dataset info
    if len(sys.argv) < 3:
        print("Usage:\n  python -m src.utils excel <path> | json <path> | csv <path>")
        sys.exit(1)
    kind, path = sys.argv[1], sys.argv[2]
    if kind == 'excel':
        data = load_excel_dataset(path)
    elif kind == 'json':
        data = load_json_dataset(path)
    elif kind == 'csv':
        data = load_csv_dataset(path)
    else:
        print(f"Unknown kind: {kind}")
        sys.exit(1)
    info = validate_dataset(data)
    print(info)
