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
    """Map category codes to top-level class (0–7).

    Accepts values like:
    - "0" → 0 (no hate)
    - "1a", "3b", "6a" → 1, 3, 6
    - "2 homofobija" → 2
    Any malformed or missing value maps to 0.
    """
    if code is None:
        return NO_HATE_SPEECH_CATEGORY
    code = str(code).strip().lower()
    if code == "":
        return NO_HATE_SPEECH_CATEGORY
    if code == "0":
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


def _extract_text(row: Dict[str, Any]) -> str:
    """Robustly extract the text field from a row/dict (supports Text/text/Tekst)."""
    for key in ("text", "Text", "Tekst"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # Fallback: try to find the first non-empty string
    for k, v in row.items():
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _normalize_record(text: str, category_value: Any, has_hate_value: Any = None) -> Dict[str, Any]:
    """Create a normalized record dict using new schema.

    - category_value can be "0", "1a", "2 homofobija", 3, etc.
    - has_hate_value is optional; if missing, inferred from category != 0
    """
    category = _map_serbian_category_code(category_value)
    has_hate = bool(has_hate_value) if isinstance(has_hate_value, bool) else (category != NO_HATE_SPEECH_CATEGORY)
    return {
        "text": text,
        "has_hate_speech": has_hate,
        "category": int(category),
    }


def load_excel_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load dataset from an Excel file like data/hate_speech_labeled_samples.xlsx.

    New expected columns (robust to casing/localization):
    - 'text' (or 'Text'/'Tekst') for the sentence
    - 'category' (or 'Category') as codes like 0, 1a, 1b, 2, 3b...

    Backward compatibility: if 'Id category' exists, it will be used; 'Labelar1' is ignored now.

    Returns list of dicts: {text: str, has_hate_speech: bool, category: int}
    """
    df = pd.read_excel(filepath)
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        row_dict = dict(row)
        text = _extract_text(row_dict)
        if not text:
            continue
        cat_code = (
            row_dict.get("category")
            or row_dict.get("Category")
            or row_dict.get("Id category")  # legacy fallback
        )
        # Prefer explicit boolean if provided. Otherwise infer from category.
        has_hate_val = row_dict.get("has_hate_speech")
        rec = _normalize_record(text, cat_code, has_hate_val if isinstance(has_hate_val, bool) else None)
        records.append(rec)
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
        if 'text' in sample and not isinstance(sample.get('text'), str):
            errors.append(f"Sample {i} 'text' is not a string")
        if 'has_hate_speech' in sample and not isinstance(sample.get('has_hate_speech'), bool):
            warnings.append(f"Sample {i} 'has_hate_speech' should be bool")
        if 'category' in sample:
            cat_raw = sample.get('category')
            cat = _map_serbian_category_code(cat_raw) if not isinstance(cat_raw, int) else cat_raw
            if not isinstance(cat, int):
                warnings.append(f"Sample {i} 'category' is not an int-like value")
            else:
                if cat < 0 or cat > 7:
                    warnings.append(f"Sample {i} 'category' should be 0-7")

    total_samples = len(data)
    stats = {
        "total_samples": total_samples,
        "hate_speech_count": sum(1 for s in data if bool(s.get('has_hate_speech', False))),
        "no_hate_count": sum(1 for s in data if not bool(s.get('has_hate_speech', False))),
        "category_distribution": {},
    }
    for cid in range(8):
        count = sum(1 for s in data if _map_serbian_category_code(s.get('category')) == cid)
        if count > 0:
            stats['category_distribution'][cid] = count

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings, "statistics": stats}


def print_dataset_info(data: List[Dict[str, Any]]) -> None:
    """Pretty-print basic dataset stats."""
    info = validate_dataset(data)
    stats = info.get("statistics", {})
    print("Dataset info:")
    print(f"  Total samples: {stats.get('total_samples', 0)}")
    print(f"  Hate speech:   {stats.get('hate_speech_count', 0)}")
    print(f"  No hate:       {stats.get('no_hate_count', 0)}")
    dist = stats.get('category_distribution', {})
    if dist:
        print("  Category distribution (0-7):")
        for k in sorted(dist.keys()):
            print(f"    {k}: {dist[k]}")


if __name__ == "__main__":
    path = "data/hate_speech_labeled_samples.xlsx"
    data = load_excel_dataset(path)
    print_dataset_info(data)
