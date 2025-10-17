"""
Utility functions for HateSpeech-LLM-Detector (Serbian)
"""
from typing import List, Dict, Any
from typing import Optional
import json
from pathlib import Path
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

    if not records:
        print("Dataset je prazan ili nije moguće učitati podatke.")
        exit(1)
    
    return records


def load_default_model_tags(models_json_path: Optional[str] = None) -> Dict[str, str]:
    """Load default Ollama model tags from models/models.json.

    Args:
        models_json_path: Optional explicit path to models.json. If not provided,
            will look for '<repo_root>/models/models.json' and then fallback to
            '<repo_root>/data/models.json'.

    Returns:
        Dict mapping model display name (lowercased) -> ollama tag string.
        Returns empty dict if file not found or invalid.
    """
    # Determine base directory from this utils.py file location
    repo_root = Path(__file__).resolve().parents[1]

    candidates = []
    if models_json_path:
        candidates.append(Path(models_json_path))
    # Primary location
    candidates.append(repo_root / "models" / "models.json")
    # Backward-compat fallback
    candidates.append(repo_root / "data" / "models.json")

    json_path: Optional[Path] = None
    for p in candidates:
        if p.exists():
            json_path = p
            break

    if json_path is None:
        # Graceful fallback: return empty mapping
        return {}

    try:
        # Support simple '//' comments by stripping them before parsing
        with open(json_path, "r", encoding="utf-8") as f:
            raw = f.read()
        import re as _re
        cleaned = _re.sub(r"//.*", "", raw)
        data = json.loads(cleaned)
        # Normalize keys and values to strings and lower-case keys
        return {str(k).strip().lower(): str(v).strip() for k, v in dict(data).items()}
    except Exception:
        return {}


def build_model_tags(models: Optional[List[str]] = None, models_json_path: Optional[str] = None) -> Dict[str, str]:
    """Build mapping of model names to Ollama tags using models.json.

    Behavior matches previous logic:
    - If models list is empty/None: use all keys from models.json
    - For each requested name, use tag from models.json if present; otherwise use the name itself
    """
    default_tags = load_default_model_tags(models_json_path)
    mapping: Dict[str, str] = {}
    if not models:
        names = list(default_tags.keys())
    else:
        names = [str(n).strip().lower() for n in models]
    for key in names:
        mapping[key] = default_tags.get(key) or key

    print("\nModeli za evaluaciju (ime -> ollama tag):")
    for k, v in mapping.items():
        print(f"  {k} -> {v}")

    return mapping


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
