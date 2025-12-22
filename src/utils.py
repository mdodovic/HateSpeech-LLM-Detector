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


def parse_category_and_subcategory(code: Any) -> Dict[str, Any]:
    """Parse raw category cell into top-level int (0-7) and subcategory code.

    Accepts values like:
    - 0, "0" -> category=0, subcategory=""
    - "1a", "3b", "6a" -> category=1/3/6, subcategory="a"/"b"/"a"
    - "2 homofobija" -> category=2, subcategory=""

    Returns dict: {"category": int, "subcategory": str}
    """
    if code is None:
        return {"category": NO_HATE_SPEECH_CATEGORY, "subcategory": ""}
    s = str(code).strip().lower()
    if s == "":
        return {"category": NO_HATE_SPEECH_CATEGORY, "subcategory": ""}
    # Exact zero
    if s == "0":
        return {"category": 0, "subcategory": ""}
    # Pattern like '3b'
    m_code = re.match(r"^([0-7])\s*([a-z])$", s)
    if m_code:
        cat = int(m_code.group(1))
        sub = m_code.group(2)
        return {"category": cat, "subcategory": sub}
    # Leading digit like '2 homofobija'
    m_num = re.match(r"^([0-7])\b", s)
    if m_num:
        cat = int(m_num.group(1))
        return {"category": cat, "subcategory": ""}
    return {"category": NO_HATE_SPEECH_CATEGORY, "subcategory": ""}


def _map_serbian_category_code(code: Any) -> int:
    """Map category codes to top-level class (0–7).

    Accepts values like:
    - "0" → 0 (no hate)
    - "1a", "3b", "6a" → 1, 3, 6 (top-level only)
    - "2 homofobija" → 2
    Any malformed or missing value maps to 0.
    """
    parsed = parse_category_and_subcategory(code)
    return int(parsed["category"]) if isinstance(parsed.get("category"), int) else NO_HATE_SPEECH_CATEGORY


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
    parsed = parse_category_and_subcategory(category_value)
    category = int(parsed["category"])
    subcategory = parsed["subcategory"]
    has_hate = bool(has_hate_value) if isinstance(has_hate_value, bool) else (category != NO_HATE_SPEECH_CATEGORY)
    return {
        "text": text,
        "has_hate_speech": has_hate,
        "category": int(category),
        "subcategory": subcategory,
    }


def load_excel_dataset(filepath: str, mode = None) -> List[Dict[str, Any]]:
    """Load dataset from an Excel file like data/single_sentence_hate_speech_labeled_samples.xlsx.

    New expected columns (robust to casing/localization):
    - 'text' (or 'Text'/'Tekst') for the sentence
    - 'category' (or 'Category') as codes like 0, 1a, 1b, 2, 3b...

    Backward compatibility: if 'Id category' exists, it will be used; 'Labelar1' is ignored now.

    Returns list of dicts: {text: str, has_hate_speech: bool, category: int, subcategory: str}
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

        # Support multiple ground-truth codes in a single cell (e.g., "4a,6a")
        all_codes: List[str] = []
        if isinstance(cat_code, str) and ("," in cat_code or ";" in cat_code):
            raw_codes = [c.strip() for c in re.split(r"[;,]", cat_code) if str(c).strip() != ""]
            parsed = [parse_category_and_subcategory(code) for code in raw_codes] if raw_codes else []
            all_codes = raw_codes
            all_categories = [int(p.get("category", 0)) for p in parsed]
            all_subcats = [str(p.get("subcategory", "") or "") for p in parsed]
            has_hate = any(c != 0 for c in all_categories)
            primary_idx = next((i for i, c in enumerate(all_categories) if c != 0), None)
            if primary_idx is not None:
                primary_cat = int(all_categories[primary_idx])
                primary_sub = all_subcats[primary_idx]
            else:
                primary_cat = int(all_categories[0]) if all_categories else 0
                primary_sub = all_subcats[0] if all_subcats else ""
            rec = {
                "text": text,
                "has_hate_speech": bool(has_hate_val) if isinstance(has_hate_val, bool) else has_hate,
                "category": primary_cat,
                "subcategory": primary_sub,
                "all_codes": all_codes,
                "all_categories": all_categories,
                "all_subcategories": all_subcats,
            }
            records.append(rec)
        else:
            rec = _normalize_record(text, cat_code, has_hate_val if isinstance(has_hate_val, bool) else None)
            # Add single-code view as 'all_*' for downstream consistency
            parsed_single = parse_category_and_subcategory(cat_code)
            code_str = str(parsed_single.get("category", 0)) + (str(parsed_single.get("subcategory", "")) or "")
            rec["all_codes"] = [code_str] if code_str != "" else []
            rec["all_categories"] = [rec["category"]]
            rec["all_subcategories"] = [rec["subcategory"]]
            records.append(rec)

    if not records:
        print("Dataset je prazan ili nije moguće učitati podatke.")
        exit(1)

    return records


def load_excel_full_text_dataset(filepath: str) -> List[Dict]:
    """Load full-text dataset where the 'Category' cell may contain comma-separated codes.

    Example row (as in the screenshot):
        Text: <long text>
        Category: "0, 0, 1c, 6a, 0, 0, 0"

    Returns a list of dict entries with both a primary category/subcategory for
    compatibility with current evaluators and the full list for future use:
        {
            'text': str,
            'has_hate_speech': bool,           # True if any parsed category != 0
            'category': int,                   # primary non-zero category or 0
            'subcategory': str,                # primary subcategory (letter) or ''
            'all_codes': List[str],            # raw codes like ["0","1c","6a"]
            'all_categories': List[int],       # top-level ints like [0,1,6]
            'all_subcategories': List[str],    # letters like ["","c","a"]
        }
    """
    df = pd.read_excel(filepath)
    records: List[Dict] = []

    for _, row in df.iterrows():
        row_dict = dict(row)
        text = str(row_dict.get("Text") or row_dict.get("text") or "").strip()
        if not text:
            continue

        cat_cell = row_dict.get("Category") or row_dict.get("category") or ""
        # Split by comma/semicolon, keep order
        raw_codes = [c.strip() for c in re.split(r"[;,]", str(cat_cell)) if str(c).strip() != ""]
        parsed = [parse_category_and_subcategory(code) for code in raw_codes] if raw_codes else []

        all_categories = [int(p.get("category", 0)) for p in parsed]
        all_subcats = [str(p.get("subcategory", "") or "") for p in parsed]
        has_hate = any(c != 0 for c in all_categories)

        # Choose primary non-zero as ground truth for compatibility with existing pipeline
        primary_idx = next((i for i, c in enumerate(all_categories) if c != 0), None)
        if primary_idx is not None:
            primary_cat = int(all_categories[primary_idx])
            primary_sub = all_subcats[primary_idx]
        else:
            primary_cat = 0
            primary_sub = ""

        records.append({
            "text": text,
            "has_hate_speech": has_hate,
            "category": primary_cat,
            "subcategory": primary_sub,
            "all_codes": raw_codes,
            "all_categories": all_categories,
            "all_subcategories": all_subcats,
        })

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
    path = "data/single_sentence_hate_speech_labeled_samples.xlsx"
    data = load_excel_dataset(path)
    print_dataset_info(data)
