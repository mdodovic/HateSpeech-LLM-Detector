"""
Utility functions for HateSpeech-LLM-Detector
"""
from typing import List, Dict, Any
import sys
import json
import csv

# Constants
NO_HATE_SPEECH_CATEGORY = 0


def load_json_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json_dataset(data: List[Dict[str, Any]], filepath: str):
    """Save dataset to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_csv_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load dataset from CSV file. Expected columns: text, has_hate_speech, category"""
    data: List[Dict[str, Any]] = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text', '')
            has_hate = row.get('has_hate_speech', 'false').strip().lower() in {"1", "true", "yes", "y"}
            try:
                category = int(row.get('category', NO_HATE_SPEECH_CATEGORY))
            except Exception:
                category = NO_HATE_SPEECH_CATEGORY
            data.append({
                'text': text,
                'has_hate_speech': bool(has_hate),
                'category': category,
            })
    return data


def save_csv_dataset(data: List[Dict[str, Any]], filepath: str):
    """Save dataset to CSV file"""
    if not data:
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            f.write('text,has_hate_speech,category\n')
        return
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        fieldnames = list(data[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def convert_csv_to_json(csv_path: str, json_path: str):
    data = load_csv_dataset(csv_path)
    save_json_dataset(data, json_path)
    print(f"Converted {csv_path} to {json_path}")
    print(f"Total samples: {len(data)}")


def convert_json_to_csv(json_path: str, csv_path: str):
    data = load_json_dataset(json_path)
    save_csv_dataset(data, csv_path)
    print(f"Converted {json_path} to {csv_path}")
    print(f"Total samples: {len(data)}")


def validate_dataset(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate dataset structure and return statistics"""
    errors: List[str] = []
    warnings: List[str] = []

    if not data:
        errors.append("Dataset is empty")
        return {"valid": False, "errors": errors, "warnings": warnings, "statistics": {"total_samples": 0}}

    required_fields = ['text', 'has_hate_speech', 'category']

    for i, sample in enumerate(data):
        # Required fields
        for field in required_fields:
            if field not in sample:
                errors.append(f"Sample {i} missing field: {field}")
        # Types
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
        # Logical consistency
        if 'has_hate_speech' in sample and 'category' in sample:
            try:
                cat = int(sample['category'])
                has = bool(sample['has_hate_speech'])
                if not has and cat != NO_HATE_SPEECH_CATEGORY:
                    warnings.append(f"Sample {i}: no hate speech but category != 0")
            except Exception:
                pass

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
            stats["category_distribution"][cid] = count

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings, "statistics": stats}


def print_dataset_info(filepath: str):
    """Print information about a dataset file"""
    # Determine file type and load
    if filepath.endswith('.json'):
        data = load_json_dataset(filepath)
    elif filepath.endswith('.csv'):
        data = load_csv_dataset(filepath)
    else:
        print(f"Unsupported file type: {filepath}")
        return

    validation = validate_dataset(data)

    print(f"\nDataset Info: {filepath}")
    print("=" * 60)

    if validation["valid"]:
        print("\u2713 Dataset is valid")
    else:
        print("\u2717 Dataset has errors:")
        for error in validation["errors"]:
            print(f"  - {error}")

    if validation["warnings"]:
        print("\nWarnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    stats = validation["statistics"]
    print(f"\nStatistics:")
    print(f"  Total samples: {stats.get('total_samples', 0)}")
    if stats.get('total_samples', 0) > 0:
        hs = stats['hate_speech_count']
        nh = stats['no_hate_count']
        tot = stats['total_samples']
        print(f"  Hate speech: {hs} ({hs/tot*100:.1f}%)")
        print(f"  No hate speech: {nh} ({nh/tot*100:.1f}%)")
        if stats.get('category_distribution'):
            print(f"\n  Category distribution:")
            for cat_id, count in sorted(stats['category_distribution'].items()):
                print(f"    Category {cat_id}: {count} ({count/tot*100:.1f}%)")


def merge_datasets(filepaths: List[str], output_path: str):
    """Merge multiple datasets into one"""
    merged_data: List[Dict[str, Any]] = []
    for filepath in filepaths:
        if filepath.endswith('.json'):
            data = load_json_dataset(filepath)
        elif filepath.endswith('.csv'):
            data = load_csv_dataset(filepath)
        else:
            print(f"Skipping unsupported file type: {filepath}")
            continue
        merged_data.extend(data)
        print(f"Added {len(data)} samples from {filepath}")

    if output_path.endswith('.json'):
        save_json_dataset(merged_data, output_path)
    elif output_path.endswith('.csv'):
        save_csv_dataset(merged_data, output_path)
    else:
        print(f"Unsupported output file type: {output_path}")
        return

    print(f"\nMerged dataset saved to {output_path}")
    print(f"Total samples: {len(merged_data)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python utils.py info <dataset_file>")
        print("  python utils.py convert <input_file> <output_file>")
        print("  python utils.py merge <output_file> <input1> <input2> ...")
        sys.exit(1)

    command = sys.argv[1]

    if command == "info":
        if len(sys.argv) < 3:
            print("Usage: python utils.py info <dataset_file>")
            sys.exit(1)
        print_dataset_info(sys.argv[2])

    elif command == "convert":
        if len(sys.argv) < 4:
            print("Usage: python utils.py convert <input_file> <output_file>")
            sys.exit(1)
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        if input_file.endswith('.json') and output_file.endswith('.csv'):
            convert_json_to_csv(input_file, output_file)
        elif input_file.endswith('.csv') and output_file.endswith('.json'):
            convert_csv_to_json(input_file, output_file)
        else:
            print("Unsupported conversion")

    elif command == "merge":
        if len(sys.argv) < 4:
            print("Usage: python utils.py merge <output_file> <input1> <input2> ...")
            sys.exit(1)
        output_file = sys.argv[2]
        input_files = sys.argv[3:]
        merge_datasets(input_files, output_file)

    else:
        print(f"Unknown command: {command}")
