"""
Utility functions for HateSpeech-LLM-Detector
"""

import json
import csv
from typing import List, Dict, Any
import os


def load_json_dataset(filepath: str) -> List[Dict[str, Any]]:
    """
    Load dataset from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of dataset samples
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json_dataset(data: List[Dict[str, Any]], filepath: str):
    """
    Save dataset to JSON file
    
    Args:
        data: List of dataset samples
        filepath: Path to save JSON file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_csv_dataset(filepath: str) -> List[Dict[str, Any]]:
    """
    Load dataset from CSV file
    Expected columns: text, has_hate_speech, category
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of dataset samples
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'text': row['text'],
                'has_hate_speech': row['has_hate_speech'].lower() in ['true', '1', 'yes'],
                'category': int(row['category'])
            })
    return data


def save_csv_dataset(data: List[Dict[str, Any]], filepath: str):
    """
    Save dataset to CSV file
    
    Args:
        data: List of dataset samples
        filepath: Path to save CSV file
    """
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        if not data:
            return
        
        fieldnames = list(data[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def convert_csv_to_json(csv_path: str, json_path: str):
    """
    Convert CSV dataset to JSON format
    
    Args:
        csv_path: Path to input CSV file
        json_path: Path to output JSON file
    """
    data = load_csv_dataset(csv_path)
    save_json_dataset(data, json_path)
    print(f"Converted {csv_path} to {json_path}")
    print(f"Total samples: {len(data)}")


def convert_json_to_csv(json_path: str, csv_path: str):
    """
    Convert JSON dataset to CSV format
    
    Args:
        json_path: Path to input JSON file
        csv_path: Path to output CSV file
    """
    data = load_json_dataset(json_path)
    save_csv_dataset(data, csv_path)
    print(f"Converted {json_path} to {csv_path}")
    print(f"Total samples: {len(data)}")


def validate_dataset(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate dataset structure and return statistics
    
    Args:
        data: List of dataset samples
        
    Returns:
        Dictionary with validation results and statistics
    """
    errors = []
    warnings = []
    
    if not data:
        errors.append("Dataset is empty")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    # Check required fields
    required_fields = ['text', 'has_hate_speech', 'category']
    
    for i, sample in enumerate(data):
        # Check required fields exist
        for field in required_fields:
            if field not in sample:
                errors.append(f"Sample {i}: Missing required field '{field}'")
        
        # Check field types
        if 'text' in sample and not isinstance(sample['text'], str):
            errors.append(f"Sample {i}: 'text' should be string")
        
        if 'has_hate_speech' in sample and not isinstance(sample['has_hate_speech'], bool):
            errors.append(f"Sample {i}: 'has_hate_speech' should be boolean")
        
        if 'category' in sample:
            if not isinstance(sample['category'], int):
                errors.append(f"Sample {i}: 'category' should be integer")
            elif not (0 <= sample['category'] <= 7):
                errors.append(f"Sample {i}: 'category' should be 0-7, got {sample['category']}")
        
        # Check logical consistency
        if 'has_hate_speech' in sample and 'category' in sample:
            if not sample['has_hate_speech'] and sample['category'] != 0:
                warnings.append(f"Sample {i}: has_hate_speech=False but category={sample['category']} (should be 0)")
    
    # Calculate statistics
    stats = {
        "total_samples": len(data),
        "hate_speech_count": sum(1 for s in data if s.get('has_hate_speech', False)),
        "no_hate_count": sum(1 for s in data if not s.get('has_hate_speech', False)),
        "category_distribution": {}
    }
    
    for i in range(8):
        count = sum(1 for s in data if s.get('category') == i)
        if count > 0:
            stats["category_distribution"][i] = count
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "statistics": stats
    }


def print_dataset_info(filepath: str):
    """
    Print information about a dataset file
    
    Args:
        filepath: Path to dataset file (JSON or CSV)
    """
    # Determine file type and load
    if filepath.endswith('.json'):
        data = load_json_dataset(filepath)
    elif filepath.endswith('.csv'):
        data = load_csv_dataset(filepath)
    else:
        print(f"Unsupported file type: {filepath}")
        return
    
    # Validate and get info
    validation = validate_dataset(data)
    
    print(f"\nDataset Info: {filepath}")
    print("=" * 60)
    
    if validation["valid"]:
        print("✓ Dataset is valid")
    else:
        print("✗ Dataset has errors:")
        for error in validation["errors"]:
            print(f"  - {error}")
    
    if validation["warnings"]:
        print("\nWarnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
    
    stats = validation["statistics"]
    print(f"\nStatistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Hate speech: {stats['hate_speech_count']} ({stats['hate_speech_count']/stats['total_samples']*100:.1f}%)")
    print(f"  No hate speech: {stats['no_hate_count']} ({stats['no_hate_count']/stats['total_samples']*100:.1f}%)")
    
    print(f"\n  Category distribution:")
    for cat_id, count in sorted(validation["statistics"]["category_distribution"].items()):
        print(f"    Category {cat_id}: {count} ({count/stats['total_samples']*100:.1f}%)")


def merge_datasets(filepaths: List[str], output_path: str):
    """
    Merge multiple datasets into one
    
    Args:
        filepaths: List of paths to dataset files
        output_path: Path to save merged dataset
    """
    merged_data = []
    
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
    
    # Save merged dataset
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
    # Example usage
    import sys
    
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
