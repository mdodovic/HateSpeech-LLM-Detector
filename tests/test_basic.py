"""
Basic tests for the hate speech detection framework
These tests verify structure and logic without requiring model downloads
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
from categories import HATE_SPEECH_CATEGORIES, CATEGORY_DESCRIPTIONS, get_category_prompt
from evaluation import HateSpeechEvaluator


def test_categories():
    """Test category definitions"""
    print("Testing categories...")
    
    # Check we have 8 categories (0-7)
    assert len(HATE_SPEECH_CATEGORIES) == 8, "Should have 8 categories"
    assert len(CATEGORY_DESCRIPTIONS) == 8, "Should have 8 descriptions"
    
    # Check all IDs from 0-7 are present
    for i in range(8):
        assert i in HATE_SPEECH_CATEGORIES, f"Category {i} missing"
        assert i in CATEGORY_DESCRIPTIONS, f"Description {i} missing"
    
    # Check category 0 is "no hate speech"
    assert "no hate" in HATE_SPEECH_CATEGORIES[0].lower(), "Category 0 should be no hate speech"
    
    # Check prompt generation
    prompt = get_category_prompt()
    assert len(prompt) > 0, "Prompt should not be empty"
    assert "categor" in prompt.lower(), "Prompt should mention categories"
    
    print("✓ Categories test passed")


def test_evaluation_binary():
    """Test binary classification evaluation"""
    print("Testing binary evaluation...")
    
    evaluator = HateSpeechEvaluator()
    
    # Perfect predictions
    y_true = [True, False, True, False, True]
    y_pred = [True, False, True, False, True]
    
    metrics = evaluator.evaluate_binary_classification(y_true, y_pred)
    
    assert metrics["accuracy"] == 1.0, "Perfect predictions should give accuracy 1.0"
    assert metrics["precision"] == 1.0, "Perfect predictions should give precision 1.0"
    assert metrics["recall"] == 1.0, "Perfect predictions should give recall 1.0"
    assert metrics["f1"] == 1.0, "Perfect predictions should give F1 1.0"
    
    # Imperfect predictions
    y_true2 = [True, False, True, False]
    y_pred2 = [True, False, False, False]
    
    metrics2 = evaluator.evaluate_binary_classification(y_true2, y_pred2)
    
    assert 0 <= metrics2["accuracy"] <= 1, "Accuracy should be between 0 and 1"
    assert metrics2["accuracy"] == 0.75, "Should get 3/4 correct"
    
    print("✓ Binary evaluation test passed")


def test_evaluation_multiclass():
    """Test multiclass classification evaluation"""
    print("Testing multiclass evaluation...")
    
    evaluator = HateSpeechEvaluator()
    
    # Perfect predictions
    y_true = [0, 1, 2, 3, 4, 5, 6, 7]
    y_pred = [0, 1, 2, 3, 4, 5, 6, 7]
    
    metrics = evaluator.evaluate_multiclass_classification(y_true, y_pred)
    
    assert metrics["accuracy"] == 1.0, "Perfect predictions should give accuracy 1.0"
    assert metrics["f1_macro"] == 1.0, "Perfect predictions should give F1 macro 1.0"
    
    # Imperfect predictions
    y_true2 = [0, 1, 2, 3, 0, 1, 2, 3]
    y_pred2 = [0, 1, 1, 3, 0, 2, 2, 3]  # 6/8 correct
    
    metrics2 = evaluator.evaluate_multiclass_classification(y_true2, y_pred2)
    
    assert 0 <= metrics2["accuracy"] <= 1, "Accuracy should be between 0 and 1"
    assert metrics2["accuracy"] == 0.75, "Should get 6/8 correct"
    
    print("✓ Multiclass evaluation test passed")


def test_evaluation_token_coverage():
    """Test token coverage evaluation"""
    print("Testing token coverage evaluation...")
    
    evaluator = HateSpeechEvaluator()
    
    tokens_covered = [10, 20, 30, 0]
    total_tokens = [100, 100, 100, 100]
    
    metrics = evaluator.evaluate_token_coverage(tokens_covered, total_tokens)
    
    assert metrics["total_tokens_analyzed"] == 400, "Should sum all total tokens"
    assert metrics["total_tokens_covered"] == 60, "Should sum all covered tokens"
    assert abs(metrics["mean_coverage_ratio"] - 0.15) < 0.001, "Mean should be approximately (0.1+0.2+0.3+0.0)/4 = 0.15"
    
    print("✓ Token coverage test passed")


def test_dataset_format():
    """Test that sample dataset has correct format"""
    print("Testing dataset format...")
    
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'sample_dataset.json')
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    assert len(data) > 0, "Dataset should not be empty"
    
    for i, sample in enumerate(data):
        assert "text" in sample, f"Sample {i} should have 'text' field"
        assert "has_hate_speech" in sample, f"Sample {i} should have 'has_hate_speech' field"
        assert "category" in sample, f"Sample {i} should have 'category' field"
        
        assert isinstance(sample["text"], str), f"Sample {i} text should be string"
        assert isinstance(sample["has_hate_speech"], bool), f"Sample {i} has_hate_speech should be bool"
        assert isinstance(sample["category"], int), f"Sample {i} category should be int"
        assert 0 <= sample["category"] <= 7, f"Sample {i} category should be 0-7"
        
        # If no hate speech, category should be 0
        if not sample["has_hate_speech"]:
            assert sample["category"] == 0, f"Sample {i} with no hate speech should have category 0"
    
    print("✓ Dataset format test passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("Running Basic Tests for HateSpeech-LLM-Detector")
    print("="*60 + "\n")
    
    try:
        test_categories()
        test_evaluation_binary()
        test_evaluation_multiclass()
        test_evaluation_token_coverage()
        test_dataset_format()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60 + "\n")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {str(e)}\n")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
