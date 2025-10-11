"""
HateSpeech-LLM-Detector Package

A comprehensive framework for detecting and categorizing hate speech using LLMs.
"""

from .llm_detector import LLMDetector
from .evaluation import HateSpeechEvaluator
from .categories import HATE_SPEECH_CATEGORIES, CATEGORY_DESCRIPTIONS, get_category_prompt
from .utils import (
    load_json_dataset, 
    save_json_dataset, 
    load_csv_dataset, 
    save_csv_dataset,
    validate_dataset,
    print_dataset_info
)

__version__ = "1.0.0"
__all__ = [
    "LLMDetector",
    "HateSpeechEvaluator",
    "HATE_SPEECH_CATEGORIES",
    "CATEGORY_DESCRIPTIONS",
    "get_category_prompt",
    "load_json_dataset",
    "save_json_dataset",
    "load_csv_dataset",
    "save_csv_dataset",
    "validate_dataset",
    "print_dataset_info"
]
