"""
Demo script for Hate Speech Detection
This script demonstrates how to use the detector on individual texts
"""

import sys
sys.path.append('../src')

from llm_detector import LLMDetector
from categories import get_category_prompt, HATE_SPEECH_CATEGORIES


def demo_single_text():
    """Demonstrate detection on a single text"""
    
    # Example texts
    texts = [
        "I love spending time with people from different backgrounds.",
        "Those immigrants are ruining our country and should go back where they came from.",
        "Women are not capable of being good engineers."
    ]
    
    # Initialize detector (using a smaller model for demo)
    print("Initializing LLM detector...")
    print("Note: For this demo, you should use a small model like 'microsoft/phi-2' or similar")
    print("Larger models (Llama, Mistral, etc.) require more GPU memory")
    
    model_name = input("\nEnter model name (e.g., 'microsoft/phi-2'): ").strip()
    if not model_name:
        print("No model specified. Exiting.")
        return
    
    try:
        detector = LLMDetector(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    categories_prompt = get_category_prompt()
    
    # Analyze each text
    for i, text in enumerate(texts, 1):
        print(f"\n{'='*60}")
        print(f"Text {i}: {text}")
        print(f"{'='*60}")
        
        result = detector.analyze_text_complete(text, categories_prompt)
        
        print(f"\nTask 1 - Has hate speech: {result['has_hate_speech']}")
        print(f"Explanation: {result['binary_explanation'][:200]}...")
        
        print(f"\nTask 2 - Extracted hate sentences:")
        if result['hate_sentences']:
            for sent in result['hate_sentences']:
                print(f"  - {sent}")
        else:
            print("  (none)")
        print(f"Token coverage: {result['tokens_covered']}/{result['total_tokens']} "
              f"({result['token_coverage_ratio']:.2%})")
        
        print(f"\nTask 3 - Category: {result['category']} - {HATE_SPEECH_CATEGORIES[result['category']]}")
        print(f"Explanation: {result['category_explanation'][:200]}...")


def interactive_demo():
    """Interactive demo where user can input their own text"""
    
    print("Interactive Hate Speech Detection Demo")
    print("="*60)
    
    model_name = input("Enter model name (e.g., 'microsoft/phi-2'): ").strip()
    if not model_name:
        print("No model specified. Exiting.")
        return
    
    try:
        detector = LLMDetector(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    categories_prompt = get_category_prompt()
    
    while True:
        print("\n" + "="*60)
        text = input("\nEnter text to analyze (or 'quit' to exit): ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            print("Empty text. Please try again.")
            continue
        
        print("\nAnalyzing...")
        result = detector.analyze_text_complete(text, categories_prompt)
        
        print(f"\n--- Results ---")
        print(f"Has hate speech: {result['has_hate_speech']}")
        print(f"Category: {result['category']} - {HATE_SPEECH_CATEGORIES[result['category']]}")
        print(f"Token coverage: {result['token_coverage_ratio']:.2%}")
        
        if result['hate_sentences']:
            print(f"\nExtracted hate speech sentences:")
            for sent in result['hate_sentences']:
                print(f"  - {sent}")


if __name__ == "__main__":
    print("Hate Speech Detection Demo")
    print("="*60)
    print("1. Demo with predefined texts")
    print("2. Interactive mode (enter your own texts)")
    
    choice = input("\nSelect mode (1 or 2): ").strip()
    
    if choice == "1":
        demo_single_text()
    elif choice == "2":
        interactive_demo()
    else:
        print("Invalid choice. Exiting.")
