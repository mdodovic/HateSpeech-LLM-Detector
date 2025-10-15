"""
Demo script for Hate Speech Detection
This script demonstrates how to use the detector on individual texts
"""

import sys
sys.path.append('./examples/src')

from llm_detector import LLMDetector
from categories import get_category_prompt, HATE_SPEECH_CATEGORIES


def demo_single_text():
    """Demonstrate detection on a single text"""

    # Example texts
    texts = [
        "I love spending time with people from different backgrounds.",
        "Those immigrants are ruining our country and should go back where they came from.",
        "Women are not capable of being good engineers.",
    ]

    # Initialize detector (using a local Ollama model tag)
    print("Initializing LLM detector (Ollama backend)...")
    print("Note: Ensure Ollama is running: ollama serve")
    print("Try a small model tag, e.g.: 'phi3:mini', 'llama3.2:3b', or 'mistral:7b'")

    model_name = "phi3"

    try:
        detector = LLMDetector(model_name)
    except Exception as e:
        print(f"Error initializing detector: {e}")
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
        print(
            f"Token coverage: {result['tokens_covered']}/{result['total_tokens']} "
            f"({result['token_coverage_ratio']:.2%})"
        )

        print(
            f"\nTask 3 - Category: {result['category']} - "
            f"{HATE_SPEECH_CATEGORIES.get(result['category'], 'Unknown')}"
        )
        print(f"Explanation: {result['category_explanation'][:200]}...")


if __name__ == "__main__":
    demo_single_text()
