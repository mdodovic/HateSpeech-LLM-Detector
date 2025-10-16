"""
Demo skripta za detekciju govora mržnje (samo srpski)
"""

import sys
sys.path.append('./examples/src')

from src.llm_detector import LLMDetector
from src.categories import get_category_prompt, HATE_SPEECH_CATEGORIES


def demo_single_text():
    print("Inicijalizacija LLM detektora (Ollama backend)...")
    print("Uverite se da je Ollama pokrenuta. Ako /api/chat nije dostupno, koristi se /api/generate.")
    model_name = "phi3"  # ili npr. "phi3:mini", "llama3.2:3b", "mistral:7b"

    # Primeri tekstova (srpski)
    texts = [
        "Volim da provodim vreme sa ljudima iz različitih sredina.",
        "Ti imigranti nam uništavaju zemlju i treba da se vrate odakle su došli.",
        "Žene nisu sposobne da budu dobri inženjeri.",
    ]

    detector = LLMDetector(model_name)
    categories_prompt = get_category_prompt()

    for i, text in enumerate(texts, 1):
        print(f"\n{'='*60}")
        print(f"Tekst {i}: {text}")
        print(f"{'='*60}")
        result = detector.analyze_text_complete(text, categories_prompt)
        print(f"Ima govor mržnje: {result['has_hate_speech']}")
        print(f"Kategorija: {result['category']} - {HATE_SPEECH_CATEGORIES.get(result['category'], 'Nepoznato')}")
        print(f"Pokrivenost tokena: {result['token_coverage_ratio']:.2%}")


if __name__ == "__main__":
    demo_single_text()
