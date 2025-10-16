"""
Demo skripta za detekciju govora mržnje (samo srpski)
"""

import argparse
from typing import Optional

from src.llm_detector import LLMDetector
from src.categories import get_category_prompt, HATE_SPEECH_CATEGORIES
from src.utils import load_excel_dataset


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


def run(file_path: str, model_name: str = "phi3", limit: Optional[int] = None) -> None:
    """Učitaj Excel sa rečenicama i za svaku pokreni binarnu detekciju.

    Očekivani Excel format: kolona 'text' (ili 'Text'/'Tekst').
    """
    print("Učitavam dataset iz Excel fajla…")
    records = load_excel_dataset(file_path)
    if not records:
        print("Dataset je prazan ili nije moguće učitati podatke.")
        return

    if limit is not None:
        records = records[:max(0, int(limit))]

    print(f"Učitano uzoraka: {len(records)}")
    print("Inicijalizujem LLM detektor…")
    detector = LLMDetector(model_name)

    print("\nPokrećem binarnu detekciju govora mržnje po rečenici…")
    for idx, rec in enumerate(records, start=1):
        text = rec.get("text", "").strip()
        if not text:
            continue
        has_hate, explanation = detector.detect_hate_speech_binary(text)
        short_text = (text[:120] + "…") if len(text) > 120 else text
        print(f"\n[{idx}] Tekst: {short_text}")
        print(f"    Rezultat (has_hate_speech): {has_hate}")
        print(f"    Objašnjenje: {explanation}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Detekcija govora mržnje pomoću LLM (Ollama)")
    # parser.add_argument("--excel", type=str, default="data/hate_speech_labeled_samples.xlsx", help="Putanja do Excel fajla sa podacima")
    # parser.add_argument("--model", type=str, default="phi3", help="Ollama model, npr. 'phi3', 'phi3:mini', 'llama3.2:3b'")
    # parser.add_argument("--limit", type=int, default=None, help="Opcionalno: ograniči broj primera za obradu")
    # parser.add_argument("--demo", action="store_true", help="Pokreni demo sa hardkodiranim tekstovima umesto Excela")
    # args = parser.parse_args()

    # if args.demo:
    #     demo_single_text()
    # else:
    run(
        file_path="data/hate_speech_labeled_samples.xlsx",
        model_name="mistral",
    )
