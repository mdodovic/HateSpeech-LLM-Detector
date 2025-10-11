"""
Example usage of the HateSpeech-LLM-Detector prompt system.

This script demonstrates how to load and use prompts for hate speech detection.
"""

from prompt_loader import PromptLoader


def main():
    # Initialize the prompt loader
    loader = PromptLoader()
    
    print("=" * 60)
    print("HateSpeech-LLM-Detector - Prompt Usage Example")
    print("=" * 60)
    print()
    
    # Show available languages
    languages = loader.list_available_languages()
    print(f"Available languages: {', '.join(languages)}")
    print()
    
    # Example 1: Load Serbian classification prompt
    print("Example 1: Serbian Classification Prompt")
    print("-" * 60)
    serbian_classification = loader.get_classification_prompt('serbian')
    
    # Format with example text
    sample_text = "Ovo je primer teksta koji treba analizirati."
    formatted_prompt = serbian_classification.format(text=sample_text)
    print(formatted_prompt)
    print()
    
    # Example 2: Load Serbian system prompt
    print("Example 2: Serbian System Prompt")
    print("-" * 60)
    serbian_system = loader.get_system_prompt('serbian')
    print(serbian_system)
    print()
    
    # Example 3: Load English classification prompt
    print("Example 3: English Classification Prompt")
    print("-" * 60)
    english_classification = loader.get_classification_prompt('english')
    
    # Format with example text
    sample_text_en = "This is a sample text to analyze."
    formatted_prompt_en = english_classification.format(text=sample_text_en)
    print(formatted_prompt_en)
    print()
    
    # Example 4: Load Serbian analysis prompt
    print("Example 4: Serbian Analysis Prompt")
    print("-" * 60)
    serbian_analysis = loader.get_analysis_prompt('serbian')
    
    # Format with example text
    analysis_text = "Tekst koji zahteva detaljnu analizu."
    formatted_analysis = serbian_analysis.format(text=analysis_text)
    print(formatted_analysis)
    print()
    
    # Example 5: List available prompts for a language
    print("Example 5: Available Prompts for Serbian")
    print("-" * 60)
    serbian_prompts = loader.list_available_prompts('serbian')
    print(f"Serbian prompts: {', '.join(serbian_prompts)}")
    print()
    
    print("=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
