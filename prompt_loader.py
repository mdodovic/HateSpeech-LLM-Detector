"""
Prompt Loader Utility for HateSpeech-LLM-Detector

This module provides functionality to load and manage prompts for hate speech detection
across different languages.
"""

import os
from pathlib import Path
from typing import Optional


class PromptLoader:
    """Loads and manages prompts for hate speech detection."""
    
    PROMPT_TYPES = ['system', 'classification', 'analysis']
    SUPPORTED_LANGUAGES = ['serbian', 'english']
    
    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize the PromptLoader.
        
        Args:
            prompts_dir: Path to the prompts directory. If None, uses default location.
        """
        if prompts_dir is None:
            # Default to prompts directory in the same location as this file
            base_dir = Path(__file__).parent
            self.prompts_dir = base_dir / 'prompts'
        else:
            self.prompts_dir = Path(prompts_dir)
        
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")
    
    def load_prompt(self, language: str, prompt_type: str) -> str:
        """
        Load a specific prompt.
        
        Args:
            language: Language code (e.g., 'serbian', 'english')
            prompt_type: Type of prompt ('system', 'classification', 'analysis')
            
        Returns:
            The prompt text as a string
            
        Raises:
            ValueError: If language or prompt_type is not supported
            FileNotFoundError: If the prompt file doesn't exist
        """
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported languages: {', '.join(self.SUPPORTED_LANGUAGES)}"
            )
        
        if prompt_type not in self.PROMPT_TYPES:
            raise ValueError(
                f"Unsupported prompt type: {prompt_type}. "
                f"Supported types: {', '.join(self.PROMPT_TYPES)}"
            )
        
        prompt_filename = f"{prompt_type}_prompt.txt"
        prompt_path = self.prompts_dir / language / prompt_filename
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def get_system_prompt(self, language: str) -> str:
        """Load system prompt for a specific language."""
        return self.load_prompt(language, 'system')
    
    def get_classification_prompt(self, language: str) -> str:
        """Load classification prompt for a specific language."""
        return self.load_prompt(language, 'classification')
    
    def get_analysis_prompt(self, language: str) -> str:
        """Load analysis prompt for a specific language."""
        return self.load_prompt(language, 'analysis')
    
    def list_available_languages(self) -> list:
        """List all available languages."""
        languages = []
        for item in self.prompts_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                languages.append(item.name)
        return sorted(languages)
    
    def list_available_prompts(self, language: str) -> list:
        """List all available prompt types for a specific language."""
        lang_dir = self.prompts_dir / language
        if not lang_dir.exists():
            return []
        
        prompts = []
        for item in lang_dir.iterdir():
            if item.is_file() and item.suffix == '.txt':
                # Extract prompt type from filename (e.g., 'system_prompt.txt' -> 'system')
                prompt_type = item.stem.replace('_prompt', '')
                prompts.append(prompt_type)
        return sorted(prompts)


# Convenience function for quick access
def load_prompt(language: str, prompt_type: str, prompts_dir: Optional[str] = None) -> str:
    """
    Convenience function to load a prompt.
    
    Args:
        language: Language code (e.g., 'serbian', 'english')
        prompt_type: Type of prompt ('system', 'classification', 'analysis')
        prompts_dir: Optional custom prompts directory
        
    Returns:
        The prompt text as a string
    """
    loader = PromptLoader(prompts_dir)
    return loader.load_prompt(language, prompt_type)


if __name__ == '__main__':
    # Example usage
    loader = PromptLoader()
    
    print("Available languages:", loader.list_available_languages())
    print()
    
    # Load Serbian prompts
    print("=== Serbian System Prompt ===")
    print(loader.get_system_prompt('serbian'))
    print()
    
    print("=== Serbian Classification Prompt (sample) ===")
    classification = loader.get_classification_prompt('serbian')
    print(classification.format(text="Primer teksta za analizu"))
    print()
    
    # Load English prompts
    print("=== English System Prompt ===")
    print(loader.get_system_prompt('english'))
