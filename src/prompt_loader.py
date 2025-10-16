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

    def __init__(self, prompts_dir = None):
        """
        Initialize the PromptLoader.
        
        Args:
            prompts_dir: Path to the prompts directory. If None, uses default location.
        """
        if prompts_dir is None:
            self.prompts_dir = Path(os.path.join(Path(__file__).parent, 'prompts'))
        else:
            self.prompts_dir = Path(prompts_dir)
        
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")

    def load_prompt(self, prompt_type: str) -> str:
        """
        Load a specific prompt.
        
        Args:
            prompt_type: Type of prompt ('system', 'classification', 'analysis')
            
        Returns:
            The prompt text as a string
            
        Raises:
            ValueError: If language or prompt_type is not supported
            FileNotFoundError: If the prompt file doesn't exist
        """

        if prompt_type not in self.PROMPT_TYPES:
            raise ValueError(
                f"Unsupported prompt type: {prompt_type}. "
                f"Supported types: {', '.join(self.PROMPT_TYPES)}"
            )
        
        prompt_filename = f"{prompt_type}_prompt.txt"
        prompt_path = self.prompts_dir / prompt_filename

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def get_system_prompt(self) -> str:
        """Load system prompt."""
        return self.load_prompt('system')

    def get_classification_prompt(self) -> str:
        """Load classification prompt."""
        return self.load_prompt('classification')

    def get_analysis_prompt(self) -> str:
        """Load analysis prompt."""
        return self.load_prompt('analysis')

    def list_available_prompts(self) -> list:
        """List all available prompt types."""
        return self.PROMPT_TYPES
        


# Convenience function for quick access
def load_prompt(prompt_type: str, prompts_dir: Optional[str] = None) -> str:
    """
    Convenience function to load a prompt.
    
    Args:
        prompt_type: Type of prompt ('system', 'classification', 'analysis')
        prompts_dir: Optional custom prompts directory
        
    Returns:
        The prompt text as a string
    """
    loader = PromptLoader(prompts_dir)
    return loader.load_prompt(prompt_type)


if __name__ == '__main__':
    # Example usage
    loader = PromptLoader()
    
    # Load Serbian prompts
    print("=== Serbian System Prompt ===")
    print(loader.get_system_prompt())
    print()
    
    print("=== Serbian Classification Prompt (sample) ===")
    classification = loader.get_classification_prompt()
    print(classification.format(text="Primer teksta za analizu"))
    print()
    
    print("=== Serbian Analysis Prompt (sample) ===")
    analysis = loader.get_analysis_prompt()
    print(analysis.format(text="Primer teksta za analizu"))
    print()
