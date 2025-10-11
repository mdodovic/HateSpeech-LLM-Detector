# HateSpeech-LLM-Detector

A hate speech detection system using Large Language Models (LLMs) with support for multiple languages, with primary focus on Serbian language.

## Features

- 🌐 Multi-language support (Serbian, English)
- 📝 Organized prompt templates for hate speech detection
- 🔍 Multiple detection modes: classification and detailed analysis
- 🎯 Easy-to-use prompt loader utility
- 📊 Structured JSON responses for easy integration

## Project Structure

```
HateSpeech-LLM-Detector/
├── prompts/              # Prompt templates organized by language
│   ├── serbian/         # Serbian language prompts (Primary)
│   │   ├── system_prompt.txt
│   │   ├── classification_prompt.txt
│   │   └── analysis_prompt.txt
│   ├── english/         # English language prompts
│   │   ├── system_prompt.txt
│   │   ├── classification_prompt.txt
│   │   └── analysis_prompt.txt
│   └── README.md        # Detailed prompts documentation
├── prompt_loader.py     # Utility for loading and managing prompts
├── example_usage.py     # Example usage demonstrations
└── README.md           # This file
```

## Quick Start

### Loading Prompts

```python
from prompt_loader import PromptLoader

# Initialize the loader
loader = PromptLoader()

# Load Serbian classification prompt
serbian_prompt = loader.get_classification_prompt('serbian')
formatted = serbian_prompt.format(text="Tekst za analizu")

# Load English system prompt
english_system = loader.get_system_prompt('english')
```

### Running Examples

```bash
# See available prompts and usage examples
python example_usage.py

# Test the prompt loader directly
python prompt_loader.py
```

## Prompt Types

### System Prompt
Defines the role and expertise of the LLM for hate speech detection.

### Classification Prompt
Binary classification with structured JSON output:
- `is_hate_speech`: Boolean indicator
- `confidence`: Confidence score (0-1)
- `categories`: Detected hate speech categories
- `explanation`: Brief explanation

### Analysis Prompt
Detailed analysis providing:
- Hate speech determination
- Specific problematic elements
- Target groups identification
- Severity assessment
- Moderation recommendations

## Supported Languages

- **Serbian (Srpski)** - Primary language with full support
- **English** - Full support for comparison and broader usage

## Adding New Languages

See [prompts/README.md](prompts/README.md) for instructions on adding support for additional languages.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
