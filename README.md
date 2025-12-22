# HateSpeech-LLM-Detector

Results for: microsoft/phi-2

--- Task 1: Binary Hate Speech Detection ---
Accuracy       : 0.9500
Precision      : 0.9200
Recall         : 0.9000
F1             : 0.9100

--- Task 3: Hate Speech Categorization ---
Accuracy                 : 0.8800
Precision macro          : 0.8500
Recall macro             : 0.8300
F1 macro                 : 0.8400

--- Task 2: Token Coverage Statistics ---
Total tokens analyzed            : 1250
Total tokens covered             : 385
Mean coverage ratio              : 0.3080
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended for larger models)

## Research Applications

This framework is designed for research purposes including:

- Comparing different LLMs for hate speech detection
- Analyzing token-level hate speech coverage
- Fine-grained categorization of hate speech types
- Developing and evaluating hate speech detection systems

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{hatespeech_llm_detector,
  title = {HateSpeech-LLM-Detector: A Framework for Hate Speech Detection using LLMs},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/mdodovic/HateSpeech-LLM-Detector}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This tool is designed for research and educational purposes. Hate speech detection is a sensitive topic and results should be carefully validated. The tool's predictions may not always be accurate and should not be used as the sole basis for content moderation decisions.
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

## Ensemble Majority Voting (Ansambl)

The script `single_sentence_run_ansamble.py` now supports an ensemble mode that queries all models defined in `models/models.json` and produces a majority-vote prediction.

Voting rules:
- `has_hate_speech`: majority True/False across models.
- `category`: majority among non-zero categories from models that predicted hate; ties → lowest category id.
- `subcategory`: majority letter among models that predicted the chosen category; ignores empty; ties → alphabetical.
- If majority is no hate → category=0, subcategory="".

### Run Only Ensemble
```bash
python single_sentence_run_ansamble.py --skip-individual --ensemble \
  --excel data/single_sentence_hate_speech_labeled_samples_small.xlsx
```

### Run Individual Evaluations + Ensemble
```bash
python single_sentence_run_ansamble.py --ensemble \
  --excel data/single_sentence_hate_speech_labeled_samples_small.xlsx
```

### Restrict to a Subset of Models
```bash
python single_sentence_run_ansamble.py --ensemble --models llama,qwen3
```

### Use Two-Prompt Mode Inside Ensemble (slower)
```bash
python single_sentence_run_ansamble.py --ensemble --ensemble-one-prompt=false
```
(`--ensemble-one-prompt` omitted means one-prompt mode is used.)

Results are appended to `results/single_sentence_comparison.xlsx` under sheet name `Ensemble`.

