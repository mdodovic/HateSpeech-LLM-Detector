# HateSpeech-LLM-Detector

A comprehensive framework for detecting and categorizing hate speech using Large Language Models (LLMs). This tool supports multiple LLMs including Llama, Mistral, Qwen, Phi, DeepSeek, and others for research purposes.

## Overview

This framework performs three key tasks:

1. **Binary Detection (Task 1)**: Determines if text contains hate speech (Yes/No)
2. **Sentence Extraction (Task 2)**: Extracts specific sentences containing hate speech and measures token coverage
3. **Categorization (Task 3)**: Classifies hate speech into 8 predefined categories (0-7)

## Hate Speech Categories

The framework uses the following classification system:

- **0**: No hate speech - No offensive content targeting protected groups
- **1**: Race/Ethnicity-based hate speech - Targeting race, ethnicity, or skin color
- **2**: Religion-based hate speech - Targeting religious beliefs or practices
- **3**: Gender-based hate speech - Targeting gender identity or expression
- **4**: Sexual orientation-based hate speech - Targeting LGBTQ+ individuals
- **5**: Disability-based hate speech - Targeting physical, mental, or developmental disabilities
- **6**: Nationality-based hate speech - Targeting country of origin or nationality
- **7**: Other forms of hate speech - Age, social class, or other characteristics

## Evaluation Metrics

### Task 1 & Task 3 (Classification)
- **Accuracy**: Overall correctness
- **Precision**: Correct positive predictions / Total positive predictions
- **Recall**: Correct positive predictions / Total actual positives
- **F1 Score**: Harmonic mean of precision and recall

### Task 2 (Extraction)
- **Tokens Covered**: Number of tokens in extracted hate speech sentences
- **Total Tokens**: Total tokens in the input text
- **Token Coverage Ratio**: Proportion of text identified as hate speech

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mdodovic/HateSpeech-LLM-Detector.git
cd HateSpeech-LLM-Detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run evaluation on a dataset with multiple models:

```bash
python src/main.py \
    --dataset examples/sample_dataset.json \
    --models "microsoft/phi-2" "mistralai/Mistral-7B-Instruct-v0.1" \
    --output-dir results \
    --device auto
```

### Dataset Format

Your dataset should be a JSON file with the following structure:

```json
[
  {
    "text": "Your text here...",
    "has_hate_speech": true,
    "category": 1
  },
  {
    "text": "Another text...",
    "has_hate_speech": false,
    "category": 0
  }
]
```

### Interactive Demo

Try the interactive demo to test individual texts:

```bash
cd examples
python demo.py
```

### Supported Models

The framework supports any HuggingFace causal language model. Recommended models include:

- **Phi**: `microsoft/phi-2`, `microsoft/phi-3-mini`
- **Llama**: `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-3-8B-Instruct`
- **Mistral**: `mistralai/Mistral-7B-Instruct-v0.1`, `mistralai/Mistral-7B-Instruct-v0.2`
- **Qwen**: `Qwen/Qwen2-7B-Instruct`, `Qwen/Qwen1.5-7B-Chat`
- **DeepSeek**: `deepseek-ai/deepseek-llm-7b-chat`

Note: You may need to request access for some models (e.g., Llama) through HuggingFace.

## Project Structure

```
HateSpeech-LLM-Detector/
├── src/
│   ├── categories.py       # Hate speech category definitions
│   ├── llm_detector.py     # Main LLM detector class
│   ├── evaluation.py       # Evaluation metrics and reporting
│   └── main.py            # Main evaluation script
├── examples/
│   ├── sample_dataset.json # Example dataset
│   └── demo.py            # Interactive demo script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Output Files

The framework generates several output files in the results directory:

- `{model_name}_detailed_results.csv`: Detailed predictions for each sample
- `{model_name}_metrics.json`: All evaluation metrics
- `{model_name}_classification_report.txt`: Detailed classification report
- `model_comparison.csv`: Comparison across all evaluated models

## Example Output

```
============================================================
Results for: microsoft/phi-2
============================================================

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
