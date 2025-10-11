# HateSpeech-LLM-Detector: Project Summary

## Overview

This repository contains a complete framework for detecting and categorizing hate speech using Large Language Models (LLMs). The implementation addresses the research requirements for:

1. **Binary Detection** - Detecting if text contains hate speech
2. **Sentence Extraction** - Extracting specific hate speech sentences with token coverage metrics
3. **Categorization** - Classifying hate speech into 8 categories (0-7)

## Project Structure

```
HateSpeech-LLM-Detector/
├── Documentation
│   ├── README.md                    # Main documentation
│   ├── QUICKSTART.md               # 5-minute getting started guide
│   ├── USAGE.md                    # Detailed usage examples
│   ├── CATEGORIES.md               # Category definitions & guidelines
│   ├── IMPLEMENTATION_NOTES.md     # Technical implementation details
│   └── PROJECT_SUMMARY.md          # This file
│
├── Configuration
│   ├── requirements.txt            # Python dependencies
│   ├── config_template.yaml        # Configuration template
│   └── setup.py                    # Package installation script
│
├── Source Code (src/)
│   ├── __init__.py                 # Package initialization
│   ├── categories.py               # Category definitions
│   ├── llm_detector.py            # Core LLM detector class
│   ├── evaluation.py              # Evaluation metrics
│   ├── utils.py                   # Dataset utilities
│   └── main.py                    # Command-line interface
│
├── Examples (examples/)
│   ├── demo.py                    # Interactive demonstration
│   ├── sample_dataset.json        # Example dataset
│   └── expected_output_example.txt # Example output format
│
├── Tests (tests/)
│   └── test_basic.py              # Unit tests
│
└── Output (generated at runtime)
    └── results/                   # Evaluation results directory
```

## Key Features

### 1. Multi-Model Support

Supports any HuggingFace causal language model:
- **Phi** (microsoft/phi-2, microsoft/phi-3-mini)
- **Llama** (meta-llama/Llama-2-7b-chat-hf)
- **Mistral** (mistralai/Mistral-7B-Instruct-v0.1)
- **Qwen** (Qwen/Qwen2-7B-Instruct)
- **DeepSeek** (deepseek-ai/deepseek-llm-7b-chat)

### 2. Three Detection Tasks

**Task 1: Binary Detection**
- Yes/No hate speech classification
- Metrics: Accuracy, Precision, Recall, F1

**Task 2: Sentence Extraction**
- Extract specific hate speech sentences
- Metrics: Tokens covered, Total tokens, Coverage ratio

**Task 3: Categorization**
- Classify into 8 categories (0-7)
- Metrics: Accuracy, Precision (macro/weighted), Recall, F1

### 3. Eight Hate Speech Categories

0. **No hate speech** - Neutral/positive content
1. **Race/Ethnicity** - Targeting race, ethnicity, skin color
2. **Religion** - Targeting religious beliefs
3. **Gender** - Targeting gender identity
4. **Sexual Orientation** - Targeting LGBTQ+ individuals
5. **Disability** - Targeting disabilities
6. **Nationality** - Targeting country of origin
7. **Other** - Age, class, other characteristics

### 4. Comprehensive Evaluation

**Binary Metrics**:
- Accuracy, Precision, Recall, F1 Score

**Multi-class Metrics**:
- Macro-averaged (equal weight per class)
- Weighted-averaged (weighted by frequency)
- Per-class precision, recall, F1
- Confusion matrix

**Token Coverage Metrics**:
- Total tokens analyzed
- Total tokens covered
- Mean, median, std, min, max coverage ratios

### 5. Dataset Utilities

- Load/save JSON and CSV formats
- Format conversion
- Dataset validation
- Merge multiple datasets
- Statistics and info display

## Usage Examples

### Basic Usage

```bash
python src/main.py \
    --dataset examples/sample_dataset.json \
    --models "microsoft/phi-2" \
    --output-dir results
```

### Multiple Models

```bash
python src/main.py \
    --dataset examples/sample_dataset.json \
    --models "microsoft/phi-2" "mistralai/Mistral-7B-Instruct-v0.1" \
    --output-dir results
```

### Interactive Demo

```bash
cd examples
python demo.py
```

### Dataset Utilities

```bash
# Get dataset info
python src/utils.py info examples/sample_dataset.json

# Convert CSV to JSON
python src/utils.py convert data.csv data.json

# Merge datasets
python src/utils.py merge output.json dataset1.json dataset2.json
```

## Output Files

For each model evaluated:
- `{model}_detailed_results.csv` - All predictions
- `{model}_metrics.json` - All metrics
- `{model}_classification_report.txt` - Classification report

When evaluating multiple models:
- `model_comparison.csv` - Comparison table

## Technical Implementation

### Core Architecture

**LLMDetector Class**:
- Loads models from HuggingFace
- Implements three detection tasks
- Handles GPU/CPU device management
- Generates responses with temperature control

**HateSpeechEvaluator Class**:
- Computes binary classification metrics
- Computes multi-class metrics
- Calculates token coverage statistics
- Generates reports and comparisons

**Dataset Format**:
```json
{
  "text": "Text to analyze",
  "has_hate_speech": true/false,
  "category": 0-7
}
```

### Prompt Engineering

Each task uses carefully designed prompts:

**Binary Detection**:
```
Analyze the following text and determine if it contains hate speech.
[Definition]
Text: "{text}"
Answer with either "YES" or "NO" followed by explanation.
```

**Sentence Extraction**:
```
Given the following text, identify and extract ONLY the sentences 
that contain hate speech. List each sentence on a new line.
Text: "{text}"
Hate speech sentences:
```

**Categorization**:
```
[Category descriptions 0-7]
Analyze the following text and classify it into one of the 
categories above (0-7).
Text: "{text}"
Category:
```

## Dependencies

- `transformers>=4.30.0` - HuggingFace Transformers
- `torch>=2.0.0` - PyTorch
- `scikit-learn>=1.3.0` - Metrics
- `numpy>=1.24.0` - Numerical operations
- `pandas>=2.0.0` - Data manipulation
- `tqdm>=4.65.0` - Progress bars
- `accelerate>=0.20.0` - Model loading

## Testing

Comprehensive test suite in `tests/test_basic.py`:
- Category definitions validation
- Binary classification metrics
- Multi-class metrics
- Token coverage calculation
- Dataset format validation

Run tests:
```bash
python tests/test_basic.py
```

Expected output: "✓ ALL TESTS PASSED"

## Documentation

1. **README.md** - Main documentation, installation, usage
2. **QUICKSTART.md** - 5-minute getting started guide
3. **USAGE.md** - Detailed usage examples and tips
4. **CATEGORIES.md** - Category definitions and annotation guidelines
5. **IMPLEMENTATION_NOTES.md** - Technical implementation details
6. **PROJECT_SUMMARY.md** - This comprehensive overview

## Example Dataset

Included in `examples/sample_dataset.json`:
- 10 samples (5 hate speech, 5 non-hate)
- Balanced representation of categories
- Clear positive and negative examples

## Research Applications

This framework is designed for:
- Comparing different LLMs for hate speech detection
- Analyzing token-level hate speech coverage
- Fine-grained categorization of hate speech types
- Developing hate speech detection systems
- Academic research on hate speech
- Model performance benchmarking

## Performance Characteristics

**Inference Speed** (approximate, on RTX 3090):
- Phi-2 (2.7B): ~1-2 seconds per sample
- Mistral-7B: ~3-5 seconds per sample
- Llama-2-7B: ~3-5 seconds per sample
- Llama-2-13B: ~6-10 seconds per sample

**Memory Requirements**:
- 2.7B model: ~6GB GPU (FP16)
- 7B model: ~14GB GPU (FP16)
- 13B model: ~26GB GPU (FP16)

## Evaluation Metrics Details

### For Task 1 & Task 3 (Classification)

**Accuracy**: Proportion of correct predictions
```
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision**: Proportion of positive predictions that are correct
```
precision = TP / (TP + FP)
```

**Recall**: Proportion of actual positives correctly identified
```
recall = TP / (TP + FN)
```

**F1 Score**: Harmonic mean of precision and recall
```
f1 = 2 * (precision * recall) / (precision + recall)
```

### For Task 2 (Token Coverage)

**Token Coverage Ratio**: Proportion of text identified as hate speech
```
coverage_ratio = tokens_in_hate_sentences / total_tokens
```

**Aggregate Statistics**:
- Mean coverage across all samples
- Median, standard deviation
- Min and max coverage ratios

## Quality Assurance

✅ All Python files compile without errors
✅ All unit tests pass
✅ Sample dataset validated
✅ Complete documentation
✅ Example outputs provided
✅ Utilities tested and working

## License

GNU General Public License v3.0 - See LICENSE file

## Contributing

Contributions welcome! Areas for enhancement:
- Batch processing implementation
- Multi-language support
- Additional models
- Fine-tuning scripts
- Web interface
- Additional metrics

## Citation

```bibtex
@software{hatespeech_llm_detector,
  title = {HateSpeech-LLM-Detector: A Framework for Hate Speech Detection using LLMs},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/mdodovic/HateSpeech-LLM-Detector}
}
```

## Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Submit a pull request
- Contact repository maintainers

---

**Status**: ✅ Complete and ready for use

**Last Updated**: 2025-10-11

**Version**: 1.0.0
