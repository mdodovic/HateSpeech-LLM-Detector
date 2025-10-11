# Usage Guide

## Quick Start

### 1. Prepare Your Dataset

Create a JSON file with your labeled data:

```json
[
  {
    "text": "Your text sample here",
    "has_hate_speech": true,
    "category": 1
  }
]
```

### 2. Run Evaluation

```bash
python src/main.py \
    --dataset path/to/your/dataset.json \
    --models "microsoft/phi-2" \
    --output-dir results
```

### 3. View Results

Check the `results/` directory for:
- Detailed predictions CSV
- Metrics JSON
- Classification reports
- Model comparisons

## Advanced Usage

### Using Multiple Models

Compare different LLMs:

```bash
python src/main.py \
    --dataset examples/sample_dataset.json \
    --models "microsoft/phi-2" "mistralai/Mistral-7B-Instruct-v0.1" "Qwen/Qwen2-7B-Instruct" \
    --output-dir multi_model_results
```

### GPU Configuration

Specify device:

```bash
# Automatic device selection (default)
python src/main.py --dataset data.json --models "model-name" --device auto

# Force CPU
python src/main.py --dataset data.json --models "model-name" --device cpu

# Force CUDA
python src/main.py --dataset data.json --models "model-name" --device cuda
```

## Using as a Library

### Basic Detection

```python
from src.llm_detector import LLMDetector
from src.categories import get_category_prompt

# Initialize detector
detector = LLMDetector("microsoft/phi-2")

# Analyze text
text = "Your text to analyze"
categories_prompt = get_category_prompt()
result = detector.analyze_text_complete(text, categories_prompt)

print(f"Has hate speech: {result['has_hate_speech']}")
print(f"Category: {result['category']}")
print(f"Token coverage: {result['token_coverage_ratio']:.2%}")
```

### Task-Specific Usage

```python
from src.llm_detector import LLMDetector
from src.categories import get_category_prompt

detector = LLMDetector("microsoft/phi-2")
text = "Sample text"

# Task 1: Binary detection only
has_hate, explanation = detector.detect_hate_speech_binary(text)

# Task 2: Extract hate speech sentences
sentences, covered, total = detector.extract_hate_speech_sentences(text)

# Task 3: Categorize
category, explanation = detector.categorize_hate_speech(text, get_category_prompt())
```

### Custom Evaluation

```python
from src.evaluation import HateSpeechEvaluator

evaluator = HateSpeechEvaluator()

# Binary classification metrics
y_true = [True, False, True, False]
y_pred = [True, False, False, False]
metrics = evaluator.evaluate_binary_classification(y_true, y_pred)

# Category classification metrics
y_true_cat = [0, 1, 2, 3]
y_pred_cat = [0, 1, 1, 3]
cat_metrics = evaluator.evaluate_multiclass_classification(y_true_cat, y_pred_cat)

# Token coverage
tokens_covered = [10, 20, 0, 15]
total_tokens = [100, 150, 80, 120]
token_metrics = evaluator.evaluate_token_coverage(tokens_covered, total_tokens)
```

## Interactive Demo

### Predefined Texts Demo

```bash
cd examples
python demo.py
# Select option 1
# Enter model name: microsoft/phi-2
```

### Interactive Mode

```bash
cd examples
python demo.py
# Select option 2
# Enter model name: microsoft/phi-2
# Enter your own texts to analyze
```

## Tips for Best Results

1. **Model Selection**: 
   - Start with smaller models (phi-2, phi-3) for testing
   - Use larger models (Llama, Mistral) for better accuracy
   - Ensure you have sufficient GPU memory

2. **Dataset Quality**:
   - Use balanced datasets with diverse examples
   - Include clear positive and negative samples
   - Ensure consistent labeling

3. **Hardware Requirements**:
   - 7B models: 16GB+ GPU RAM
   - 13B models: 24GB+ GPU RAM
   - CPU inference: Possible but very slow

4. **Temperature Settings**:
   - Default (0.1): More deterministic, better for consistency
   - Higher values: More creative but less reproducible

## Troubleshooting

### Out of Memory Errors

```python
# Use smaller models or enable CPU offloading
detector = LLMDetector("microsoft/phi-2")  # Smaller model

# Or use 8-bit quantization (requires bitsandbytes)
# Add to llm_detector.py model loading:
# load_in_8bit=True
```

### Model Access Issues

Some models require authentication:

```bash
# Login to HuggingFace
huggingface-cli login

# Request access to gated models on HuggingFace website
# (e.g., Llama models)
```

### Slow Inference

- Use GPU instead of CPU
- Reduce max_new_tokens in prompts
- Use smaller models for testing
- Batch processing (requires code modification)

## Output Interpretation

### Binary Metrics
- **Accuracy > 0.9**: Excellent performance
- **F1 > 0.85**: Good balance of precision/recall
- **High Precision, Low Recall**: Conservative (misses some hate speech)
- **Low Precision, High Recall**: Aggressive (false positives)

### Category Metrics
- **Macro F1**: Performance across all categories equally
- **Weighted F1**: Performance weighted by category frequency
- Use confusion matrix to identify problematic categories

### Token Coverage
- **High coverage**: Model identifies large portions as hate speech
- **Low coverage**: Model is conservative in extraction
- Compare with binary detection for consistency

## Common Workflows

### 1. Initial Model Selection

```bash
# Test multiple models on small dataset
python src/main.py \
    --dataset examples/sample_dataset.json \
    --models "microsoft/phi-2" "mistralai/Mistral-7B-Instruct-v0.1" \
    --output-dir model_selection
```

### 2. Full Evaluation

```bash
# Run best model on full dataset
python src/main.py \
    --dataset data/full_dataset.json \
    --models "best-model-name" \
    --output-dir final_results
```

### 3. Error Analysis

```python
import pandas as pd

# Load detailed results
df = pd.read_csv("results/model_detailed_results.csv")

# Find misclassifications
errors = df[df['true_has_hate'] != df['pred_has_hate']]
print(errors[['text', 'true_has_hate', 'pred_has_hate']])
```

## Performance Benchmarks

Approximate inference times (on RTX 3090):

| Model | Size | Time per sample |
|-------|------|----------------|
| Phi-2 | 2.7B | ~1-2 seconds |
| Mistral-7B | 7B | ~3-5 seconds |
| Llama-2-7B | 7B | ~3-5 seconds |
| Llama-2-13B | 13B | ~6-10 seconds |

*Times include all three tasks per sample*
