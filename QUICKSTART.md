# Quick Start Guide

Get started with HateSpeech-LLM-Detector in 5 minutes!

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/mdodovic/HateSpeech-LLM-Detector.git
cd HateSpeech-LLM-Detector

# Install dependencies
pip install -r requirements.txt
```

## 2. Prepare Your Data

Create a JSON file with your labeled data:

```json
[
  {
    "text": "Example text to analyze",
    "has_hate_speech": false,
    "category": 0
  },
  {
    "text": "Another example with hate speech",
    "has_hate_speech": true,
    "category": 1
  }
]
```

**Category Codes**:
- 0: No hate speech
- 1: Race/Ethnicity
- 2: Religion
- 3: Gender
- 4: Sexual orientation
- 5: Disability
- 6: Nationality
- 7: Other

See [CATEGORIES.md](CATEGORIES.md) for detailed descriptions.

## 3. Run Evaluation

### Option A: Using the Sample Dataset

```bash
python src/main.py \
    --dataset examples/sample_dataset.json \
    --models "microsoft/phi-2" \
    --output-dir results
```

### Option B: Using Your Own Dataset

```bash
python src/main.py \
    --dataset path/to/your/dataset.json \
    --models "microsoft/phi-2" "mistralai/Mistral-7B-Instruct-v0.1" \
    --output-dir my_results
```

## 4. View Results

Results are saved in the output directory:

```bash
cd results
ls -lh
```

**Output Files**:
- `*_detailed_results.csv` - All predictions
- `*_metrics.json` - Evaluation metrics
- `*_classification_report.txt` - Detailed report
- `model_comparison.csv` - Model comparison (if multiple models)

## 5. Try Interactive Demo

```bash
cd examples
python demo.py
```

Select mode and enter a model name (e.g., `microsoft/phi-2`).

## Common Commands

### Run with Multiple Models

```bash
python src/main.py \
    --dataset examples/sample_dataset.json \
    --models "microsoft/phi-2" \
           "mistralai/Mistral-7B-Instruct-v0.1" \
           "Qwen/Qwen2-7B-Instruct" \
    --output-dir multi_model_results
```

### Use CPU Instead of GPU

```bash
python src/main.py \
    --dataset examples/sample_dataset.json \
    --models "microsoft/phi-2" \
    --device cpu \
    --output-dir cpu_results
```

### Force GPU

```bash
python src/main.py \
    --dataset examples/sample_dataset.json \
    --models "microsoft/phi-2" \
    --device cuda \
    --output-dir gpu_results
```

## Recommended Models for Testing

**Small Models (Good for Testing)**:
- `microsoft/phi-2` (2.7B parameters)
- `microsoft/phi-3-mini` (3.8B parameters)

**Medium Models (Better Accuracy)**:
- `mistralai/Mistral-7B-Instruct-v0.1` (7B)
- `Qwen/Qwen2-7B-Instruct` (7B)

**Large Models (Best Accuracy, More Resources)**:
- `meta-llama/Llama-2-7b-chat-hf` (7B, requires access)
- `meta-llama/Llama-2-13b-chat-hf` (13B, requires access)

## Hardware Requirements

| Model Size | GPU RAM | Inference Time (per sample) |
|-----------|---------|---------------------------|
| 2-3B      | 8GB+    | ~1-2 seconds             |
| 7B        | 16GB+   | ~3-5 seconds             |
| 13B       | 24GB+   | ~6-10 seconds            |

*CPU inference is possible but much slower*

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

```bash
pip install torch transformers
```

### "CUDA out of memory"

Use a smaller model or switch to CPU:
```bash
python src/main.py --dataset data.json --models "microsoft/phi-2" --device cpu
```

### "Model requires authentication"

For models like Llama:
```bash
huggingface-cli login
# Then request access on the model's HuggingFace page
```

### "ImportError: cannot import name"

Reinstall dependencies:
```bash
pip install -r requirements.txt --force-reinstall
```

## Next Steps

1. **Read the full documentation**: [README.md](README.md)
2. **Understand categories**: [CATEGORIES.md](CATEGORIES.md)
3. **Learn advanced usage**: [USAGE.md](USAGE.md)
4. **Run tests**: `python tests/test_basic.py`

## Getting Help

- Check [USAGE.md](USAGE.md) for detailed examples
- Review [CATEGORIES.md](CATEGORIES.md) for category definitions
- Open an issue on GitHub for bugs or questions

## Citation

If you use this tool in research, please cite:

```bibtex
@software{hatespeech_llm_detector,
  title = {HateSpeech-LLM-Detector},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/mdodovic/HateSpeech-LLM-Detector}
}
```

---

**Ready to start?** Run the tests to verify everything works:

```bash
python tests/test_basic.py
```

If all tests pass, you're ready to go! 🚀
