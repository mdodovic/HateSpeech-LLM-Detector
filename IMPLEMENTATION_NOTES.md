# Implementation Notes

This document provides technical details about the implementation of HateSpeech-LLM-Detector.

## Architecture Overview

The framework is organized into modular components:

```
src/
├── __init__.py          # Package initialization and exports
├── categories.py        # Category definitions and prompts
├── llm_detector.py      # Core LLM detection class
├── evaluation.py        # Metrics and evaluation functions
├── utils.py            # Dataset utilities
└── main.py             # Command-line interface
```

## Core Components

### 1. LLMDetector (`llm_detector.py`)

**Purpose**: Wrapper class for LLM-based hate speech detection.

**Key Methods**:
- `detect_hate_speech_binary()`: Task 1 - Binary classification
- `extract_hate_speech_sentences()`: Task 2 - Sentence extraction
- `categorize_hate_speech()`: Task 3 - Multi-class categorization
- `analyze_text_complete()`: Run all three tasks

**Design Decisions**:
- Uses HuggingFace Transformers for model loading
- Supports automatic device selection (CUDA/CPU)
- Temperature-based sampling for controlled generation
- Token-level analysis for coverage metrics

### 2. HateSpeechEvaluator (`evaluation.py`)

**Purpose**: Comprehensive evaluation metrics.

**Metrics Implemented**:
- Binary classification: accuracy, precision, recall, F1
- Multi-class: macro/weighted metrics for all categories
- Token coverage: mean, median, std, min, max coverage ratios

**Design Decisions**:
- Uses scikit-learn for standard metrics
- Provides both aggregate and per-class metrics
- Generates classification reports and confusion matrices

### 3. Category System (`categories.py`)

**Purpose**: Define and describe hate speech categories.

**Categories (0-7)**:
0. No hate speech
1. Race/Ethnicity
2. Religion
3. Gender
4. Sexual orientation
5. Disability
6. Nationality
7. Other

**Design Decisions**:
- Comprehensive descriptions for each category
- Generated prompts include full category context
- Based on international hate speech definitions

### 4. Utilities (`utils.py`)

**Purpose**: Dataset management and validation.

**Features**:
- Load/save JSON and CSV formats
- Format conversion between JSON/CSV
- Dataset validation with error checking
- Dataset merging and statistics

## Task Implementations

### Task 1: Binary Detection

**Approach**: 
- Prompt engineering with clear hate speech definition
- Parse "YES"/"NO" from LLM response
- Fallback to keyword detection for robustness

**Prompt Template**:
```
Analyze the following text and determine if it contains hate speech.
[Definition of hate speech]

Text: "{text}"

Answer with either "YES" or "NO" followed by a brief explanation.
```

### Task 2: Sentence Extraction

**Approach**:
- Instruction-based extraction
- Parse line-separated sentences
- Calculate token coverage using tokenizer

**Metrics**:
- Tokens covered: Total tokens in extracted sentences
- Total tokens: Total tokens in original text
- Coverage ratio: tokens_covered / total_tokens

**Prompt Template**:
```
Given the following text, identify and extract ONLY the sentences 
that contain hate speech. List each sentence on a new line.

Text: "{text}"

Hate speech sentences:
```

### Task 3: Categorization

**Approach**:
- Provide full category descriptions in prompt
- Extract category number (0-7) from response
- Use regex to find first category number mentioned

**Prompt Template**:
```
[Category descriptions 0-7]

Analyze the following text and classify it into one of the 
categories above (0-7).

Text: "{text}"

Category:
```

## Evaluation Methodology

### Binary Classification Metrics

- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1**: 2 * (Precision * Recall) / (Precision + Recall)

Where:
- TP: True Positives (correctly identified hate speech)
- TN: True Negatives (correctly identified non-hate speech)
- FP: False Positives (non-hate speech classified as hate)
- FN: False Negatives (hate speech missed)

### Multi-class Metrics

**Macro Averaging**: 
- Calculate metrics for each class independently
- Average across all classes (equal weight)
- Best for imbalanced datasets

**Weighted Averaging**:
- Calculate metrics for each class
- Weight by class frequency
- Reflects overall dataset performance

### Token Coverage

**Calculation**:
```python
for each text:
    coverage_ratio = tokens_in_extracted_sentences / total_tokens_in_text

mean_coverage = average of all coverage_ratios
```

**Interpretation**:
- 0.0: No hate speech detected
- 0.1-0.3: Conservative extraction (typical)
- 0.3-0.5: Moderate extraction
- 0.5+: Liberal extraction

## Model Support

### Supported Model Families

1. **Phi** (Microsoft): 2.7B-3.8B parameters
2. **Mistral**: 7B parameters
3. **Llama** (Meta): 7B-13B parameters
4. **Qwen**: 7B parameters
5. **DeepSeek**: 7B parameters

### Model Loading

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # GPU
    # torch_dtype=torch.float32  # CPU
    device_map="auto",
    trust_remote_code=True
)
```

**Memory Optimization**:
- FP16 precision on GPU (reduces memory by 50%)
- Automatic device mapping
- Gradient computation disabled (inference only)

## Dataset Format

### JSON Format (Primary)

```json
[
  {
    "text": "Text to analyze",
    "has_hate_speech": true,
    "category": 1
  }
]
```

### CSV Format (Alternative)

```csv
text,has_hate_speech,category
"Text to analyze",true,1
"Another text",false,0
```

**Requirements**:
- `text`: String, required
- `has_hate_speech`: Boolean, required
- `category`: Integer 0-7, required
- Consistency: category=0 when has_hate_speech=false

## Performance Considerations

### Inference Speed

**Factors**:
- Model size (larger = slower)
- Sequence length (longer = slower)
- Hardware (GPU >> CPU)
- Batch size (not implemented, sequential processing)

**Optimization Opportunities**:
- Implement batch processing
- Use quantization (8-bit, 4-bit)
- Cache model weights
- Parallel processing for multiple models

### Memory Usage

**GPU Memory**:
- 2.7B model: ~6GB (FP16)
- 7B model: ~14GB (FP16)
- 13B model: ~26GB (FP16)

**CPU Memory**:
- 2x GPU memory (FP32)

## Error Handling

### Model Loading Errors
- Check HuggingFace access tokens
- Verify model name spelling
- Ensure sufficient memory

### Parsing Errors
- Fallback to keyword detection
- Default to category 0 if parsing fails
- Log unparseable responses

### Dataset Errors
- Validation before processing
- Clear error messages
- Continue processing on individual sample errors

## Extension Points

### Adding New Models

1. Ensure HuggingFace compatibility
2. Test on sample data
3. Add to recommended models list
4. Document memory requirements

### Adding New Categories

1. Update `HATE_SPEECH_CATEGORIES` dict
2. Update `CATEGORY_DESCRIPTIONS` dict
3. Update validation (max category number)
4. Update documentation

### Custom Prompts

Modify prompt templates in `llm_detector.py`:
- `detect_hate_speech_binary()`: Binary detection prompt
- `extract_hate_speech_sentences()`: Extraction prompt
- `categorize_hate_speech()`: Categorization prompt

### Custom Metrics

Add to `evaluation.py`:
```python
def custom_metric(y_true, y_pred):
    # Your metric calculation
    return value
```

## Testing

### Unit Tests

Located in `tests/test_basic.py`:
- Category definitions
- Binary classification metrics
- Multi-class metrics
- Token coverage calculation
- Dataset format validation

**Run tests**:
```bash
python tests/test_basic.py
```

### Integration Testing

Test with sample dataset:
```bash
python src/main.py \
    --dataset examples/sample_dataset.json \
    --models "microsoft/phi-2" \
    --output-dir test_results
```

## Known Limitations

1. **Sequential Processing**: No batch inference (slow for large datasets)
2. **Prompt Sensitivity**: Results vary with prompt wording
3. **Language**: Primarily designed for English text
4. **Model Availability**: Some models require HuggingFace access
5. **Parsing Reliability**: LLM responses may not always follow format

## Future Enhancements

1. **Batch Processing**: Implement batched inference
2. **Quantization**: Add 8-bit/4-bit quantization support
3. **Multi-language**: Support for other languages
4. **Fine-tuning**: Tools for model fine-tuning
5. **UI**: Web interface for interactive use
6. **Caching**: Cache model predictions
7. **Ensemble**: Combine multiple model predictions

## References

- HuggingFace Transformers: https://huggingface.co/docs/transformers
- scikit-learn Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- Hate Speech Research: See CATEGORIES.md for references

## Version History

### v1.0.0 (2025)
- Initial release
- Support for multiple LLMs
- Three-task framework
- Comprehensive evaluation metrics
- Dataset utilities
- Documentation and examples
