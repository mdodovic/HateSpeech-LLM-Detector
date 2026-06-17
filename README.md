# HateSpeech-LLM-Detector

Research code for Serbian hate-speech detection and fine-grained categorization. The project compares local LLM prompting through Ollama, evaluates external LLM predictions, analyzes annotated Excel datasets, and fine-tunes BERTic classifiers for sentence-level hate-speech tasks.

The code is organized around annotated paragraph and single-sentence datasets where labels use compact category codes such as `0`, `1a`, `3b`, or `6c`.

## Main Capabilities

- Binary hate-speech detection: hate speech vs. no hate speech.
- Ternary classification for datasets with offenses: no offense, offense `U`, hate speech.
- Top-level category classification: classes `0` through `7`.
- Subcategory classification: `0`, `1a`, `1b`, `1c`, `2`, `3a`, `3b`, `4a`, `4b`, `5`, `6a`, `6b`, `6c`, `7`.
- Single-sentence LLM evaluation with one-prompt, two-prompt, few-shot, and ensemble variants.
- Full-text paragraph evaluation by splitting paragraphs into sentences.
- BERTic fine-tuning for binary, ternary, category, and subcategory classifiers.
- Dataset analysis, plotting, annotator agreement, and dataset conversion utilities.

## Repository Layout

```text
.
|-- data/                         # Excel datasets and dataset utilities
|-- models/                       # Ollama model configuration
|-- results/                      # Generated Excel reports, plots, and old results
|-- src/
|   |-- categories.py             # Hate-speech category and subcategory definitions
|   |-- evaluation.py             # Shared metric helpers
|   |-- llm_detector.py           # Ollama-backed detector
|   |-- prompts/                  # Prompt templates
|   `-- utils.py                  # Dataset/model parsing helpers
|-- finetune_bertic_*.py          # BERTic fine-tuning scripts
|-- *_run.py                      # LLM evaluation scripts
|-- dataset_analysis.py           # Dataset statistics and plots
|-- gemini_results.py             # Evaluation of saved Gemini predictions
`-- requirements.txt
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS, activate with:

```bash
source venv/bin/activate
```

The fine-tuning scripts use PyTorch and Hugging Face Transformers. The pinned `requirements.txt` includes a CUDA build of PyTorch; adjust the PyTorch packages if your machine needs a CPU-only or different CUDA build.

## Ollama Setup

The LLM evaluation scripts use a local Ollama server at `http://localhost:11434`.

Install Ollama, then pull the models listed in `models/models.json`:

```bash
ollama pull llama3
ollama pull mistral
ollama pull deepseek-r1
ollama pull phi3
ollama pull qwen3
ollama pull phi4
```

Start the server:

```bash
ollama serve
```

Model display names and Ollama tags are configured in [models/models.json](models/models.json):

```json
{
  "llama": "llama3",
  "mistral": "mistral",
  "deepseek": "deepseek-r1",
  "phi3": "phi3",
  "qwen3": "qwen3",
  "phi4": "phi4"
}
```

## Label Schema

Category definitions live in [src/categories.py](src/categories.py).

| Code | Meaning |
| --- | --- |
| `0` | No hate speech |
| `1` | Racial and ethnic/national hate |
| `1a` | Race / skin color |
| `1b` | Ethnic affiliation |
| `1c` | Nationality / origin |
| `2` | Religious hate |
| `3` | Sex- and gender-based hate |
| `3a` | Sex / sexism |
| `3b` | LGBTQ+ identities |
| `4` | Physical traits and health-based hate |
| `4a` | Physical appearance |
| `4b` | Illness / disability |
| `5` | Age- and generation-based hate |
| `6` | Socioeconomic hate |
| `6a` | Socioeconomic status / class |
| `6b` | Occupation / profession |
| `6c` | Political intolerance |
| `7` | Sports and fan-based hate |
| `U` | Offense, used by the offense-aware datasets/scripts |

Several scripts support multi-label annotations in cells, for example `4a,6a` or parenthesized sentence-level groups such as `(6c;0), 0, 1a`.

## Data Files

Current tracked datasets include:

- `data/access_paragraph_hate_speech_with_offenses.xlsx` - source paragraph annotations with offense labels.
- `data/paragraph_hate_speech_offenses.xlsx` - paragraph dataset preserving offense labels.
- `data/paragraph_hate_speech_no_offenses.xlsx` - paragraph dataset where offenses are treated as no hate.
- `data/single_sentence_hate_speech_offenses.xlsx` - sentence dataset preserving offense labels.
- `data/single_sentence_hate_speech_no_offenses.xlsx` - sentence dataset where offenses are treated as no hate.
- `data/single_sentence_llm_predictions.xlsx` - saved external LLM predictions for single-sentence evaluation.
- `data/paragraph_llm_predictions.xlsx` - saved external LLM predictions for paragraph evaluation.

The legacy LLM scripts default to `data/single_sentence_hate_speech.xlsx` and `data/paragraph_hate_speech.xlsx`. If those files are not present, either generate them with the dataset utility below or edit the script call at the bottom of the file to use one of the existing `*_offenses.xlsx` or `*_no_offenses.xlsx` files.

Expected dataset columns are usually:

- `ID`
- `Text`
- `Category`

The generic loader also accepts lowercase variants like `text` and `category`.

## Prompt-Based LLM Evaluation

### Single-Sentence Evaluation

Runs all models from `models/models.json` and compares the two-prompt flow against the one-prompt flow:

```bash
python single_sentence_run.py
```

Default output:

```text
results/single_sentence_comparison_multiple_cat.xlsx
```

If `data/single_sentence_hate_speech.xlsx` is missing, edit the `run(...)` call at the bottom of [single_sentence_run.py](single_sentence_run.py), for example:

```python
run(excel_path="data/single_sentence_hate_speech_no_offenses.xlsx")
```

### Few-Shot Single-Sentence Evaluation

Uses `src/prompts/classify_few_shot.txt` for category classification:

```bash
python single_sentence_few_shot_run.py
```

Default output:

```text
results/single_sentence_few_shot.xlsx
```

The current script loops over debug sizes and uses only `llama` by default; edit the `__main__` block to change the dataset, models, or debug range.

### Single-Sentence Ensemble

Runs majority voting across the configured model subset:

```bash
python single_sentence_run_ensemble.py
```

Default output:

```text
results/single_sentence_ensemble.xlsx
```

Configuration constants are at the top of [single_sentence_run_ensemble.py](single_sentence_run_ensemble.py):

- `DATASET_PATH`
- `RESULTS_XLSX`
- `USE_ONE_PROMPT`
- `MODEL_SUBSET`

### Full-Text Evaluation

Classifies every sentence in each paragraph using `src/prompts/classify_full_all.txt`:

```bash
python full_text_run.py
```

Default output:

```text
results/full_text_comparison.xlsx
```

As with the single-sentence script, update the `run(...)` call if `data/paragraph_hate_speech.xlsx` is not present.

### Full-Text Ensemble

Runs majority voting per sentence:

```bash
python full_text_run_ensemble.py
```

Default output:

```text
results/full_text_ensemble.xlsx
```

Configuration constants are at the top of [full_text_run_ensemble.py](full_text_run_ensemble.py).

## Evaluating Saved Gemini Predictions

Compare saved Gemini-style predictions against ground truth:

```bash
python gemini_results.py \
  --gt data/single_sentence_hate_speech_no_offenses.xlsx \
  --llm data/single_sentence_llm_predictions.xlsx \
  --output results/gemini_results.xlsx
```

The script reports binary accuracy/F1, category accuracy/F1, subcategory accuracy/F1, and a category classification report.

## BERTic Fine-Tuning

The fine-tuning scripts train `classla/bcms-bertic` sequence classifiers using paragraph context plus the target sentence:

```text
[CLS] full_paragraph [SEP] target_sentence [SEP]
```

Binary hate speech:

```bash
python finetune_bertic_binary.py
```

Ternary no-offense/offense/hate:

```bash
python finetune_bertic_ternary.py
```

Eight-way top-level category classification:

```bash
python finetune_bertic_categories.py
```

Fourteen-way subcategory classification:

```bash
python finetune_bertic_subcategories.py
```

Common options:

```bash
python finetune_bertic_categories.py \
  --epochs 10 \
  --batch_size 8 \
  --lr 5e-5 \
  --sentence_path data/single_sentence_hate_speech_no_offenses.xlsx \
  --paragraph_path data/paragraph_hate_speech_no_offenses.xlsx \
  --output_dir bertic_finetuned_categories \
  --output results/bertic/bertic_categories_results.xlsx
```

Additional options include `--freeze`, `--dropout`, `--weight_decay`, `--label_smoothing`, `--gradient_accumulation_steps`, `--max_length`, `--val_split`, and `--seed` depending on the script.

Hyperparameter search helpers are available as PowerShell scripts:

- `hp_search_binary.ps1`
- `hp_freeze_search_binary.ps1`
- `hp_search_freeze.ps1`

## Dataset Utilities

Build paragraph-level and sentence-level datasets from the annotator spreadsheet:

```bash
python data/paragraph_dataset_single_sentence_creator.py \
  --file data/access_paragraph_hate_speech_with_offenses.xlsx \
  --output data/single_sentence_hate_speech.xlsx \
  --paragraph_output data/paragraph_hate_speech.xlsx
```

To replace offense label `U` with `0`:

```bash
python data/paragraph_dataset_single_sentence_creator.py \
  --remove_offense \
  --output data/single_sentence_hate_speech_no_offenses.xlsx \
  --paragraph_output data/paragraph_hate_speech_no_offenses.xlsx
```

Check whether a supposed single-sentence file contains multi-sentence rows:

```bash
python data/single_sentence_checker.py --file data/single_sentence_hate_speech_no_offenses.xlsx --not_ok
```

Other helper scripts in `data/` support paragraph checks and LLM prediction dataset creation.

## Dataset Analysis

Run the full dataset analysis:

```bash
python dataset_analysis.py
```

Defaults:

- Full-text dataset: `data/paragraph_hate_speech_offenses.xlsx`
- Single-sentence dataset: `data/single_sentence_hate_speech_offenses.xlsx`
- Annotator dataset: `data/access_paragraph_hate_speech_with_offenses.xlsx`
- Output workbook: `results/complete_dataset_analysis.xlsx`
- Plots: `results/plots/`

## Prompt Templates

Prompt files are in [src/prompts](src/prompts):

- `system_prompt.txt` - shared system prompt.
- `detect.txt` - binary detection.
- `classify.txt` - category/subcategory classification.
- `classify_few_shot.txt` - few-shot category/subcategory classification.
- `detect_and_classify.txt` - one-call binary and category classification.
- `classify_full_all.txt` - sentence-by-sentence paragraph classification.

## Outputs

Most scripts write Excel reports under `results/`. Existing result groups include:

- `results/single_sentence_comparison_multiple_cat.xlsx`
- `results/single_sentence_ensemble.xlsx`
- `results/single_sentence_few_shot.xlsx`
- `results/full_text_comparison.xlsx`
- `results/full_text_ensemble.xlsx`
- `results/gemini_results.xlsx`
- `results/bertic/`
- `results/plots/`

Fine-tuned model checkpoints are written to directories such as:

- `bertic_finetuned_binary/`
- `bertic_finetuned_categories/`
- `bertic_finetuned_subcategories/`
- `bertic_finetuned_ternary/`

## Troubleshooting

- `Connection refused`: make sure `ollama serve` is running.
- `model not found`: run `ollama pull <model-tag>` for each tag in `models/models.json`.
- Missing `data/single_sentence_hate_speech.xlsx` or `data/paragraph_hate_speech.xlsx`: generate them with `paragraph_dataset_single_sentence_creator.py`, or change the script default to an existing dataset.
- Sentence count mismatch in full-text evaluation: the code compares ground-truth labels to its regex sentence splitter and warns when counts differ.
- Slow LLM evaluation: reduce `MODEL_SUBSET`, use a smaller `debug` value, or test with one model first.
- CUDA/PyTorch install issues: install the PyTorch build matching your hardware from the official PyTorch instructions, then reinstall the remaining requirements.

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE).

## Disclaimer

This repository is for research and educational use. Hate-speech detection is sensitive and model outputs can be wrong, especially on ambiguous or context-dependent language. Do not use these predictions as the sole basis for moderation or enforcement decisions.
