# Software Configuration

Verification date: 2026-07-06. These values describe the current workspace environment. The repository does not contain a run manifest proving that this is the original environment used for every reported experiment.

## Verified from environment

### Operating system and hardware

- OS command `Get-CimInstance Win32_OperatingSystem | Select-Object ...` failed with `Access denied`; the OS was instead verified with `[System.Runtime.InteropServices.RuntimeInformation]`.
- OS: `Microsoft Windows 10.0.26200`, architecture `X64` (`$v=[System.Environment]::OSVersion.VersionString; $d=[System.Runtime.InteropServices.RuntimeInformation]::OSDescription; $a=[System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture`).
- GPU: NVIDIA GeForce RTX 3060, 12288 MiB VRAM; NVIDIA driver `591.86`; CUDA version reported by `nvidia-smi` as `13.1` (`nvidia-smi`).

### Python environment

- Plain `python --version`: `Python 3.13.3`.
- Project virtual environment `.\venv\Scripts\python.exe --version`: `Python 3.13.3`.
- `requirements.txt` was regenerated from `.\venv\Scripts\python.exe -m pip freeze`. Key packages: `torch==2.6.0+cu124`, `transformers==4.57.1`, `scikit-learn==1.7.2`, `scipy==1.16.2`, `pandas==2.3.3`, `numpy==2.3.3`, `requests==2.32.5`, `accelerate==1.10.1`.
- No Gemini SDK package (`google-generativeai` or `google-genai`) was present in the virtualenv freeze.

### Ollama installation and local models

- Ollama version: `0.30.7` (`ollama --version`).
- Installed Ollama models included the configured model tags plus additional models (`ollama list`): `llama3:latest`, `mistral:latest`, `qwen3.5:latest`, `phi3:latest`, `qwen3:latest`, plus `llama3.2-vision`, `deepseek-r1`, `codegemma`, `deepseek-coder`, `qwen2.5-coder`, `granite4`, `gemma3`, and `phi4`.

| Code tag | Resolved local tag | Parameters | Quantization | Context length | Model-level parameters shown by Ollama | Template summary |
| --- | --- | ---: | --- | ---: | --- | --- |
| `llama3` | `llama3:latest` | 8.0B | Q4_0 | 8192 | `num_keep=24`; stop tokens `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>` (`ollama show llama3`; `ollama show --parameters llama3`) | Llama 3 header template with `<|start_header_id|>system/user/assistant...` (`ollama show --template llama3`) |
| `mistral` | `mistral:latest` | 7.2B | Q4_K_M | 32768 | stop tokens `[INST]`, `[/INST]` (`ollama show mistral`; `ollama show --parameters mistral`) | Mistral `[INST] ... [/INST]` chat template (`ollama show --template mistral`) |
| `qwen3.5` | `qwen3.5:latest` | 9.7B | Q4_K_M | 262144 | `presence_penalty=1.5`, `temperature=1`, `top_k=20`, `top_p=0.95` (`ollama show qwen3.5`; `ollama show --parameters qwen3.5`) | Raw prompt template `{{ .Prompt }}` (`ollama show --template qwen3.5`) |
| `phi3` | `phi3:latest` | 3.8B | Q4_0 | 131072 | stop tokens `<|end|>`, `<|user|>`, `<|assistant|>` (`ollama show phi3`; `ollama show --parameters phi3`) | Phi-3 role template with `<|system|>`, `<|user|>`, `<|assistant|>` (`ollama show --template phi3`) |
| `qwen3` | `qwen3:latest` | 8.2B | Q4_K_M | 40960 | `repeat_penalty=1`, `temperature=0.6`, `top_k=20`, `top_p=0.95`; stop tokens `<|im_start|>`, `<|im_end|>` (`ollama show qwen3`; `ollama show --parameters qwen3`) | ChatML-style template with `<|im_start|>` roles, tool block, and thinking/no-thinking support (`ollama show --template qwen3`) |

## Extracted from code

### Ollama inference configuration

- Ollama calls are centralized in `src/llm_detector.py`; no other direct REST/client Ollama caller was found.
- `LLMDetector.__init__` sets `base_url="http://localhost:11434"`, `default_temperature=0.1`, `default_max_tokens=1024`, and `default_seed=42` unless overridden (`src/llm_detector.py:16`).
- A `requests.Session` is used (`src/llm_detector.py:31`), and `_post` calls `self._session.post(url, json=payload)` with no timeout argument (`src/llm_detector.py:61-62`).
- The only Ollama `options` passed by code are `temperature`, `num_predict`, and, when not `None`, `seed` (`src/llm_detector.py:89-93`). The code does not pass `top_p`, `top_k`, `num_ctx`, `repeat_penalty`, `presence_penalty`, JSON `format`, or request timeout.
- The primary endpoint is `/api/chat` with `stream=False`, `model`, `messages`, and `options` (`src/llm_detector.py:98-104`). If unavailable, it falls back to `/api/generate` with `stream=False`, `model`, `prompt`, and `options` (`src/llm_detector.py:107-116`).
- High-level methods pass `max_new_tokens=self.max_tokens` and `temperature=self.default_temperature`, so the active task calls use `num_predict=1024` and `temperature=0.1` by default (`src/llm_detector.py:170`, `src/llm_detector.py:189`, `src/llm_detector.py:240`, `src/llm_detector.py:294`, `src/llm_detector.py:390`).
- `generate_response` resolves the generation seed from `self.default_seed` unless an explicit seed is passed (`src/llm_detector.py:146-156`).
- Model tags are loaded from `models/models.json`: `llama -> llama3`, `mistral -> mistral`, `qwen35 -> qwen3.5`, `phi3 -> phi3`, `qwen3 -> qwen3` (`models/models.json:2-6`; loader in `src/utils.py:225-291`).
- Main single-sentence zero-shot and few-shot entry points currently restrict models to `["llama", "qwen3"]` (`single_sentence_run.py:506-510`; `single_sentence_few_shot_run.py:324-328`).
- Full-text non-ensemble evaluation uses all configured models when no model subset is passed (`full_text_run.py:205-228`; `src/utils.py:271-291`).
- Single-sentence ensemble uses all configured models because `MODEL_SUBSET=[]`, then creates one `LLMDetector` per model (`single_sentence_run_ensemble.py:31-32`, `single_sentence_run_ensemble.py:76-80`).
- Full-text ensemble uses all configured models because `MODEL_SUBSET=[]`, with explicit `SEED=42` (`full_text_run_ensemble.py:34-35`, `full_text_run_ensemble.py:151-155`).
- Ensemble concurrency uses `ThreadPoolExecutor(max_workers=len(detectors))` in both ensemble scripts (`single_sentence_run_ensemble.py:149-152`; `full_text_run_ensemble.py:179-195`).
- Seed-analysis runs repeat Ollama experiments over `[42, 123, 2024, 3407, 271828]` (`seed_analysis.py:36`) and passes each seed into the single-sentence, full-text, and full-text-ensemble evaluators (`seed_analysis.py:338-351`).

### BERTic fine-tuning configuration

- The HuggingFace baseline uses `AutoTokenizer`, `AutoModelForSequenceClassification`, `TrainingArguments`, `Trainer`, and `EarlyStoppingCallback` (`finetune_bertic_binary.py:31-37`; `finetune_bertic_categories.py:31-37`; `finetune_bertic_subcategories.py:21-27`; `cv_bertic.py:48-52`).
- Base model and tokenizer checkpoint: `classla/bcms-bertic` for binary, category, subcategory, ternary, and CV scripts (`finetune_bertic_binary.py:268-272`; `finetune_bertic_categories.py:248-252`; `finetune_bertic_subcategories.py:248-252`; `finetune_bertic_ternary.py:273-277`; `cv_bertic.py:59`, `cv_bertic.py:297-300`).
- Input format is paragraph context plus target sentence; tokenization uses `truncation="only_first"`, `padding="max_length"`, and `max_length=args.max_length` (`finetune_bertic_binary.py:88-95`; `finetune_bertic_categories.py:115-122`; `finetune_bertic_subcategories.py:115-122`; `cv_bertic.py:196-203`).
- Binary/category/subcategory defaults: `epochs=10`, `batch_size=8`, `lr=5e-5`, `max_length=512`, `val_split=0.15`, `seed=42`, `dropout=0.1`, `weight_decay=0.01`, `label_smoothing=0.1`, `gradient_accumulation_steps=4`, and freeze strategy `none` (`finetune_bertic_binary.py:165-186`; `finetune_bertic_categories.py:145-168`; `finetune_bertic_subcategories.py:145-168`).
- Ternary defaults differ: `batch_size=16`, `lr=2e-5`, `max_length=512`, `epochs=10`, `val_split=0.15`, `seed=42`; no dropout, label smoothing, or gradient accumulation argument is defined in that script (`finetune_bertic_ternary.py:181-202`).
- CV defaults: `task=binary`, `k=3`, `epochs=10`, `batch_size=8`, `lr=5e-5`, `max_length=512`, `seed=42`, `dropout=0.1`, `weight_decay=0.01`, `label_smoothing=0.1`, `gradient_accumulation_steps=4` (`cv_bertic.py:408-423`).
- Dropout is passed as both `classifier_dropout` and `hidden_dropout_prob` in the binary/category/subcategory/CV scripts (`finetune_bertic_binary.py:271-275`; `finetune_bertic_categories.py:251-255`; `finetune_bertic_subcategories.py:251-255`; `cv_bertic.py:298-302`).
- TrainingArguments use epoch evaluation/saving/logging, `load_best_model_at_end=True`, `metric_for_best_model="f1"` for binary/category/subcategory/CV and `"f1_macro"` for ternary, `greater_is_better=True`, `fp16=torch.cuda.is_available()`, `seed=args.seed`, and `report_to="none"` (`finetune_bertic_binary.py:324-344`; `finetune_bertic_categories.py:298-318`; `finetune_bertic_subcategories.py:298-318`; `finetune_bertic_ternary.py:292-310`; `cv_bertic.py:311-331`).
- Warmup uses `warmup_ratio=0.15` for binary/category/subcategory/CV and `warmup_ratio=0.1` for ternary (`finetune_bertic_binary.py:333`; `finetune_bertic_categories.py:307`; `finetune_bertic_subcategories.py:307`; `finetune_bertic_ternary.py:299`; `cv_bertic.py:320`).
- No explicit `optim` or `lr_scheduler_type` is set in the code; HuggingFace `Trainer` defaults apply under the installed Transformers version.
- Loss weighting uses `sklearn.utils.class_weight.compute_class_weight("balanced", ...)` and a custom `WeightedTrainer` with `torch.nn.CrossEntropyLoss(weight=...)` (`finetune_bertic_binary.py:263-264`, `finetune_bertic_binary.py:349-356`; `finetune_bertic_categories.py:240-242`, `finetune_bertic_categories.py:323-333`; `finetune_bertic_subcategories.py:240-242`, `finetune_bertic_subcategories.py:323-333`; `finetune_bertic_ternary.py:268-269`, `finetune_bertic_ternary.py:313-318`; `cv_bertic.py:288-294`, `cv_bertic.py:216-224`).
- Label smoothing is applied in the custom loss for binary/category/subcategory (`finetune_bertic_binary.py:353-356`; `finetune_bertic_categories.py:330-333`; `finetune_bertic_subcategories.py:330-333`). In `cv_bertic.py`, `label_smoothing_factor=args.label_smoothing` is set in `TrainingArguments`, but the overridden custom loss does not pass `label_smoothing` (`cv_bertic.py:319`; `cv_bertic.py:216-224`).
- Early stopping uses `EarlyStoppingCallback(early_stopping_patience=3)` (`finetune_bertic_binary.py:365`; `finetune_bertic_categories.py:342`; `finetune_bertic_subcategories.py:342`; `finetune_bertic_ternary.py:326`; `cv_bertic.py:340`).
- Train/validation splitting uses `train_test_split(..., test_size=args.val_split, random_state=args.seed, stratify=...)` for binary and ternary (`finetune_bertic_binary.py:247-249`; `finetune_bertic_ternary.py:252-254`). Category/subcategory use `train_test_split(..., random_state=args.seed)` without a `stratify` argument (`finetune_bertic_categories.py:222-225`; `finetune_bertic_subcategories.py:222-225`).
- Cross-validation uses `StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=args.seed)`, grouping sentences by paragraph ID (`cv_bertic.py:462`; paragraph grouping described in `cv_bertic.py:4-12`).

### Gemini baseline code

- The repository contains evaluation of saved Gemini-style predictions, not the Gemini API generation call (`README.md:231-242`; `gemini_results.py:1-9`).
- `gemini_results.py` reads a ground-truth Excel file and a saved LLM prediction Excel file, aligned by row index (`gemini_results.py:79-93`; CLI defaults at `gemini_results.py:186-193`).
- No exact Gemini API model string, SDK version, or generation configuration was found in repository source files. This is also consistent with the virtualenv freeze, which contains no Google Gemini SDK package.

### Dataset processing and evaluation

- Dataset expansion and checking scripts use a regex sentence splitter based on punctuation (`.`, `!`, `?`, ellipsis) followed by optional punctuation/whitespace and a lookahead for Latin, Serbian Latin, Cyrillic, digits, punctuation, or emoji ranges (`data/paragraph_dataset_single_sentence_creator.py:7-15`; `data/paragraph_dataset_llm_creator.py:8-16`; `data/paragraph_checker_annotators.py:8-20`; `dataset_analysis.py:61-71`).
- Full-text LLM evaluation asks the model to output sentence-level classifications via `classify_all_sentences`; parsing is based on semicolon-separated model output lines (`src/llm_detector.py:374-420`). The full-text ensemble falls back to a regex splitter only when no model produces sentence entries (`full_text_run_ensemble.py:124-138`, `full_text_run_ensemble.py:218-227`).
- Shared evaluation uses scikit-learn `accuracy_score`, `f1_score`, `confusion_matrix`, and `classification_report` (`src/evaluation.py:9-14`, `src/evaluation.py:23-37`, `src/evaluation.py:52-65`).
- Additional single-sentence and full-text category/subcategory matching uses custom logic around parsed category codes, including exact, category-only, and multi-label success checks (`single_sentence_run.py:126-186`; `full_text_run.py:92-147`; `full_text_run_ensemble.py:281-308`).
- Bootstrap CIs use 10,000 percentile bootstrap resamples, CI `0.95`, seed `42`, `np.random.default_rng(seed)`, and scikit-learn metric functions (`bootstrap_ci.py:29-35`, `bootstrap_ci.py:46-78`, `bootstrap_ci.py:112-120`, `bootstrap_ci.py:184-228`, `bootstrap_ci.py:231-269`, `bootstrap_ci.py:274-388`).
- McNemar uses SciPy's exact two-sided binomial test via `scipy.stats.binomtest`; discordant pairs are `b10` and `b01`, with an uncorrected chi-square statistic also reported (`mcnemar_test.py:1-6`, `mcnemar_test.py:37-40`, `mcnemar_test.py:220-268`, `mcnemar_test.py:271-308`).

## Flagged Inconsistencies And Reproducibility Gaps

- Gemini API reproducibility gap: no Gemini API call, exact Gemini model string, SDK version, or generation config is present in the repo; only saved prediction evaluation is available (`gemini_results.py:79-93`; `README.md:231-242`).
- `models/models.json` currently configures `qwen3.5` and omits `deepseek-r1`/`phi4` (`models/models.json:2-6`), while the README example lists `deepseek` and `phi4` and does not list `qwen3.5` (`README.md:77-87`). The seed-analysis example also includes `deepseek` (`seed_analysis.py:14`), but with the current `models/models.json`, `--models deepseek` would fall back to the literal tag `deepseek`, not the installed `deepseek-r1:latest` (`src/utils.py:271-291`).
- The code passes only `temperature`, `num_predict`, and `seed` to Ollama (`src/llm_detector.py:89-93`). Model-level defaults therefore differ across models for unpassed parameters: for example, `qwen3` has `top_k=20`, `top_p=0.95`, `repeat_penalty=1`, and `qwen3.5` has `presence_penalty=1.5`, `top_k=20`, `top_p=0.95` (`ollama show --parameters qwen3`; `ollama show --parameters qwen3.5`), while Llama/Mistral/Phi show no comparable sampling parameters.
- No `num_ctx` is passed in REST payloads, despite large model context lengths reported by `ollama show`; any paper statement implying an explicit context-window setting should be revised.
- No JSON `format`/structured-output mode and no HTTP request timeout are configured (`src/llm_detector.py:61-62`, `src/llm_detector.py:98-116`).
- Direct calls to `single_sentence_run.two_prompts_evaluation_model_on_records(..., seed=None)` or `one_prompt_evaluation_model_on_records(..., seed=None)` omit the Ollama seed because those helper defaults are `None`; the top-level `run(...)` passes seed `42` by default (`single_sentence_run.py:29-34`, `single_sentence_run.py:246-250`, `single_sentence_run.py:399-423`).
- In `cv_bertic.py`, label smoothing is declared in `TrainingArguments`, but the custom weighted loss ignores `label_smoothing`; this likely makes CV label smoothing ineffective (`cv_bertic.py:319`, `cv_bertic.py:216-224`).
- The default data paths in `single_sentence_run_ensemble.py` and `full_text_run_ensemble.py` refer to `data/single_sentence_hate_speech.xlsx` and `data/paragraph_hate_speech.xlsx`, which were not present in the current `data` directory listing; seed-analysis uses the existing `*_no_offenses.xlsx` paths (`single_sentence_run_ensemble.py:29`; `full_text_run_ensemble.py:32`; `seed_analysis.py:41-45`).

## Software Configuration

The experiments were executed in the current verified environment using Python 3.13.3 with PyTorch 2.6.0+cu124, Transformers 4.57.1, scikit-learn 1.7.2, SciPy 1.16.2, pandas 2.3.3, NumPy 2.3.3, and Requests 2.32.5. Local LLM inference was performed through Ollama 0.30.7 using the model tags `llama3` (8.0B, Q4_0), `mistral` (7.2B, Q4_K_M), `qwen3.5` (9.7B, Q4_K_M), `phi3` (3.8B, Q4_0), and `qwen3` (8.2B, Q4_K_M), as resolved to the locally installed `:latest` variants. The Ollama REST payload explicitly set `temperature=0.1`, `num_predict=1024`, `stream=False`, and a generation seed when supplied; it did not explicitly set `top_p`, `top_k`, `num_ctx`, `repeat_penalty`, JSON output format, or an HTTP timeout. Seeded LLM runs used the default seed 42, and the seed-analysis protocol repeated runs over seeds 42, 123, 2024, 3407, and 271828. The BERTic baseline used the HuggingFace checkpoint and tokenizer `classla/bcms-bertic`, trained via `Trainer` with weighted cross-entropy, max sequence length 512, batch size 8 for binary/category/subcategory tasks, learning rate 5e-5, 10 epochs, warmup ratio 0.15, gradient accumulation 4, dropout 0.1, label smoothing 0.1 where applied by the custom loss, and early stopping with patience 3. GPU execution was available on an NVIDIA GeForce RTX 3060 with driver 591.86 and CUDA 13.1 as reported by `nvidia-smi`. The Gemini baseline was evaluated from saved prediction spreadsheets; the repository does not contain a verified Gemini API model version or generation configuration, so no Gemini model version is reported here.
