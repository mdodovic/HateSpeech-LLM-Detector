# Hardware Configuration

Verification date: 2026-07-06. These values describe the current machine observed from this workspace. The repository does not contain a separate hardware run manifest proving that this exact hardware was used for every historical result file.

## Verified from environment

### Operating system

- OS via .NET runtime APIs: `Microsoft Windows 10.0.26200`, architecture `X64` (`$v=[System.Environment]::OSVersion.VersionString; $d=[System.Runtime.InteropServices.RuntimeInformation]::OSDescription; $a=[System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture`).
- Python/psutil platform string: `Windows-11-10.0.26200-SP0` (`.\venv\Scripts\python.exe -c "import psutil, platform; ..."`). The OS naming differs between APIs, so the numeric build `10.0.26200` is the safest verified value.
- `systeminfo` failed with `ERROR: Access denied`.

### CPU and memory

- CPU model from registry: `13th Gen Intel(R) Core(TM) i5-13400`; identifier `Intel64 Family 6 Model 191 Stepping 2`; registry clock value `~MHz=2496` (`Get-ItemProperty HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0`).
- CPU topology from psutil: 10 physical cores and 16 logical CPUs; current reported frequency 2500.0 MHz (`.\venv\Scripts\python.exe -c "import psutil; ..."`).
- Registry processor-key count and .NET processor count both returned 16 logical processors (`(Get-ChildItem HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor).Count`; `[Environment]::ProcessorCount`).
- RAM from psutil: 34,116,444,160 bytes total, approximately 31.77 GiB (`.\venv\Scripts\python.exe -c "import psutil; print(psutil.virtual_memory().total)"`).
- Swap/pagefile from psutil: 12,884,901,888 bytes, approximately 12.0 GiB (`.\venv\Scripts\python.exe -c "import psutil; print(psutil.swap_memory().total)"`).
- WMI/CIM CPU and physical-memory commands failed with `Access denied` (`Get-CimInstance Win32_Processor`; `Get-CimInstance Win32_PhysicalMemory`).

### GPU and CUDA

- GPU from `nvidia-smi`: NVIDIA GeForce RTX 3060, WDDM driver model, PCI bus ID `00000000:01:00.0`, 12,288 MiB total VRAM, 170 W power cap, no MIG support (`nvidia-smi`; `nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id --format=csv,noheader`).
- NVIDIA driver version: `591.86`; CUDA version reported by the NVIDIA driver: `13.1` (`nvidia-smi`).
- PyTorch CUDA status: `torch.cuda.is_available() == True`; PyTorch CUDA runtime `12.4`; cuDNN version `90100`; one CUDA device detected; device 0 is `NVIDIA GeForce RTX 3060` with 12,884,377,600 bytes total memory (`.\venv\Scripts\python.exe -c "import torch; ..."`).
- At the time of capture, `nvidia-smi` showed approximately 1105 MiB of GPU memory in use by desktop/GUI processes, not by training or Ollama inference (`nvidia-smi`).

### Ollama runtime state

- No Ollama model was loaded at the time of inspection: `ollama ps` returned only the header row (`ollama ps`).
- Installed Ollama model sizes and quantization are documented in `SOFTWARE_CONFIG.md`; model loading/offloading behavior at runtime was not directly observable because no model was active.

## Extracted from code

### BERTic fine-tuning hardware use

- BERTic training uses HuggingFace `TrainingArguments` with `fp16=torch.cuda.is_available()`. On the verified environment, this condition evaluates to `True`, so mixed precision is enabled when these scripts run on this machine (`finetune_bertic_binary.py:341`; `finetune_bertic_categories.py:315`; `finetune_bertic_subcategories.py:315`; `finetune_bertic_ternary.py:307`; `cv_bertic.py:328`).
- Binary/category/subcategory/CV training defaults use `per_device_train_batch_size=8`, `per_device_eval_batch_size=args.batch_size * 2` (16 by default), and `gradient_accumulation_steps=4` (`finetune_bertic_binary.py:166`, `finetune_bertic_binary.py:327-329`; `finetune_bertic_categories.py:146`, `finetune_bertic_categories.py:301-303`; `finetune_bertic_subcategories.py:146`, `finetune_bertic_subcategories.py:301-303`; `cv_bertic.py:413`, `cv_bertic.py:314-316`, `cv_bertic.py:420`).
- Ternary BERTic training defaults use `per_device_train_batch_size=16` and `per_device_eval_batch_size=32`; no gradient accumulation option is defined in that script (`finetune_bertic_ternary.py:186`, `finetune_bertic_ternary.py:295-296`).
- Custom weighted losses move class weights to the active tensor/device (`labels.device` or `outputs.logits.device`), so loss computation follows the model/device chosen by Trainer (`finetune_bertic_binary.py:353-356`; `finetune_bertic_categories.py:330-333`; `finetune_bertic_subcategories.py:330-333`; `finetune_bertic_ternary.py:317`; `cv_bertic.py:221-223`).
- No `dataloader_num_workers` or explicit CPU worker count is set in `TrainingArguments`; HuggingFace Trainer defaults apply.

### Local LLM inference hardware use

- Ollama inference is called over local HTTP at `http://localhost:11434`; the Python code does not specify GPU layers, CPU threads, memory limits, or offload settings. Hardware placement is delegated to the Ollama service (`src/llm_detector.py:16`, `src/llm_detector.py:61-62`, `src/llm_detector.py:98-116`).
- Single-model LLM runs are sequential over records and prompts in the non-ensemble scripts (`single_sentence_run.py:416-423`; `full_text_run.py:223-228`).
- Ensemble inference uses `ThreadPoolExecutor(max_workers=len(detectors))`, creating one concurrent future per configured detector/model for each record (`single_sentence_run_ensemble.py:149-152`; `full_text_run_ensemble.py:179-195`).
- Ensemble model sets are controlled by `MODEL_SUBSET=[]`, which means all tags in `models/models.json` are used unless overridden (`single_sentence_run_ensemble.py:31-32`, `single_sentence_run_ensemble.py:76-80`; `full_text_run_ensemble.py:34-35`, `full_text_run_ensemble.py:151-155`; `models/models.json:2-6`).

## Flagged hardware reproducibility gaps

- The repository does not record per-run hardware utilization, GPU memory usage, Ollama processor placement, CPU thread counts, or whether each Ollama model was fully GPU-resident, partially offloaded, or CPU-resident during the original experiments.
- `nvidia-smi` reports driver CUDA support `13.1`, while PyTorch reports CUDA runtime `12.4`; this is normal for PyTorch wheels but should be reported precisely to avoid ambiguity.
- WMI/CIM hardware queries were denied in this environment. CPU and RAM were verified through registry and psutil instead.
- The ensemble scripts can submit multiple Ollama requests concurrently, but the code does not limit concurrency by GPU memory. On a 12 GiB RTX 3060, concurrent loading of several 2.2-6.6 GB quantized models may cause runtime-dependent offloading or serialization inside Ollama.
- Storage capacity was not included because the direct PowerShell filesystem capacity check returned non-informative zero values in this sandboxed session.

## Hardware configuration

The experiments were verified on a Windows system with build `10.0.26200` and x64 architecture, using a 13th Gen Intel Core i5-13400 CPU with 10 physical cores and 16 logical processors. The machine had 34,116,444,160 bytes of system memory, corresponding to approximately 31.77 GiB of RAM, and a 12.0 GiB pagefile/swap allocation as reported by psutil. GPU acceleration was available through a single NVIDIA GeForce RTX 3060 with 12,288 MiB of VRAM on NVIDIA driver 591.86; `nvidia-smi` reported CUDA 13.1, while the installed PyTorch build reported CUDA runtime 12.4 and cuDNN 90100. The BERTic fine-tuning scripts enable FP16 automatically when `torch.cuda.is_available()` is true, which was the case in the verified environment. Ollama model placement and GPU offload were managed by the Ollama service rather than by repository code, and no Ollama model was loaded at the time `ollama ps` was inspected. Ensemble LLM experiments can issue one concurrent request per configured local model, so actual GPU memory pressure during ensemble runs may depend on Ollama's runtime scheduling and model residency.
