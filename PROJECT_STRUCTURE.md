# Test8 Project Structure and File Roles

This document organizes the repository by folder and purpose.

## 1) Source Modules

- `src/core/configs.py`: Hyperparameters and runtime/model configuration.
- `src/core/paths.py`: Centralized project-relative path definitions.
- `src/core/models.py`: Core defense modules (CIDT, SAE projection, intervention, generation pipeline).
- `src/core/data_utils.py`: Multi-turn dataset and dataloader utilities.
- `src/core/metrics.py`: Safety and utility metrics.
- `src/core/attacks.py`: FITD attacker prompt construction.
- `src/core/benchmark_loader.py`: Local-first benchmark dataset loader/downloader and normalization.

## 2) Runnable Scripts

### Evaluation
- `scripts/eval/evaluate_safety.py`: Safety evaluation (ASR/HRI).
- `scripts/eval/evaluate_utility.py`: Utility evaluation (MMLU slice).
- `scripts/eval/run_full_benchmarks.py`: Unified safety + utility benchmark entrypoint.

### Training / Preprocessing
- `scripts/train/train_cidt.py`: Train CIDT router.
- `scripts/train/build_mixed_data.py`: Build `mixed_train_data.json`.
- `scripts/train/preprocess_fitd_from_github.py`: Parse FITD logs to `fitd_train_data.json`.
- `scripts/train/extract_benign_activations.py`: Build benign activation matrix `Hb`.
- `scripts/train/sae_feature_selection.py`: Select refusal-related SAE features.
- `scripts/train/download_sae_data.py`: Download/build SAE benign/malicious prompt data.

### Demo / Tools
- `scripts/demo/main.py`: End-to-end demo run.
- `scripts/tools/test.py`: Inspect SAE safetensors checkpoints.

## 3) Data Directory (`data/`)

### Safety Datasets
- `data/advbench.csv`: AdvBench single-turn harmful prompts.
- `data/harmful_behaviors.csv`: Original harmful behaviors CSV source.
- `data/jailbreakbench.csv`: JailbreakBench prompts.
- `data/maliciousinstruct.csv`: MaliciousInstruct prompts.
- `data/multibreak.jsonl`: MultiBreak-style multi-turn harmful dialogs.
- `data/crescendo.json`: Crescendo-style multi-turn harmful dialogs.

### Utility Datasets
- `data/mmlu/`: Local MMLU subject folders.
- `data/gsm8k_test.jsonl`: GSM8K test set.
- `data/mt_bench/question.jsonl`: MT-Bench prompts.
- `data/humaneval.jsonl`: HumanEval tasks.

### SAE Data
- `data/sae_benign_prompts.json`: Benign prompts for SAE analysis.
- `data/sae_malicious_prompts.json`: Harmful prompts for SAE analysis.
- `data/verified_refusal_features.pt`: Saved selected SAE refusal features.

## 4) Artifacts (`artifacts/`)

### Checkpoints
- `artifacts/checkpoints/cidt_final.pt`: Trained CIDT checkpoint.
- `artifacts/checkpoints/Hb_benign_layer15.pt`: Benign hidden-activation matrix.
- `artifacts/checkpoints/verified_refusal_features.pt`: Refusal feature artifact.

### Reports
- `artifacts/reports/full_benchmark_report.json`: Defended benchmark report.
- `artifacts/reports/full_benchmark_report_baseline.json`: Baseline benchmark report.

### Logs
- `artifacts/logs/test_log.log`: Demo/runtime logs.

Root-level compatibility symlinks are kept for old paths with same file names.

## 5) External References

- `_external/crescendoattacker/`: Local external source repository used to materialize Crescendo data.

## 6) Path Convention (Updated)

All scripts now read paths from `src/core/paths.py` (instead of project-absolute hardcoded strings), so the project can be moved or copied without rewriting code.

- If you move the repository, only `configs.py` environment variables may need adjustment:
  - `MODEL_PATH`
  - `SAE_DIR`
  - optional `FITD_LOG_ROOT` (for `preprocess_fitd_from_github.py`)

## 7) How To Run (New Paths)

Run commands from project root:

- `python scripts/demo/main.py`
- `python scripts/train/train_cidt.py`
- `python scripts/eval/evaluate_safety.py`
- `python scripts/eval/evaluate_utility.py`
- `python scripts/eval/run_full_benchmarks.py --quick`
