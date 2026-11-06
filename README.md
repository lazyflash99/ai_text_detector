# AI-Generated Text Detection Across Domains: Robustness Under Paraphrase and Domain Shift

A robust, crash-resilient, open-source benchmarking pipeline designed for AI-generated text detection under paraphrase attacks and domain shifts (e.g., RAID and M4 benchmarks).

## Target Constraints
- Tailored for Google Colab Free T4 (16 GB VRAM, ~12 GB RAM, ~80 GB disk) and Kaggle (P100/T4).
- Resumable pipelines with atomic chunked persistence to Google Drive or local storage.
- Adaptive OOM guard with automatic batch halving and CPU fallback.
- Scale tiers: `smoke` (test run <5 min), `small` (~5k samples), `medium` (~20k samples).

## Directory Layout
- `configs/`: YAML configuration files defining experiment tiers, models, and limits.
- `src/`: Modular Python library containing utilities, data loaders, detectors, attack harnesses, and evaluation logic.
- `notebooks/`: Ready-to-run Jupyter/Colab notebooks.
- `outputs/`: Default local output directory for checkpoints, metrics, and generated texts.
- `tests/`: Automated unit tests and crash/resume simulation suites.

## Quickstart (Colab / Local)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run smoke tests:
   ```bash
   python -m unittest tests/test_utils.py
   ```
