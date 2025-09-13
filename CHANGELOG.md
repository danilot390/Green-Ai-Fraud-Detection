# Changelog

All notable changes to this project will be documented in this file.  
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-09-13
([compare]())
### Added
- Integrated XAI into the main pipeline with the LIME.
- `src/pipeline/xai.py`: 
  * Added `run_xai` function to generate LIME explanations for a specified number of test cases. Fraudulent transactions are prioritized if present. Explanations are saved as HTML files.

### Changed
- `README.md`: Updated with additional project information.
- `config/experiments_config.yaml`: Added the `xai_cases` feature.
- `src/XAI/xai_hybrid_model.py`: Updated `explain_model_with_lime` to handle logging and save results to specific files.
- `src/data/make_dataset.py`: Improved folder handling for dataset storage.
- `src/pipeline/evaluation.py`: Fixed FLOPs measurement for conventional models.
- `src/pipeline/green_ai.py`: Integrated XAI into the pipeline.
- `src/pipeline/tracking.py`: Improved handling of storage.
- `src/training/trainer.py`: Fixed prefixes for identifying compression techniques in models.


## [0.1.0] - 2025-09-12
([commit 97a7b1f2039d270639965ad5bc4c88806fb07007](https://github.com/danilot390/Green-Ai-Fraud-Detection/commit/97a7b1f2039d270639965ad5bc4c88806fb07007))

### Added
- Applied pruning and quantization-aware training (QAT) in the training pipeline.
- Modularized main pipeline into dedicated modules: model, training, evaluation, tracking, setup.
- Refactored utilities for clarity and consistency.
- `src/pipeline/green_ai.py`: runs the end-to-end Green AI fraud detection pipeline, including loading configs, initializing dataloaders and model, applying optional compression, training, saving quantized model, evaluation, and saving results/plots/metadata.
- `src/pipeline/__init__.py`: treat pipeline as a package.
- `src/pipeline/evaluation.py`: evaluates model, measuring latency, FLOPs, size, metrics (accuracy, precision, recall, F1, ROC-AUC, average precision, confusion matrix), and generates plots.
- `src/pipeline/model.py`: 
  * `setup_model`: load model and optionally apply compression (QAT/pruning) based on config.
  * `finalize_model`: convert to quantized model if required.
- `src/pipeline/setup.py`: `setup_experiment` to load configs, logger, and experiment directories.
- `src/pipeline/tracking.py`: 
  * `start_tracker`: start energy consumption tracker.
  * `stop_tracker`: end energy consumption tracker.
- `src/pipeline/training.py`: `run_training` to run the training loop using the `Trainer` class.
- `src/data/dataloaders.py`: `get_dataloaders` function.
- `src/training/compression.py`: handle compression techniques (`prepare_qat_model`, `convert_qat_model`, `apply_pruning`, `remove_pruning`).
- `src/utils/model_utils.py`: common model tasks (`get_model_use`, `load_model`, `get_model_size`).

### Modified
- `config/model_config.yaml`: updated performance parameters; moved compression params out.
- `config/training_config.yaml`: added compression (quantization & pruning) params.
- `src/models/snn_model.py`: fixed configuration handling for the second layer.
- `src/training/trainer.py`: updated `Trainer.run` to support pruning and QAT.
- `src/utils/common.py`: 
  * Added `save_to_json`.
  * Updated `get_device` to accept device.
  * Refactored `to_numpy_squeeze` → `to_int_array`.
- `src/utils/metrics.py`: 
  * Refactored `calculate_metrics`; added average precision and confusion matrix.
  * Added `evaluate_model` to run a model on a dataloader and return metrics.
  * Added `bench_mark_inference` to benchmark average inference time over a number of runs.
- `src/utils/reproducibility.py`: fixed small error in `set_seed`.

### Deleted
- `green_ai.py`: moved into `src/pipeline`.
