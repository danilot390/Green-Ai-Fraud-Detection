# Changelog

## [Unreleased] - 2025-09-12
### Added
- Applied pruning and quantization-aware training (QAT) in the training pipeline.
- Modularized main pipeline into dedicated moduels: model, training, evaluation, tracking, and setup.
- Refactored utilities for clarity and consistency.

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
- `src/utils/reproducibility.py`: fixed a small error in `set_seed`.

### Deleted
- `green_ai.py`: moved into `src/pipeline`.

### New
- `src/pipeline/green_ai.py`: runs the end-to-end Green AI fraud detection pipeline:
  * Load configs, set up logging, device, and reproducibility.
  * Initialize dataloaders and model (SNN, CNN, or hybrid).
  * Optionally apply compression (quantization, pruning, or both).
  * Define loss/optimizer and train the model.
  * Optionally convert and save a quantized model.
  * Evaluate on test set (latency, FLOPs, metrics).
  * Save results, plots, and experiment metadata.
- `src/pipeline/__init__.py`: treat pipeline as a package.
- `src/pipeline/evaluation.py`: evaluate model, measuring latency, FLOPs, size, metrics (accuracy, precision, recall, F1, ROC-AUC, average precision, confusion matrix), and generate plots.
- `src/pipeline/model.py`:
  * `setup_model`: load model and optionally apply compression (QAT/pruning) based on config.
  * `finalize_model`: convert to quantized model if required.
- `src/pipeline/setup.py`: `setup_experiment` to load configs, logger, and experiment directories.
- `src/pipeline/tracking.py`:
  * `start_tracker`: start energy consumption tracker.
  * `stop_tracker`: end energy consumption tracker.
- `src/pipeline/training.py`: `run_training` to run the training loop using the `Trainer` class.
- `src/data/dataloaders.py`: `get_dataloaders` function.
- `src/training/compression.py`: handle compression techniques:
  * `prepare_qat_model`: prepare model for QAT based on config.
  * `convert_qat_model`: convert trained QAT model to quantized model.
  * `apply_pruning`: apply pruning on model based on settings.
  * `remove_pruning`: remove pruning reparametrization to sparsify the model.
- `src/utils/model_utils.py`: common model tasks:
  * `get_model_use`: get active models based on config.
  * `load_model`: load model based on config.
  * `get_model_size`: save the model state dict to a temporary file and return its size in MB.