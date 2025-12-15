# Changelog

All notable changes to this project will be documented in this file.  
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-14
([commit ]())
###Added
- `src/models/green_xgboost_stack_model.py`: XGBoost-based Meta-learner for the Stacked Hybrid Model.
- `notebooks/creditcards_eda.ipynb`: exploratory data analyses (EDA) for the **Credit Card Fraud Detection** dataset.
- `notebooks/synthetic_data_eda.ipynb`: exploratory data analyses (EDA) for the **PaySim Synthetic Financial Fraud** dataset.
- `notebooks/experiment_eval.ipynb`: notebook to assess experimental outcomes.

### Changed
- `README.md`: updated to reflect the lastest architectural and pipeline changes.
- `config/data_config.yaml`: removed unnecesary parameters.
- `config/experiments_config.yaml`: removed unavailable parameters for the tracker.
- `config/model_config.yaml`:
  - Added `Hybrid_model` configuration.
  - Updated `snn_model` and `conventional_nn_model` configurations.
- `config/training_config.yaml`: added configuration for meta-learner `xgboost_params`.
- `requirements.txt`: reorganized depndencies by category.
- `setup.py`: updated to install `src` as a Python package.
- `src/XAI/xai_hybrid_model.py`: updated `batch_lime_explanations` & `explain_model_with_lime` to suppport the Stacked Hybrid Model.
- `src/data/preprocess.py`: extended the `FraudDataset` class to support the staked learning:
  - Added `get_all_data_tensor` to return the full dataset as a single PyTorch tensor.
  - Added `get_all_labels_numpy`: to return all labels as a NumPy array.
- `src/models/conventional_model.py`: added **GELU** activation support.
- `src/models/hybrid_model.py`: refactored and extended `HybridModel`:
  - Added support for configurable multi-layer SNN & MLP architectures.
  - Refactored `forward` logi by extracting  `extract_fused_features` to support feature extraction for XGBoost Meta-Learner.
- `src/pipeline/evaluation.py`: updated `run_evaluation`:
  - Added support for the Stacked Hybrid Model
  - Added warm-up handling
  - Improved robustness of FLOPs to GFLOPs conversion.
- `src/pipeline/green_ai.py`: improved XAI invocation logic.
- `src/pipeline/training.py`: added Stacked Meta-Learner training for the Hybrid model.
- `src/pipeline/xai.py`: added configurable parameters for `batch_lime_explanation`.
- `src/training/trainer.py`: added **F2-score metric** support.
- `src/utils/flops.py`: improved approximation logic in `calculate_flops_hybrid` & `calculate_flops_hybrid_ml`.
- `src/utils/metrics.py`: 
  - Added `f2_score` in `calculate_metrics`.
  - Simplified and remove unnecesaary logic from `evaluate_model`
- `src/utils/model_utils.py`:
  - Updated `get_model_use`.
  - Channged `get_best_model` to use `f2_score` as the primary selection metric.

### Breaking changes
- The **hybrid Model Architecture** has been refactored to support **stacked learning** with an XGBoost Meta-Learner. Existing checkpoints trained earlier hybrid implementations are **not backward-compatible**.
- Model selection logic now prioritized **F2-score** instead F1-score. This may change the "Best model" selected during evaluation, particulary on highly imbalanced datasets.
- FLOPs estimation logic for hybrid models has been revised. FLOPs values reported in earlier experiments should not be directly compared with results from v1.0.0

### Known Limitations
- FLOPs and energy consumption metrics rely on approximation and profilin heuristic rather than hardware-level measurement, and should be interpreted as relative indicators, not absolute values.
- XAI explantions using LIME are:
  - Local to individual instances 
  - Sentsitive to feature sclainfg and sampling randmoness
  - Not guaranteed  to reflect global model bahavior
- Spiking Neural Networks (SNN) components are evaluated on CPU based simulation, shich may not fully reflect performance characteristics on neuromorphic hardware.
- The framework is designed for research and benchmarking purposes and is not optimized for real-time or production deployment.
- Class imbalnce handling is dataset-specific; results may not generalize without recalibration when appliied to unseen financial environments

## [0.2.2] - 2025-10-09
([commit 1b5873c](https://github.com/danilot390/Green-Ai-Fraud-Detection/commit/1b5873cc3e57027499e3a3931ba80046c4d5eb01))
### Added
- New baseline model to compare with Green Pipeline in terms of performance and sustainability.
- `src/models/xbost_cnn_bilstm.py` with `XBoost_CNN_BiLSTM` hybrid model combining CNN, BiLSTM, and MLP architectures, with optional integration of XGBoost-derived embeddings for tabular data.
- `src/models/losses.py` with `FocalLoss` function for binary classification tasks.
- `batch_lime_explanations` in `src/XAI/xai_hybrid_model.py` to generate LIME explanations for multiple data points and return top key features.
- `Batch Norm` added to `src/models/conventional_model.py` for improved performance.
- `Quantization-Dinamic` functionality added, and improved `Quantization Aware Training` & `Pruning` pipelines in `src/training/compression.py`.
- `get_best_model` in `src/utils/model_utils.py` to save the best model based on F1 improvements or pruning/quantization tolerance.
- `ml_cnn_bl_model` flag added in `load_model` function (`src/utils/model_utils.py`).
- `find_best_threshold` function added and integrated into `evaluate_model` (`src/utils/metrics.py`).
- `estimate_lstm_flops`, `calculate_flops_hybrid_ml` added and `calculate_flops_hybrid` improved in `src/utils/flops.py`.

### Changed
- Reordered and updated `config/experiments_config.yaml` with `dataset_name` & `xai_cases`.
- Updated `config/model_config.yaml` for conventional model and added `xgb_cnn_bilstm_model`.
- Updated `Loss config` in `config/training_config.yaml`.
- Reordered parameters in `config/xai_config.yaml`.
- `src/XAI/xai_hybrid_model.py` updated for batch LIME explanations.
- `src/pipeline/evaluation.py` updated to measure latency and FLOPs for `XBoost_CNN_BiLSTM` and split metrics into performance and sustainability.
- `src/pipeline/green_ai.py` adapted to function changes.
- `src/pipeline/model.py` updated to support configurable loss function.
- `src/pipeline/tracking.py` updated emissions logging.
- `src/pipeline/training.py` improved quantization in the training pipeline.
- `src/pipeline/xai.py` refactored for general XAI analyses and saving specific analyses by configuration.
- `src/training/trainer.py` updated with optimized scheduler, quantization functionalities, and early stopping in `trainer.run`.
- `src/utils/plotting.py` updated to stop showing graphs as they are saved automatically.
- `src/utils/reproducibility.py` updated logging handling.
### Fixed
*(No bug fixes reported in this version.)*

### Removed
- Deleted unnecessary `src/utils/.gitkeep` file.

### Dependencies
- Added `xgboost` library in `requirements.txt`.

## [0.1.1] - 2025-09-13
([commit b0c8252](https://github.com/danilot390/Green-Ai-Fraud-Detection/commit/b0c82528bb3cec3503a3d1458fcb9c4540e1eced))
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
