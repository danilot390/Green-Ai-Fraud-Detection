# GREEN-AI FRAUD DETECTION
This project implements a novel hybrid Deep Learning Pipeline to combat financial fraud. The framework combines 
* **Spiking Neural Networks (SNNs)** for temporal feature extraction
* **Conventional Neural Networks (CNNs)** for static feature extraction
* **Ensemble and Meta-Learning approaches**, including **XGBoost-based Meta-Learner**
* **Explainable AI (XAI)** techniques to support interpretability and regulatory alignment.

The proposed approach aims to **reduce computational cost and energy consumption** while maintainig competitive predictive performance on highly imbalanced financial fraud datasets. The framework supports systematic evaluation of **accutracy, efficiency, eplainabliity trade-offs** across multiple model families.

## Key Features
* Modular, configuration-driven pipeline.
* Conventional, spiking, hybrid and ensemble architectures.
* Energy and sustainability metrics (letency, FLOPs, model size, emissions).
* Model compression (Pruning, Quantization-Aware Training).
* Post-hoc explainability with LIME.
* Rreproducible experiments and structured logging.

## Requiremnts
* Python 3.10+ 
* Required libraires listed in `requirements.txt`.

it is **higly recommended to use a virtual enviornment:**
```python
# Create a virtual enviornment 
python -m venv .venv

#Activate the virtual enviornment
# Windows
.venv\Scripts\activate
# macOs/Linux
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

## Dataset
This project uses **2 publicly avaible datasets for fraud detection**. The datasets should be download from Kaggle and placed in `data/raw/ directory`.
1. **Credit Card Fraud Detection:** A higly imbalanced dataset of transactions made by European cardholders.
    **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
    **File:** `creditcard.csv`
2. **Synthetic Financial Dataset:** A simulated dataset of mobile money transactions
    **Source:** [Kaggle - Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1/data)
    **File:** `PS_20174392719_1491204439457_log.csv`

Expected directory structure:
```bash
project_root/
└── data/
    └── raw/
        ├── creditcard.csv
        └── PS_20174392719_1491204439457_log.csv

```
**Note:** If filenames differ, they can be adjusted in `config/data_config.yaml`.

## Project Structure
The project is organized into **modular and scalable architecture** for clarity, reproducibility, and maintainability:
```bash
project_root/
├── config/                       # Configuration files
├── data/
│   ├── raw/                      # Original datasets
│   └── processed/                # Cleaned and preprocessed datasets
├── notebooks/                    # Dataset's EDA & Result Analizes
├── models/
│   └── checkpoints/              # Model checkpoints 
├── src/
│   ├── pipeline/                 # Main pipeline 
│   ├── data/                     # Data preprocessing 
│   ├── models/                   # Model architectures
│   ├── ensembles/                # Ensemble and stacking models
│   ├── training/                 # Training logic
│   ├── xai/                      # Explainable AI modules
│   └── utils/                    # Shared utility functions
├── experiments/                  # Logs and experiments outputs
├── CHANGELOG.md                  # Project changelog
├── CONTRIBUTORS.md               
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py                      
└── LICENSE
```

## Running the pipeline
The pipeline runs in **two steps**:

### Step 1: Process the Datasets
Clean the raw data and convert it into tensors for model training. Processed outputs are saved in `data/processed/`:
```python
python -m src.data.make_dataset
```

### Step 2: Execute Green AI Pipeline
Run training, compression, evaluation, and XAI analysis:
```python
python -m src.pipeline.green_ai
```
**Configuration options**
* Model type: 
    - Conventional (Multi-Layer Perceptron), 
    - Spiking Neural Network - SNN, 
    - Hybrid Green AI model, 
    - XGBoost-CNN-BiLSTM,  
    - XGBoost-Hybrid & XGBoost-CNN_BiLSTM-Transformer (Ilebery & Sun) ensembles.
* Compression techniques: Pruning, QAT.
* Training and evaluation parameters.
* Experiments parameteres.
All settings are controlled through configuration files.

## Pipeline Output and Results
### Logs:
The pipeline generates logs including:
    * Training progress
    * Evaluation metrics
    * Latency and energy consumption
Example evaluation output:
``` yaml
---- Test Set Metrics ----
accuracy: 0.9993
precision: 0.8594
recall: 0.7432
f1_score: 0.7971
f2_score: 0.7639
auc_roc: 0.9646
pr_auc: 0.8035
Confusion Matrix: TP=55.0000, FP=9.0000, FN=19.0000, TN=42630.0000

---- Green Model Evaluation ----
latency_ms: 1.6627
flops_gflops: 0.0007
size_model: 0.0214
emissions_kg_co2e: 0.0001
```
### Explainable AI (LIME)
For Hybrid model, the pipeline generates ***local explanations**.
Example XAI output:
```yaml
---- Explainable AI with LIME started ----
Number of cases selected: 25 (per class distribution)
Total features considered for analysis: 10.
XAI Analysis: Across all 50 fraudulent cases, V4 <= -0.61 (appearing: 10), V18 <= -0.59 (appearing: 6), V5 <= -0.50 (appearing: 5) consistently had the strongest influence on model predictions (both positive and negative, depending on the case).
The `XAI_per_points.json` is saved in `experiments/run_20251214_200118/xai`.
The `xai_config.json` is saved in `experiments/run_20251214_200118/xai`.
The `top_key_features.json` is saved in `experiments/run_20251214_200118/xai`.
The `hyperparams.json` is saved in `experiments/run_20251214_200118`.
```
### Visualizations
Optional via configuration generated visual artifacts include:
    * Confussion Matrix
    * ROC Curve

## Model Architectures
The project supports four architectures:
### 1. Conventional Neural Network (CNN) - Multi-Layer Perceptron (MLP)
Traditional continous-value neural network trained with backpropagation.
### 2. Spiking Neural Network (SNN)
Processes information with discrete, time-dependent spikes, offering **energy efficiency** and fast computation.
### 3. Hybrid Model
Uses an SNN as a **temporal feature extractor**, a CNN as the **static feature extractor**, and a Meta-learnening components.
### 4. XGBoost-CNN-BiLSTM-Transformer ensemble (Ilebery & Sun, 2024)
The Ileberi & Sun (2024) model is implemented for comparative and benchmarking purposes only, using static hyperparameters as described in the original paper. It serves as a reference ensemble architecture to evaluate the proposed Green AI Hybrid Model in terms - Predictive performance, Computational efficiency, Energy and emissions trade-offs, Explainability behavior.
### 5. XGBoost CNN BiLSTM model
Ensemble architecture combining tree-based learning and deep sequence modeling.

### Support Artifacts
* Hyperparamenters: `hyperparams.json`
* Energy and emissions logs: `emissions.csv` & `powermetrics_log.txt`
* XAI expplanations: `top_key_features.json`, `xai_config.json`, `XAI_per_points.json`

## Explainability and Responsible AI
The framework integrates **LIME (Local Interpretable Model-Agnostic Explanations)** for Hybrid Green AI to:
* Explain individual predictions.
* Identify feature contributions.
* Improve transparecny and trust in fraud detections systems.

## Project Status
This repository represents a **research and experimental framework**, suitable fro:
* Academic evaluation
* Green AI benchmarking
* Responsible AI and explainability studies

## Citation
If you use or reference the Ileberi & Sun (2024) ensemble implementation, please cite
```bibtex
@article{Ileberi&Sun2024,
  author    = {Emmanuel Ileberi and Yanxia Sun},
  title     = {A Hybrid Deep Learning Ensemble for Credit Card Fraud Detection},
  journal   = {IEEE Access},
  volume    = {12},
  pages     = {1--14},
  year      = {2024},
  publisher = {IEEE},
  doi       = {10.1109/ACCESS.2024.3502542}
}
```

## Author
**Name**: Danilo Angel Tito Rodriguez
**Role**: AI Research Engineering | Green AI & Explainable Machine Learning
**LinkedIn**: www.linkedin.com/in/danilo-tito-7313931a7