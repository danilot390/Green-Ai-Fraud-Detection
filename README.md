# GREEN-AI FRAUD DETECTION
This project implements a novel hybrid Deep Learning Pipeline to combat financial fraud. It combines **Spiking Neural Networks (SNNs)** for temporal feature extraction, **Conventional Neural Networks (CNNs)** for static feature extraction, and **XGBoost-based Meta-Learner**, and integrates **Explainable AI (XAI)** techniques to support interpretability and regulatory alignment.

The proposed approach aims to **reduce computational cost and energy consumption** while maintainig competitive predictive performance on highly imbalanced financial fraud datasets. The framework supports systematic evaluation of **accutracy, efficiency, eplainabliity trade-offs** across multiple model families.

## Key Features
* Modular, configuration-driven pipeline.
* Conventional, spiking, and hybrid neural architectures.
* Energy and sustainability metrics (letency, FLOPs, model size, emissions).
* Model compression (Pruning, Quantization-Aware Training).
* Post-hoc explainability with LIME.
* Rreproducible experiments and structured logging.

## Requiremnts
It requires Python 3.10+  and the required libraires listed in `requirements.txt`.

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
    **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/ghnshymsaini/credit-card-fraud-detection-dataset)
    (creditcard.csv)
2. **Synthetic Financial Dataset:** A simulated dataset of mobile money transactions
    **Source:** [Kaggle - Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1/data)
    (PS_20174392719_1491204439457_log.csv)

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
The project is organized into **modular and scalable architecture** for clarity and maintainability:
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
│   ├── training/                 # Training 
│   ├── xai/                      # Explainable AI modules
│   └── utils/                    # Utility functions
├── experiments/                  # Logs and experiments outputs
├── CHANGELOG.md                  # Project changelog
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py                      # Package installation
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
**Configuration options include**
* Model type: Conventional (Multi-Layer Perceptron), SNN, Hybrid, XGBoost-CNN-BiLSTM.
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
-...TIMESTAMP...-| INFO    | ---- Test Set Metrics ----
-...TIMESTAMP...-| INFO    | accuracy: 0.9993
-...TIMESTAMP...-| INFO    | precision: 0.8594
-...TIMESTAMP...-| INFO    | recall: 0.7432
-...TIMESTAMP...-| INFO    | f1_score: 0.7971
-...TIMESTAMP...-| INFO    | f2_score: 0.7639
-...TIMESTAMP...-| INFO    | auc_roc: 0.9646
-...TIMESTAMP...-| INFO    | pr_auc: 0.8035
-...TIMESTAMP...-| INFO    | Confusion Matrix: TP=55.0000, FP=9.0000, FN=19.0000, TN=42630.0000
-...TIMESTAMP...-| INFO    | ---- Green Model Evaluation ----
-...TIMESTAMP...-| INFO    | latency_ms: 1.6627
-...TIMESTAMP...-| INFO    | flops_gflops: 0.0007
-...TIMESTAMP...-| INFO    | size_model: 0.0214
-...TIMESTAMP...-| INFO    | emissions_kg_co2e: 0.0001
```
### Explainable AI (LIME)
For Stacked Hybrid model, the pipeline generates ***local explanations**.
Example XAI output:
```yaml
-...TIMESTAMP...-| INFO     | ---- Explainable AI with LIME started ----
-...TIMESTAMP...-| INFO     | Number of cases selected: 25 (per class distribution)
-...TIMESTAMP...-| INFO     | Total features considered for analysis: 10.
-...TIMESTAMP...-| INFO     | XAI Analysis: Across all 50 fraudulent cases, V4 <= -0.61 (appearing: 10), V18 <= -0.59 (appearing: 6), V5 <= -0.50 (appearing: 5) consistently had the strongest influence on model predictions (both positive and negative, depending on the case).
-...TIMESTAMP...-| INFO     | The `XAI_per_points.json` is saved in `experiments/run_20251214_200118/xai`.
-...TIMESTAMP...-| INFO     | The `xai_config.json` is saved in `experiments/run_20251214_200118/xai`.
-...TIMESTAMP...-| INFO     | The `top_key_features.json` is saved in `experiments/run_20251214_200118/xai`.
-...TIMESTAMP...-| INFO     | The `hyperparams.json` is saved in `experiments/run_20251214_200118`.
```
### Visualizations
The pipeline generates visualizations, including:
    * Confussion Matrix
    * ROC Crve

### Support Artifacts
* Hyperparamenters: `hyperparams.json`
* Energy and emissions logs: `emissions.csv` & `powermetrics_log.txt`
* XAI expplanations: `top_key_features.json`, `xai_config.json`, `XAI_per_points.json`

## Model Architectures
The project supports four architectures:
### 1. Conventional Neural Network (CNN) - Multi-Layer Perceptron (MLP)
Traditional continous-value neural network trained with backpropagation.
### 2. Spiking Neural Network (SNN)
Processes information with discrete, time-dependent spikes, offering **energy efficiency** and fast computation.
### 3. Hybrid Model
Uses an SNN as a **temporal feature extractor**, a CNN as the **static feature extractor**, and a Meta-learnening components.
### 4. XBoost CNN BiLSTM model
Ensemble architecture combining tree-based learning and deep sequence modeling.

## Explainable AI (XAI) with LIME
The framework integrates **LIME (Local Interpretable Model-Agnostic Explanations)** to:
* Explain individual predictions.
* Identify feature contributions.
* Improve transparecny and trust in fraud detections systems.

## Project Status
This repository represents a **research and experimental framework**, suitable fro:
* Academic evaluation
* Green AI benchmarking
* Responsible AI and explainability studies

## Author
**Name**: Danilo Angel Tito Rodriguez
**Role**: AI Research Engineering | Green AI & Explainable Machine Learning
**LinkedIn**: www.linkedin.com/in/danilo-tito-7313931a7