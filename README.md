# GREEN-AI FRAUD DETECTION
This project implements a novel hybrid Deep Learning Pipeline to combat financial fraud. It combines Spiking Neural Networks (SNNs) for utlra-efficient initial pattern detection with Conventional Neural Networks (CNNs) and integrates Explainable AI (XAI) techniques. This approach reduces energy consumption while maintaining high accuracy and providing interpretable results.

The pipeline evaluates differetn model architectures -including conventional, spiking, and hybrid neural networks- while measuring energy efficiency and providing insights into model decisions. It covers data preprocessing, model training with compression techniques, evaluation, and XAI analysis.

## Requiremnts
It requires Python 3.10+  and the required libraires listed in requirements.txt.

it is **higly recommended to use a virtual enviornment:**
```python
# Create a virtual enviornment 
python -m venv .venv

#Activate the virtual enviornment
# on Windows
.venv\Scripts\activate
#on macOs/Linux
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

## Dataset
This project uses 2 publicly avaible datasets for fraud detection. Both datasets should be download from Kaggle and placed into the data/raw/ directory.
1. **Credit Card Fraud Detection:** A higly unbalanced dataset of transactions made by European cardholders.
    **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/ghnshymsaini/credit-card-fraud-detection-dataset)
2. **Synthetic Financial Dataset:** A simulated dataset of mobile money transactions
    **Source:** [Kaggle - Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1/data)

After downloading, the directory should look like:
```bash
project_root/
└── data/
    └── raw/
        ├── creditcard.csv
        └── PS_20174392719_1491204439457_log.csv

```
**Note:** If the datasets are save under different names, can be fixed in config/data_config.yaml.

## Project Structure
The project is organized into **modular components** for clarity and maintainability:
```bash
project_root/
├── data/
│   ├── raw/                      # Original datasets
│   └── processed/                # Cleaned and preprocessed datasets
├── models/
│   └── checkpoints/              # Model checkpoints during training
├── src/
│   ├── pipeline/                 # Main pipeline script
│   ├── data/                     # Data preprocessing scripts
│   ├── models/                   # Model definitions and architectures
│   ├── training/                 # Model training scripts
│   ├── xai/                      # Explainable AI modules
│   └── utils/                    # Utility functions
├── config/                       # Several configuration files
├── experiments/                  # Logs and results from experiments
├── CHANGELOG.md                  # Project changelog
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py                       # Optional: for installing src as a package
└── LICENSE
```

## Run the pipeline
The pipeline runs in **two steps**:

### Step 1: Process the Datasets
Clean the raw data and convert it into tensors for model training. Processed files are saved in data/processed/:
```
python -m src.data.make_dataset
```

### Step 2: Execute the Main Pipeline
Run main **Green AI Pipeline**, which handles training, compression, evaluation, and XAI analysis:
```bash
python -m src.pipeline.green_ai
```
**Note:** Experiments can be customized through config files, including **model type and architecture** (Conventional, Spiking, or Hybrid), and **compression tecniques** (Pruning and QAT).

## Pipeline Output and Results
### Detailed logs:
The pipeline generates logs containing key model insights, such as:
    * Training progress
    * Evaluation metrics
    * Latency and energy consumption
Example evaluation output:
``` yaml
2025-09-12 23:10:43,497 | INFO     | Test Set Mertrics:
2025-09-12 23:10:43,497 | INFO     | accuracy: 0.9790
2025-09-12 23:10:43,497 | INFO     | precision: 0.3995
2025-09-12 23:10:43,497 | INFO     | recall: 0.9984
2025-09-12 23:10:43,497 | INFO     | f1_score: 0.5706
...
2025-09-12 23:10:43,497 | INFO     | latency_ms: 0.5262
2025-09-12 23:10:43,497 | INFO     | flops_gflops: 0.0000
2025-09-12 23:10:43,497 | INFO     | size_model: 0.0108
```
It also includes XAI explanations with LIME:
```yaml
2025-09-13 18:25:51,094 | INFO     | --- Explainable AI with LIME started ---
2025-09-13 18:25:51,094 | INFO     | Number of cases to analyze: 3
2025-09-13 18:25:51,094 | INFO     | Number of features to include in XAI: 10
2025-09-13 18:25:51,094 | INFO     | Case: 1
2025-09-13 18:25:51,095 | INFO     | Found a fraudulent transaction at index 28878. Explaining this instance.
2025-09-13 18:25:51,347 | INFO     | Predicted class for the instance: `Fraud`
2025-09-13 18:25:51,348 | INFO     | LIME Explanation for the selected instance
2025-09-13 18:25:51,348 | INFO     | Original prediction: `Fraud` with probability `1.0`
2025-09-13 18:25:51,348 | INFO     | Top features contributing to the prediction:
2025-09-13 18:25:51,348 | INFO     |   - V25 > 0.69: 0.0001
2025-09-13 18:25:51,348 | INFO     |   - V5 <= -0.50: 0.0001
2025-09-13 18:25:51,348 | INFO     |   - V2 > 0.49: 0.0001
2025-09-13 18:25:51,348 | INFO     |   - V23 <= -0.26: 0.0001
2025-09-13 18:25:51,348 | INFO     |   - V14 <= -0.44: 0.0001
2025-09-13 18:25:51,348 | INFO     |   - -0.59 < V24 <= 0.07: 0.0001
2025-09-13 18:25:51,348 | INFO     |   - V26 > 0.50: 0.0001
2025-09-13 18:25:51,348 | INFO     |   - 0.01 < V22 <= 0.73: 0.0001
2025-09-13 18:25:51,348 | INFO     |   - V6 <= -0.58: 0.0001
2025-09-13 18:25:51,348 | INFO     |   - V17 <= -0.58: 0.0001
...
```
### Plots
The pipeline generates visualizations, including:
    * Confussion Matrix
    * ROC Crve

### Support Files
Additional artifacts include:
    * Hyperparamenters in `hyperparams.json`
    * Energy evaluation in `emissions.csv` & `powermetrics_log.txt`
    * Results of the XAI in HTML format

## Model Architectures
The project supports three architectures:
### 1. Conventional Neural Network (CNN)
Traditional continous-value neural network trained with backpropagation.
### 2. Spiking Neural Network (SNN)
Processes information with discrete, time-dependent spikes, offering **energy efficiency** and fast computation.
### 3. Hybrid Model
Uses an SNN as a **feature extractor** and a CNN as the **decision layer**.

## Explainable AI (XAI) with LIME
The project uses **LIME (Local Interpretable Model-Agnostic Explanations)** to explain individual predictions:
    * `LimeTabularExplainer` generates a simple interpretable model around a single prediction.
    * Highlights the most important features contributing to a fraud decision.
    * Increase transparency and trust in model predictions.

