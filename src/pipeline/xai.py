import torch
import numpy as np
import os
import sys
from lime.lime_tabular import LimeTabularExplainer

from src.XAI.xai_hybrid_model import explain_model_with_lime
from src.utils.config_parser import load_config

def run_xai(model, dataset_name, xai_cases, experiment_dir, logger, device):
    """
    Runs LIME (Local Interpretable Model-agnostic Explanations) on a model to explain test instances.

    This function explains a specified number of test cases, prioritizing fraudulent transactions if present,
    and saves the explanations as HTML files.
    """
    # Load configuration files
    xai_config = load_config('config/xai_config.yaml')

    # Load the raw dataset to get an instance to explain
    processed_dir = os.path.join('data/processed', dataset_name)
    X_test = torch.load(os.path.join(processed_dir, 'X_test.pt'))
    y_test = torch.load(os.path.join(processed_dir, 'y_test.pt'))

    # Use the raw tabular data for the LIME explainer.
    X_test_tabular = X_test.numpy()
    y_test_labels = y_test.flatten().numpy().astype(int) if y_test.ndim > 1 else y_test.numpy().astype(int)

    # Create the LIME explainer instance 
    feature_names = np.loadtxt(os.path.join(processed_dir, 'feature_names.txt'), dtype=str)
    class_names = ['Non-Fraud', 'Fraud']
    
    # Using the whole test set for fitting is a good practice.
    explainer = LimeTabularExplainer(
        training_data=X_test_tabular,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        sample_around_instance = xai_config['xai_methods']['lime'].get('sample_around_instance', False),
        kernel_width=xai_config['xai_methods']['lime']['kernel_width'],
    )
    logger.info('--- Explainable AI with LIME started ---')
    logger.info(f'Number of cases to analyze: {xai_cases}')
    logger.info(f'Number of features to include in XAI: {xai_config['xai_methods']['lime'].get('num_features',5)}')
    
    fraud_indices = np.where(y_test_labels == 1)[0]
    has_fraud = len(fraud_indices) > 0
    num_test_instances = len(X_test_tabular)
    xai_paths = os.path.join(experiment_dir, 'xai')
    os.makedirs(xai_paths, exist_ok=True)

    for i in range(xai_cases):    
        logger.info(f'Case: {i + 1}')
        if has_fraud:
            rand_instance_idx = np.random.choice(fraud_indices)
            logger.info(f"Found a fraudulent transaction at index {rand_instance_idx}. Explaining this instance.")
        else:
            rand_instance_idx = np.random.randint(0, num_test_instances)
            logger.info(f"No fraudulent instances found in the test set. Explaining a random instance: {rand_instance_idx}")

        instance_to_explain = X_test_tabular[rand_instance_idx]
        path = os.path.join(xai_paths,f'{i+1}_i{rand_instance_idx}_lime.html')
        explain_model_with_lime(
            model=model,
            explainer=explainer,
            data_point=instance_to_explain,
            class_names=class_names,
            config=xai_config['xai_methods']['lime'],
            path_dir=path,
            device=device,
            logger=logger
        )