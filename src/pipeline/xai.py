import torch
import numpy as np
import os
import sys
from lime.lime_tabular import LimeTabularExplainer

from src.XAI.xai_hybrid_model import batch_lime_explanations
from src.utils.config_parser import load_config
from src.utils.common import save_to_json

def run_xai(model, dataset_name, xai_cases,time_steps, experiment_dir, logger, device):
    """
    Runs LIME on a model to explain test instances.
    """
    # Load configuration files
    xai_config = load_config('config/xai_config.yaml')

    # Load the raw dataset to get an instance to explain
    processed_dir = os.path.join('data/processed', dataset_name)
    X_test = torch.load(os.path.join(processed_dir, 'X_test.pt'))
    y_test = torch.load(os.path.join(processed_dir, 'y_test.pt'))

    # Use LIME explainer.
    X_test_tabular = X_test.numpy()
    y_test_labels = y_test.flatten().numpy().astype(int) if y_test.ndim > 1 else y_test.numpy().astype(int)

    # Create the LIME explainer  
    feature_names = np.loadtxt(os.path.join(processed_dir, 'feature_names.txt'), dtype=str)
    class_names = ['Non-Fraud', 'Fraud']
    
    explainer = LimeTabularExplainer(
        training_data=X_test_tabular,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        sample_around_instance = xai_config['lime'].get('sample_around_instance', False),
        kernel_width=xai_config['lime'].get('kernel_width', 0.75),
    )
    logger.info('---- Explainable AI with LIME started ----')
    logger.info(f'Number of cases selected: {xai_cases} (per class distribution)')
    logger.info(f'Total features considered for analysis: {xai_config['xai'].get('num_features',5)}')
    
    fraud_indices = np.where(y_test_labels == 1)[0]
    non_fraud_indices = np.where(y_test_labels == 0)[0]
    has_fraud = len(fraud_indices) > 0
    num_test_instances = len(X_test_tabular)

    if has_fraud:
        rand_f_instances_idx = np.random.choice(fraud_indices, size=xai_cases, replace=True)
        rand_nf_instances_idx =np.random.choice(non_fraud_indices, size=xai_cases, replace=True)
        rand_instances_idx = np.concatenate([rand_f_instances_idx, rand_nf_instances_idx])
    else:
        rand_instances_idx = np.random.randint(0, num_test_instances, size=xai_cases)
    
    instances_to_explain = X_test_tabular[rand_instances_idx]

    xai_results = batch_lime_explanations(
        model, 
        explainer, 
        instances_to_explain, 
        xai_config,
        time_steps,
        device, 
        logger)
    
    xai_paths = os.path.join(experiment_dir, 'xai')
    os.makedirs(xai_paths, exist_ok=True)
    common_fraud_features = ', '.join(
            f'{feature} (appearing: {count})'
            for feature, count in xai_results['global_top_fraud_features']
            )
    if xai_results['global_top_nonfrauds_features'] != xai_results['global_top_fraud_features']:
        common_nonfraud_features = ', '.join(
            f'{feature} (appearing: {count})'
            for feature, count in xai_results['global_top_nonfrauds_features']
            )
        xai_conclusion =( f'XAI Analysis: Across all {xai_cases*2} fraudulent cases, {common_fraud_features} were consistently the strongest positive indicators,'
                        f' while {common_nonfraud_features} influenced predictions negatively.')    
    else:
        xai_conclusion =(
            f'XAI Analysis: Across all {xai_cases*2} fraudulent cases, {common_fraud_features} consistently had the strongest influence on model'
            f' predictions (both positive and negative, depending on the case)'
        )
    logger.info(xai_conclusion)

    if xai_config['xai'].get('save_xai_cases', True):  
        xai_cases = {index: value for index, value in enumerate(xai_results.pop('per_point_explanations',[]))} 
        save_to_json(
            xai_cases, 
            xai_paths, 
            'XAI_per_points.json',
            logger)
    else:
        xai_results.pop('per_point_explanations')

    save_to_json(
        xai_config,
        xai_paths,
        'xai_config.json',
        logger
    )
    save_to_json(
        xai_results,
        xai_paths,
        'top_key_features.json',
        logger
    )
