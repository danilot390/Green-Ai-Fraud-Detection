import time
import numpy as np
import torch
from ptflops import get_model_complexity_info

from src.utils.flops import calculate_flops
from src.utils.metrics import calculate_metrics
from src.utils.plotting import plotting
from src.utils.model_utils import get_model_size

def run_evaluation(model, test_loader, device, is_hybrid, plots_dir, model_config, logger):
    """
    Evaluate the model, meassuring latency, FLOPs, size, metrics (accuracy, precision, recall, F1, ROC-AUC, average precision, confusion matrix), and generating plots. 
    """
    logger.info('Evaluating model')

    model.eval()
    all_preds_proba, all_y_test = [], []

    # Measure inference latency
    logger.info('Measuring inference latency...')
    real_input, _ = next(iter(test_loader))
    real_input = real_input.to(device)

    start_time = time.time()
    with torch.no_grad():
        _ = model(real_input)
    latency_ms = (time.time() - start_time)* 1000
    logger.info(f'Inference Latency: {latency_ms:.2f} ms')

    # predictions
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            probabilities = torch.sigmoid(outputs)

            all_preds_proba.extend(probabilities.cpu().numpy().flatten())
            if is_hybrid:
                all_y_test.extend(target[:, -1, :].cpu().numpy().flatten())
            else:
                all_y_test.extend(target.cpu().numpy().flatten())
    
    all_y_test = np.array(all_y_test)
    all_preds_proba = np.array(all_preds_proba)
    y_pred = (all_preds_proba >= model_config['snn_model'].get('threshold', 0.5)).astype(int)

    # FLOPs
    logger.info('Mesuaring FLOPs...')
    input_size = real_input.shape[1:]
    if is_hybrid:
        flops = calculate_flops(model, real_input)
    elif model_config['conventional_nn_model'].get('enabled', False):
        flops, _ = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False, verbose=False)
    else:
        flops = np.nan
        logger.warning('Could not calculate FLOPs. Unsupported model type.')

    # Metrics
    final_metrics = calculate_metrics(all_y_test, y_pred, all_preds_proba)
    final_metrics['latency_ms'] = latency_ms
    final_metrics['flops_gflops'] = flops/1e9 if not np.isnan(flops) else None
    final_metrics['size_model'] = get_model_size(model)


    logger.info('Test Set Mertrics:')
    for metric, value in final_metrics.items():
        if isinstance(value, (float, int)):
            logger.info(f'{metric}: {value:.4f}')
        else:
            logger.info(f'{metric}: {value}')

    # Plotting
    plotting(all_y_test, all_preds_proba, final_metrics, plots_dir)

    return final_metrics