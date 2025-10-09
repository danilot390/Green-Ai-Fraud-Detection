import time
import numpy as np
import torch
from ptflops import get_model_complexity_info

from src.utils.flops import calculate_flops_hybrid, calculate_flops_hybrid_ml
from src.utils.metrics import calculate_metrics, find_best_threshold
from src.utils.plotting import plotting
from src.utils.model_utils import get_model_size
from src.pipeline.tracking import stop_tracker

def run_evaluation(model, test_loader, device, is_hybrid, plots_dir, model_config, logger, tracker):
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

    for _ in range(5):
        with torch.no_grad():
            _ = model(real_input)

    total_latency = 0
    runs = 100
    for _ in range(100):
        start_time = time.time()
        with torch.no_grad():
            _ = model(real_input)
        total_latency += (time.time() - start_time)* 1000
    
    avg_latency = total_latency/ runs
    logger.info(f'Average Inference Latency over {runs} runs: {avg_latency:.4f} ms')

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
    best_trheshold, best_f1 = find_best_threshold(all_y_test, all_preds_proba)
    logger.info(f'Best threshold: {best_trheshold} \nBest F1: {best_f1}')
    
    y_pred = (all_preds_proba >= best_trheshold).astype(int)

    # FLOPs
    input_size = real_input.shape[1:]
    if is_hybrid:
        flops = calculate_flops_hybrid(model, real_input)
    elif model_config['conventional_nn_model'].get('enabled', False):
        flops, _ = get_model_complexity_info(model, tuple(input_size), as_strings=False, print_per_layer_stat=False, verbose=False)
    elif model.model_name == 'XBoost_CNN_BiLSTM':
        input_seq_len = real_input.shape[1]
        lstm_hidden_s = model_config['xgb_cnn_bilstm_model']['lstm_config'].get('hidden_size', 64)
        cnn_input_s = (1, input_seq_len)
        lstm_input_s = model_config['xgb_cnn_bilstm_model']['input_size']  
        flops_d = calculate_flops_hybrid_ml(
            model, 
            cnn_input_s,
            lstm_input_s,
            lstm_hidden_s,
            input_seq_len,
            logger,
            )
        flops = flops_d.get('Total FLOPs', {0})
        flops = float(next(iter(flops)))
        flops /= 1e9
    else:
        flops = np.nan
        logger.warning('Could not calculate FLOPs. Unsupported model type.')
    emissions = stop_tracker(tracker, logger)

    # Metrics
    final_metrics = calculate_metrics(all_y_test, y_pred, all_preds_proba)
    
    flops_gflops = flops / 1e9 if isinstance(flops, (int, float)) and not np.isnan(flops) else None

    green_metrics = {
        'latency_ms': avg_latency,
        'flops_gflops': flops_gflops,
        'size_model': get_model_size(model),
        'emissions_kg_co2e': emissions,
    }

    logger.info('---- Test Set Metrics ----')
    for metric, value in final_metrics.items():
        if isinstance(value, (float, int)):
            logger.info(f'{metric}: {value:.4f}')
        else:
            logger.info(f'{metric}: {value}')
    
    logger.info('---- Green Model Evaluation ----')
    for metric, value in green_metrics.items():
        if isinstance(value, (float, int)):
            logger.info(f'{metric}: {value:.4f}')
        else:
            logger.info(f'{metric}: {value}')

    # Plotting
    plotting(all_y_test, all_preds_proba, final_metrics, plots_dir)

    return final_metrics