import time
import numpy as np
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

from src.utils.flops import calculate_flops_hybrid, calculate_flops_hybrid_ml
from src.utils.metrics import calculate_metrics, find_best_threshold
from src.utils.plotting import plotting
from src.utils.model_utils import get_model_size
from src.pipeline.tracking import stop_tracker

def run_evaluation(model, test_loader, device, is_hybrid, plots_dir, model_config, logger, tracker):
    """
    Evaluate the model, measuring latency, FLOPs, size, metrics (accuracy, precision, recall, F1, ROC-AUC,
    average precision, confusion matrix), and generating plots.
    """
    logger.info('Evaluating model')

    # --- Detect if we are dealing with a stacking wrapper ---
    is_stacking = (hasattr(model, "hybrid_model") and hasattr(model, "predict_proba")
                   and not isinstance(model, nn.Module))
    if is_stacking:
        base_model = model.hybrid_model  # PyTorch HybridModel
        base_model.eval()
    else:
        base_model = model
        base_model.eval()

    all_preds_proba, all_y_test = [], []

    # Latebcy
    logger.info('Measuring inference latency...')
    real_input, _ = next(iter(test_loader))
    real_input = real_input.to(device)

    # Warm-up
    for _ in range(5):
        with torch.no_grad():
            if is_stacking:
                _ = model.predict_proba(real_input)  # returns numpy
            else:
                _ = base_model(real_input)

    total_latency = 0
    runs = 100
    for _ in range(runs):
        start_time = time.time()
        with torch.no_grad():
            if is_stacking:
                _ = model.predict_proba(real_input)
            else:
                _ = base_model(real_input)
        total_latency += (time.time() - start_time) * 1000

    avg_latency = total_latency / runs
    logger.info(f'Average Inference Latency over {runs} runs: {avg_latency:.4f} ms')
    # Predictions
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)

            if is_stacking:
                # HybridStackingModel: proba returned as numpy [B]
                proba = model.predict_proba(data)
                probabilities = np.asarray(proba).flatten()
            else:
                # Plain PyTorch model: outputs logits → apply sigmoid
                outputs = base_model(data)
                probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_preds_proba.extend(probabilities)

            # Target handling unchanged
            if is_hybrid:
                all_y_test.extend(target[:, -1, :].cpu().numpy().flatten())
            else:
                all_y_test.extend(target.cpu().numpy().flatten())

    all_y_test = np.array(all_y_test)
    all_preds_proba = np.array(all_preds_proba)

    best_threshold, best_f1 = find_best_threshold(all_y_test, all_preds_proba)
    logger.info(f'Best threshold: {best_threshold} \nBest F1: {best_f1}')

    y_pred = (all_preds_proba >= best_threshold).astype(int)

    # Flops
    input_size = real_input.shape[1:]

    if is_hybrid:
        flops = calculate_flops_hybrid(base_model, real_input, logger)

    elif model_config['conventional_nn_model'].get('enabled', False):
        flops, _ = get_model_complexity_info(
            base_model,
            tuple(input_size),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        logger.info(f'FLOPs: {flops}')

    elif getattr(base_model, "model_name", "") == 'XBoost_CNN_BiLSTM':
        input_size_scalar = real_input.shape[1]  # features
        seq_len = real_input.shape[1]
        cnn_input_s = (1, real_input.shape[1])
        lstm_hidden_s = model_config['xgb_cnn_bilstm_model']['lstm_config'].get('hidden_size', 64)

        flops_d = calculate_flops_hybrid_ml(
            base_model,
            cnn_input_s,
            input_size_scalar,
            lstm_hidden_s,
            seq_len,
            logger,
        )
        flops = flops_d.get('Total FLOPs', np.nan)
    else:
        flops = np.nan
        logger.warning('Could not calculate FLOPs. Unsupported model type.')

    # End CodeCarbon tracker (if provided)
    emissions = stop_tracker(tracker, logger)

    # Metrics
    final_metrics = calculate_metrics(all_y_test, y_pred, all_preds_proba)

    # Robust FLOPs → GFLOPs conversion
    if isinstance(flops, dict):
        # Try to extract first numeric value if dict
        try:
            first_val = next(iter(flops.values()))
            flops_val = float(first_val)
        except Exception:
            flops_val = np.nan
    else:
        flops_val = float(flops) if isinstance(flops, (int, float)) else np.nan

    flops_gflops = flops_val / 1e9 if not np.isnan(flops_val) else None

    green_metrics = {
        'latency_ms': avg_latency,
        'flops_gflops': flops_gflops,
        'size_model': get_model_size(base_model),
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
        if isinstance(value, (float, int)) and value is not None and not np.isnan(value):
            logger.info(f'{metric}: {value:.4f}')
        else:
            logger.info(f'{metric}: {value}')

    # Plotting
    plotting(all_y_test, all_preds_proba, final_metrics, plots_dir)

    return final_metrics