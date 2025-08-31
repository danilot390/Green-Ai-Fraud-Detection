import os
import time
import datetime 
import json 
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from codecarbon import EmissionsTracker
from ptflops import get_model_complexity_info

from src.utils.config_parser import load_config
from src.utils.common import get_device
from src.training.trainer import Trainer, get_dataloaders
from src.utils.metrics import calculate_metrics
from src.utils.flops import calculate_flops
from src.utils.plotting import plotting
from src.utils.reproducibility import set_seed
from src.utils.logger import setup_logger
from src.models.snn_model import SNNModel 
from src.models.conventional_model import ConventionalNN
from src.models.hybrid_model import HybridModel

def main():
    """Main function to run the final evaluation pipeline."""
    # Load configurations
    data_config = load_config('config/data_config.yaml')
    model_config = load_config('config/model_config.yaml')
    training_config = load_config('config/training_config.yaml')
    experiment_config = load_config('config/experiments_config.yaml')
    tracker_on = experiment_config['tracker'].get('enabled', True)

    set_seed(experiment_config['experiment'].get('seed', 42))

    # Run ID based on timestamp or custom
    current_experiment = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if experiment_config['experiment']['run_id'] in ['auto', 'timestamp'] else experiment_config['experiment']['run_id']

    # Directories 
    experiment_dir = os.path.join(experiment_config['experiment']['experiment_dir'], f"run_{current_experiment}")
    plots_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Setup logger 
    log_path = os.path.join(experiment_dir, 'experiment.log')
    logger = setup_logger(log_file=log_path)

    logger.info("Starting final project evaluation pipeline...")

    # Check model 
    is_snn_model = model_config['snn_model']['enabled']
    is_conventional_model = model_config['conventional_nn_model']['enabled']
    is_hybrid_model = is_snn_model and is_conventional_model

    # Model type
    model = 'Hybrid Green AI model' if is_hybrid_model else\
            'SNN Early Warning System' if is_snn_model else\
            'Compressed Fraud Detector CNN' if is_conventional_model else 'Undefined Model'
    logger.info(f"Model Selected: {model}")
    
    try:
        # Get device
        device = get_device()
        logger.info(f"Using device: {device}")
    except Exception as e:
        logger.exception(f"Error loading configuration: {e}")
        return

    # --- Initialize CodeCarbon Traker ---
    if not tracker_on:
        logger.warning("Energy consumption tracking is disabled in the configuration.")
    else:
        if experiment_config['tracker'].get('type', 'CodeCarbon') == 'CodeCarbon':
            tracker = EmissionsTracker(
                project_name=experiment_config['tracker'].get('project', 'Fraud_Detection'),
                output_dir=experiment_dir, 
                log_level='error',
                )
            tracker.start()
            logger.info(f"Measuring energy consumption with {experiment_config['tracker'].get('type', 'CodeCarbon') }...")


    try:
        # Load DataLoaders
        train_loader, val_loader, test_loader, y_train = get_dataloaders(
            dataset_name=experiment_config['experiment']['dataset_name'],
            data_config=data_config,
            training_config=training_config,
            device=device,
            is_snn_model=is_snn_model,
            is_hybrid_model=is_hybrid_model,
            logger=logger,
            )
        
        input_size = train_loader.dataset.features.shape[1] 
        time_steps = data_config['preprocessing_params']['snn_input_encoding']['time_steps'] if is_snn_model else None
        
        if is_hybrid_model:
            model = HybridModel(snn_input_size=input_size, snn_time_steps=time_steps, config=model_config).to(device)
        elif is_snn_model:
            model = SNNModel(input_size=input_size, time_steps=time_steps, config=model_config).to(device)
        elif is_conventional_model:
            model = ConventionalNN(input_size=input_size, config=model_config).to(device)
        else:
            logger.error("Invalid model configuration. At least one model must be enabled.")
        
        # Loss with pos_weight
        num_pos = (y_train == 1).sum().item()
        num_neg = (y_train == 0).sum().item()
        pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=device, dtype=torch.float)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = Adam(model.parameters(), lr=training_config['training_params']['learning_rate'])

        # --- Final Model Training ----
        logger.info("Training final model...")
        trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, device, training_config)
        trainer.run(logger=logger)
        logger.info("Final model training complete.")

        # --- Green AI Metrics Measurement ---
        # a) Measure Inference Latency 
        logger.info("Measuring inference latency...")
        trainer.model.eval()
        real_input, _ = next(iter(test_loader))
        real_input = real_input.to(device)

        start_time = time.time()
        with torch.no_grad():
            # Run a forward pass
            _ = trainer.model(real_input)
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Inference Latency: {latency_ms:.2f} ms")

        # b) Measure FLOPs
        logger.info("Measuring FLOPs...")
        if is_hybrid_model:
            flops = calculate_flops(trainer.model, real_input)
            logger.info(f"Total FLOPs: {flops} FLOPs")
        elif is_conventional_model:
            flops, _ = get_model_complexity_info(trainer.model, (input_size,), as_strings=False, print_per_layer_stat=False, verbose=False)
            logger.info(f"Total FLOPs: {flops} FLOPs")
        else:
            flops = np.nan
            logger.warning(f"Could not calculate FLOPs. Unsupported model type.")

        # c) Final Evaluation on Test Set
        logger.info("Evaluating model on the test set...")
        trainer.model.eval()
        all_preds_proba, all_y_test = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = trainer.model(data)
                probabilities = torch.sigmoid(outputs)
                all_preds_proba.extend(probabilities.cpu().numpy().flatten())
                all_y_test.extend(target[:, -1,:].cpu().numpy().flatten())  if is_hybrid_model else all_y_test.extend(target.cpu().numpy().flatten())

         
        if tracker_on: tracker.stop()
        
        # Convert lists to numpy arrays 
        all_y_test = np.array(all_y_test)
        all_preds_proba = np.array(all_preds_proba)
        y_pred = (all_preds_proba >= model_config['snn_model'].get('threshold',0.5)).astype(int)
        

        #  --- Calculate Metrics and Save Results ---
        final_metrics = calculate_metrics(all_y_test, y_pred, all_preds_proba)
        final_metrics['latency_ms'] = latency_ms
        final_metrics['emissions_kg_co2e'] = tracker.final_emissions if tracker_on else None
        final_metrics['flops_gflops'] = flops / 1e9

        logger.info("Test Set Metrics:")
        for metric, value in final_metrics.items():
            logger.info(f"{metric}: {value:.4f}") if isinstance(value, (float, int)) else f"{metric}: {value}"

        # Plotting results
        plotting(all_y_test, all_preds_proba, final_metrics, plots_dir)

        with open(os.path.join(experiment_dir, 'final_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=4)

        # Save Hyperparameters
        params = {
            "Data Config": data_config,
            "Model Config": model_config,
            "Training Config": training_config
        }
        with open(os.path.join(experiment_dir,'params.json'), 'w') as f:
            json.dump(params, f, indent=4)

        logger.info(f"Hyperparameters and results saved to {experiment_dir}")
        logger.info("Evaluation complete!")

    except Exception as e:
        logger.exception(f"Error during model training: {e}")
        tracker.stop() if tracker_on else None
        return 
if __name__ == '__main__':
    main() 