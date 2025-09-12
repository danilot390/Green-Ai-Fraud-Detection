from src.pipeline.setup import setup_experiment
from src.pipeline.tracking import start_tracker, stop_tracker
from src.pipeline.model import setup_model, finalize_model
from src.pipeline.training import run_training
from src.pipeline.evaluation import run_evaluation
from src.utils.reproducibility import set_seed
from src.utils.common import get_device, save_to_json
from src.data.dataloaders import get_dataloaders

def main():
    """
    Runs the end-to-end Green Ai fraud detection pipeline.

    Tasks:
        - Load configs, set up logging, device, and reproducibility.
        - Initialize date loaders and model (SNN, CNN, or hybrid).
        - Optionally apply compression (Quantization, Pruning or both).
        - Define loss/optimizer and train the model.
        - Optionally convert and save a quantized model.
        - Evaluate on test set (Latency, FLOPs, metrics).
        - Save results, plots, and experiment metadata.
    """
    data_config, model_config, training_config, experiment_config, logger, experiment_dir, plots_dir = setup_experiment()
    logger.info("Initiate final project evaluation pipeline...")
    set_seed(experiment_config['experiment'].get('seed', 42), logger)
    device = get_device(training_config['training_params'].get('device', 'cpu'))

    tracker = start_tracker(experiment_config, experiment_dir, logger)

    # Set up data
    train_loader, val_loader, test_loader, y_train = get_dataloaders(
        data_config, training_config, device, experiment_config['experiment']['dataset_name'], logger
    )
    input_size = train_loader.dataset.features.shape[1] 
    time_steps = data_config['preprocessing_params']['snn_input_encoding'].get('time_steps', None)
    
    # Model
    model, criterion, optimizer, qat_enabled, is_hybrid = setup_model(input_size, time_steps, device, model_config, training_config, y_train, logger)

    # Train
    model = run_training(model, criterion, optimizer, input_size, time_steps, train_loader, val_loader, device, training_config, logger, qat_enabled)
    model = finalize_model(model, qat_enabled, model_config, logger)
    
    # Evaluate
    run_evaluation(model, test_loader, device, is_hybrid, plots_dir, model_config, logger)
    stop_tracker(tracker, logger)
    
    # Save Hyperparameters
    params = {
            "Data Config": data_config,
            "Model Config": model_config,
            "Training Config": training_config,
            "Experiment Config": experiment_config
        }
    save_to_json(params, experiment_dir, 'hyperparams.json', logger)

if __name__ == '__main__':
    main() 