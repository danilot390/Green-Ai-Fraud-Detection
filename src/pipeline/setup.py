import os, datetime
from src.utils.config_parser import load_config
from src.utils.logger import setup_logger

def setup_experiment(log_file_e=True):
    """
    Loads configs, logger, experiment dirs
    """
    # Load configs
    data_config = load_config(os.path.join('config','data_config.yaml'))
    model_config = load_config(os.path.join('config','model_config.yaml'))
    training_config = load_config(os.path.join('config','training_config.yaml'))
    experiment_config = load_config(os.path.join('config','experiments_config.yaml'))

    # Run ID
    run_id = experiment_config['experiment'].get('run_id', 'auto')
    if run_id in ['auto', 'timestamp']:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Directories
    if log_file_e:
        experiment_dir = os.path.join(experiment_config['experiment'].get('experiment_dir', 'models'), f'run_{run_id}')
        plots_dir = os.path.join(experiment_dir, "plots")
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Setup logging
        log_path = os.path.join(experiment_dir, 'experiment.log')
    else:
        experiment_dir, plots_dir = None, None
    
    logger = setup_logger(log_file=log_path) if log_file_e else setup_logger()

    return data_config, model_config, training_config, experiment_config, logger, experiment_dir, plots_dir