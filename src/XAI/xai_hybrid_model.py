import torch
import numpy as np
import os
import sys

from src.models.hybrid_model import HybridModel
from src.utils.config_parser import load_config
from src.utils.common import get_device
from src.utils.logger import setup_logger

from lime.lime_tabular import LimeTabularExplainer

def explain_model_with_lime(model, explainer, data_point, class_names, config,device, logger, path_dir='lime_explanation.html'):
    """
    Generates a LIME explanation for a single data point.
    """

    def predict_fn(data):
        """ 
        Prediction function to be used by LIME.
        Expects data and returns probabilities.
        """
        # Convert numpy array to PyTorch tensor
        data_tensor = torch.from_numpy(data).float().to(device)
        
        # Reshape data to fit the hybrid model's expected input shape (Batch, TimeSteps, Features)
        num_timesteps = model.snn_time_steps
        data_tensor_sequential = data_tensor.unsqueeze(1).repeat(1, num_timesteps, 1)

        # Pass through the model
        model.eval()
        with torch.no_grad():
            logits = model(data_tensor_sequential)
            probabilities = torch.sigmoid(logits)

        # Ensure probabilities are returned in shape (N, 2) for binary classification
        if probabilities.ndim == 1:
            probabilities = probabilities.unsqueeze(1)
        if probabilities.shape[1] == 1:
            probabilities = torch.cat([1 - probabilities, probabilities], dim=1)

        # Return the probabilities as a numpy array
        probabilities = probabilities.detach().cpu().numpy()
        return probabilities

    # Generate the explanation
    explanation = explainer.explain_instance(
        data_row=data_point,
        predict_fn=predict_fn,
        num_features=config['num_features'],
        labels=[0, 1],      # The labels to explain (Non-Fraud, Fraud)
    )

    # Predicted class
    predicted_class = np.argmax(explanation.predict_proba)
    logger.info(f"Predicted class for the instance: `{class_names[predicted_class]}`")
    
    # Print the explanation for the user
    logger.info("LIME Explanation for the selected instance")
    logger.info(f"Original prediction: `{explanation.class_names[predicted_class]}` "
          f"with probability `{explanation.predict_proba[predicted_class]}`")

    # Get the top features contributing to the prediction
    logger.info("Top features contributing to the prediction:")
    sorted_features = sorted(explanation.as_list(), key=lambda x: abs(x[1]), reverse=True)
    for feature_id, weight in sorted_features:
        logger.info(f"  - {feature_id}: {weight:.4f}")

    # Save explanation as HTML
    if config.get('save_html', False):
        explanation.save_to_file(path_dir)
        logger.info(f"LIME explanation saved to `{path_dir}`.")

if __name__ == '__main__':
    # Load configuration files
    training_config = load_config('config/training_config.yaml')
    data_config = load_config('config/data_config.yaml')
    model_config = load_config('config/model_config.yaml')
    xai_config = load_config('config/xai_config.yaml')
    logger = setup_logger()

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Load the trained model
    model_path = training_config['training_params']['model_save_path']+\
                xai_config['xai_methods'].get('model', 'best_HybridModel.pth')

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train a model first.")
        sys.exit()
    print(f"Loading model from {model_path}...")

    # Instantiate the model by passing the configuration dictionary
    snn_input_size = data_config['preprocessing_params']['num_features']
    snn_time_steps = data_config['preprocessing_params']['snn_input_encoding']['time_steps']
    
    # Check the model configuration
    print("\n--- Checking Model Configuration ---")
    print(f"SNN input size: {snn_input_size}")
    print(f"SNN time steps: {snn_time_steps}")
    print(f"SNN hidden layers: {model_config['snn_model']['hidden_layers']}")
    print(f"Conventional layers: {model_config['conventional_nn_model']['mlp_layers']}")
    print("--- ---------------------------- ---")

    # Instantiate the model
    model = HybridModel(snn_input_size, snn_time_steps, model_config).to(device)
    
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=device)

    # This allows the script to run, but for a permanent solution
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: Could not load the full state_dict. Loading with strict=False due to: {e}")
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Load the test dataset to get an instance to explain
    dataset_name = data_config['dataset_training']['dataset_name']
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
    
    # Let's pick a random instance to show how it works
    fraud_indices = np.where(y_test_labels == 1)[0]
    if len(fraud_indices) > 0:
        rand_instance_idx = fraud_indices[np.random.randint(0, len(fraud_indices))]
        print(f"Found a fraudulent transaction at index {rand_instance_idx}. Explaining this instance.")
    else:
        # Fallback to a random instance if no fraud is found
        rand_instance_idx = np.random.randint(0, len(X_test_tabular))
        print(f"No fraudulent instances found in the test set. Explaining a random instance: {rand_instance_idx}")
    
    instance_to_explain = X_test_tabular[rand_instance_idx]
    
    # Generate and print the explanation for the selected instance 
    explain_model_with_lime(model, explainer, instance_to_explain, class_names, xai_config['xai_methods']['lime'], device, logger)
