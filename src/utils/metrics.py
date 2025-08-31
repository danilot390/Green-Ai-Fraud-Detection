import numpy as np 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.utils.common import to_numpy_squeezed

def calculate_metrics(y_true, y_pred_binary, y_pred_proba=None):
    """
    Calculate key classification metrics for imbalanced datasets
    """

    #  Convert to NumPy arrays and remove extra dimensions
    y_true = to_numpy_squeezed(y_true)
    y_pred_binary = to_numpy_squeezed(y_pred_binary)

    # Explicitly cast to integer type
    y_true = y_true.astype(int)
    y_pred_binary = y_pred_binary.astype(int)

    # Calculate standard metrics.
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, pos_label=1, zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall':recall,
        'f1_score':f1,
    }

    # Calculate AUC-ROC if predicted probalities are provided
    if y_pred_proba is not None and len(np.unique(y_true))>1:
        y_pred_proba = to_numpy_squeezed(y_pred_proba)
        try:
             auc_roc = roc_auc_score(y_true, y_pred_proba)
             metrics['auc_roc'] = auc_roc
        except ValueError:
            metrics['auc_roc']= np.nan
    else:
        metrics['auc_roc']= np.nan

    return metrics