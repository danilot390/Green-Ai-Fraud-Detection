import numpy as np 
import torch, time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    average_precision_score, confusion_matrix, f1_score, roc_auc_score, fbeta_score)

from src.utils.common import to_int_array

def find_best_threshold(y_true, y_probs):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

def calculate_metrics(y_true, y_pred_binary, y_pred_proba=None):
    """
    Calculate key classification metrics for imbalanced datasets
    """

    #  Convert to NumPy arrays and remove extra dimensions
    y_true = to_int_array(y_true)
    y_pred_binary = to_int_array(y_pred_binary)

    # Calculate standard metrics.
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, pos_label=1, zero_division=0)
    f2_score = fbeta_score(y_true, y_pred_binary, beta=2, pos_label=1, zero_division=0)
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall':recall,
        'f1_score':f1,
        'f2_score':f2_score
    }

    # Calculate AUC-ROC if predicted probalities are provided
    if y_pred_proba is not None and len(np.unique(y_true))>1:
        y_pred_proba = np.asarray(y_pred_proba).squeeze()

        try:
             metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics['auc_roc']= np.nan

        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        except:
            metrics['pr_auc'] = np.nan

    else:
        metrics['auc_roc']= np.nan
        metrics['pr_auc'] = np.n

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
    metrics['Confusion Matrix'] = f'TP={float(tp):.4f}, FP={float(fp):.4f}, FN={float(fn):.4f}, TN={float(tn):.4f}'

    return metrics

def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Run model on dataloader and return metrics via Calculate Metrics function.
    """
    model.eval()
    all_probs, all_targets = [], []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            if output.dim() == 3:
                output = output[:, -1, :] 

            if output.shape[1] == 1:
                probs = torch.sigmoid(output).squeeze()
            else:
                probs = torch.softmax(output, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            if target.dim() == 3:
                target = target[:,-1,0]
                
            all_targets.extend(target.view(-1).cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    best_trheshold, _ = find_best_threshold(all_targets, all_probs)

    y_pred = (all_probs >= best_trheshold).astype(int)
    
    return calculate_metrics(all_targets, y_pred, all_probs)

def bench_mark_inference(model, dataloader, device, num_runs=3):
    """
    Benchmarks the average inference time of the model over a specified number of runs.
    """
    start = time.time()
    
    for _ in range(num_runs):
        evaluate_model(model, dataloader, device)
    elapsed = (time.time() - start) / num_runs
    return elapsed