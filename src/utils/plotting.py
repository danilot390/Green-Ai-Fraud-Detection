import matplotlib.pyplot as plt    
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, RocCurveDisplay
from src.utils.config_parser import load_config

def confusion_matrix_plot(all_y_test, y_pred,artifact_path):
    """
    Generates and saves confusion matrix.   
    """
    cm = confusion_matrix(all_y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Non-Fraud', 'Fraud'])
    plt.yticks(tick_marks, ['Non-Fraud', 'Fraud'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'{artifact_path}/confusion_matrix.png') if artifact_path else None
    plt.show()

def roc_curve_plot(all_y_test, all_preds_proba, final_metrics, artifact_path):
    """
    Generates and saves ROC curve.
    """
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(all_y_test, all_preds_proba)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=final_metrics['auc_roc']).plot()
    plt.title('ROC Curve')
    plt.savefig(f'{artifact_path}/roc_curve.png') if artifact_path else None
    plt.show()

def plotting(all_y_test, all_preds_proba, final_metrics, artifact_path=False):
    """
    Generates and saves plots: confusion matrix and ROC curve.
    """

    confusion_matrix_plot(all_y_test, (all_preds_proba >= 0.5).astype(int), artifact_path)
    roc_curve_plot(all_y_test, all_preds_proba, final_metrics, artifact_path)