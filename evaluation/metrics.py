"""
Evaluation metrics for classification.
"""
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute binary classification metrics.
    
    Args:
        y_true: True labels (+1 or -1)
        y_pred: Predicted labels (+1 or -1)
        
    Returns:
        Dictionary of metrics
    """
    # Convert to binary if needed
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)
    
    # Compute confusion matrix
    # Force 2x2 shape even if only one class is present
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    return metrics


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None
) -> Dict:
    """
    Compute multi-class classification metrics.
    
    Args:
        y_true: True class indices
        y_pred: Predicted class indices
        class_names: Optional list of class names
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    if class_names is not None:
        # Force fixed size matrix over all classes [0..K-1]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    else:
        cm = confusion_matrix(y_true, y_pred)
    
    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'n_classes': len(np.unique(y_true))
    }
    
    if class_names:
        metrics['class_names'] = class_names
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str = None,
    normalize: bool = True,
    title: str = "Confusion Matrix"
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure (optional)
        normalize: Whether to normalize the matrix
        title: Plot title
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

