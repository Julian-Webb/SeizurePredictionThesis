import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, f1_score

from models.load_data import load_data
from config.paths import PatientDir


def find_best_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Find the optimal threshold for binary classification.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities (scores from 0 to 1)
        metric: Which metric to optimize for:
            - 'f1': F1 score (balance precision and recall)
            - 'roc': Youden's J statistic (maximize TPR - FPR)
            - 'precision': Maximize precision
            - 'recall': Maximize recall

    Returns:
        best_threshold: The optimal threshold value
        best_score: The score value at the optimal threshold
        metrics_dict: Dictionary of various metrics at this threshold
    """
    # Flatten probabilities if needed
    y_pred_proba = np.array(y_pred_proba).flatten()

    if metric == 'f1':
        # Try all unique probability values as thresholds
        thresholds = np.unique(y_pred_proba)
        f1_scores = []

        for threshold in thresholds:
            predictions = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, predictions, zero_division=0)
            f1_scores.append(f1)

        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_score = f1_scores[best_idx]

    elif metric == 'roc':
        # Use ROC curve to find optimal threshold (Youden's J)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr  # Youden's J statistic
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        best_score = j_scores[best_idx]

    elif metric in ['precision', 'recall']:
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_pred_proba)

        if metric == 'precision':
            best_idx = np.argmax(precision_vals[:-1])  # Exclude the last point
            best_threshold = thresholds[best_idx]
            best_score = precision_vals[best_idx]
        else:  # recall
            best_idx = np.argmax(recall_vals[:-1])
            best_threshold = thresholds[best_idx]
            best_score = recall_vals[best_idx]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Calculate metrics at the best threshold
    predictions = (y_pred_proba >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel().tolist()

    metrics_dict = {
        'threshold': best_threshold,
        'score': best_score,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall / TPR
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,  # True Negative Rate
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1': f1_score(y_true, predictions, zero_division=0),
    }

    return best_threshold, best_score, metrics_dict


def plot_threshold_metrics(y_true, y_pred_proba, thresholds=None):
    """
    Plot various metrics across different thresholds.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        thresholds: List of thresholds to evaluate. If None, uses 100 evenly spaced values.
    """
    y_pred_proba = np.array(y_pred_proba).flatten()

    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    metrics = {'threshold': [], 'precision': [], 'recall': [], 'f1': [], 'specificity': []}

    for threshold in thresholds:
        predictions = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

        metrics['threshold'].append(threshold)
        metrics['precision'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        metrics['recall'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        metrics['f1'].append(f1_score(y_true, predictions, zero_division=0))
        metrics['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(metrics['threshold'], metrics['precision'], label='Precision', linewidth=2)
    axes[0, 0].plot(metrics['threshold'], metrics['recall'], label='Recall', linewidth=2)
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Precision vs Recall')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(metrics['threshold'], metrics['f1'], label='F1 Score', linewidth=2, color='green')
    axes[0, 1].axvline(metrics['threshold'][np.argmax(metrics['f1'])], color='green', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score Across Thresholds')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(metrics['threshold'], metrics['specificity'], label='Specificity', linewidth=2)
    axes[1, 0].plot(metrics['threshold'], metrics['recall'], label='Sensitivity (Recall)', linewidth=2)
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Sensitivity vs Specificity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    axes[1, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# Main execution
pdir: PatientDir = PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-07')

mlp = tf.keras.models.load_model(pdir.ensemble_model)
# unlike for training, we neither shuffle nor subsample segs here.
data = load_data(pdir, 'features', subsample_shuffle_and_subselect_types=False, test=True)
x, y = data['x'], data['y']
x, y = x[:10000], y[:10000]
probabilities = mlp.predict(x)

# Find optimal threshold using different metrics
print("=" * 60)
print("THRESHOLD OPTIMIZATION ANALYSIS")
print("=" * 60)

for metric in ['f1', 'roc']:
    best_thresh, best_score, metrics = find_best_threshold(y, probabilities, metric=metric)
    print(f"\n{metric.upper()} Score Optimized Threshold:")
    print(f"  Best Threshold: {best_thresh:.4f}")
    print(f"  Best {metric.upper()} Score: {best_score:.4f}")
    print(f"  Sensitivity (TPR): {metrics['sensitivity']:.4f}")
    print(f"  Specificity (TNR): {metrics['specificity']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Confusion Matrix: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")

# Use F1-optimized threshold by default
best_thresh, _, _ = find_best_threshold(y, probabilities, metric='f1')
predictions = probabilities > best_thresh

print(f"\n{'=' * 60}")
print(f"Using threshold: {best_thresh:.4f}")
print(f"Predictions shape: {predictions.shape}")
print(f"Positive predictions: {predictions.sum()} ({100*predictions.sum()/len(predictions):.2f}%)")
