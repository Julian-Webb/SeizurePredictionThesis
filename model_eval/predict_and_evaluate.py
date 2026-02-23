from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

from config.paths import PatientDir, PATHS
from utils.io import pickle_path, save_dataframe_multiformat


def _ensure_results_dir(pdir: PatientDir, model: str, split: str) -> Path:
    """Return pdir.model_eval_dir/<model>/<split>, creating it if necessary."""
    results_dir = pdir.model_eval_dir / model / split
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# todo delete all these metrics?
def safe_div(numerator, denominator, default: float = 0.0):
    return numerator / denominator if denominator != 0 else default


def precision(tp, fp): return safe_div(tp, tp + fp)


def sensitivity(tp, fn): return safe_div(tp, tp + fn)  # sensitivity = recall


def specificity(tn, fp): return safe_div(tn, tn + fp)


def f1(tp, fp, fn):
    prec = precision(tp, fp)
    rec = sensitivity(tp, fn)
    return 2 * safe_div(prec * rec, prec + rec)


# todo delete?
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metrics: list[str] = None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metric_values = {
        'sensitivity': sensitivity(tp, fn),
        'specificity': specificity(tn, fp),
        'precision': precision(tp, fp),
        'f1': f1(tp, fp, fn),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }

    if metrics is None:
        metrics = metric_values.keys()

    try:
        return {m: metric_values[m] for m in metrics}
    except KeyError as e:
        raise ValueError(f"Unknown metric: {e.args[0]}")


# todo rework this function. Use all possible threshold and same for roc and the rest (I think)
# todo make this function calculate for all thresholds
def compute_per_threshold_metrics(
        y_true,
        y_scores,
):
    """Compute per-threshold metrics for later reuse.
    :param y_true: correct labels
    :param y_scores: predicted probability
    """
    # Get precision, sensitivity (recall) for all thresholds
    precisions, sensitivities, thresholds = precision_recall_curve(y_true, y_scores)
    # Drop the last value, since it's artificially added for plotting and doesn't correspond to a threshold
    precisions = precisions[:-1]
    sensitivities = sensitivities[:-1]

    # Calculate F1 scores
    f1_scores = 2 * (precisions * sensitivities) / (precisions + sensitivities + 1e-10)

    per_threshold = DataFrame(
        {'threshold': thresholds, 'precision': precisions, 'sensitivity': sensitivities, 'f1_score': f1_scores}
    ).set_index('threshold')

    # ROC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    return {
        'per_threshold': per_threshold,
        'roc': {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': roc_thresholds,
            'auc': roc_auc,
        }
    }


# todo rework this for roc + other metrics combined
def select_best_threshold(
        metric: str,
        metrics_bundle: dict,
):
    """Pick the optimal threshold for a metric from cached threshold metrics."""
    # Select which score to use
    if metric == 'roc':
        roc = metrics_bundle['roc']
        # todo last score is always 0
        j_scores = roc['tpr'] - roc['fpr']  # j_scores
        scores: Series = Series(j_scores, index=roc['thresholds'])
    else:
        per_thresh: DataFrame = metrics_bundle['per_threshold']
        scores: Series = per_thresh[metric]

    # Compute the best threshold
    thresh = scores.idxmax()
    best_score = scores.loc[thresh]

    return thresh, best_score


def plot_threshold_metrics(
        metrics_bundle: dict,
        output_path=None
):
    """Plot the cached threshold metrics and ROC curve."""
    per_thresh = metrics_bundle['per_threshold']
    thresholds = per_thresh.index  # the index is the threshold
    roc = metrics_bundle['roc']

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14, 10))

    ax = axes[0, 0]
    ax.set_title('Precision vs Sensitivity (Recall)')
    ax.plot(thresholds, per_thresh['precision'], label='Precision', linewidth=2)
    ax.plot(thresholds, per_thresh['sensitivity'], label='Sensitivity', linewidth=2)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.legend()

    ax = axes[0, 1]
    ax.set_title('F1 Score Across Thresholds')
    ax.plot(thresholds, per_thresh['f1_score'], label='F1 Score', linewidth=2, color='green')
    ax.axvline(per_thresh['f1_score'].idxmax(), color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')

    # todo handle this (specificity isn't calculated)
    # ax = axes[1, 0]
    # ax.set_title('Sensitivity vs Specificity')
    # ax.plot(thresholds, per_thresh['specificity'], label='Specificity', linewidth=2)
    # ax.plot(thresholds, per_thresh['sensitivity'], label='Sensitivity (Recall)', linewidth=2)
    # ax.set_xlabel('Threshold')
    # ax.set_ylabel('Score')
    # ax.legend()

    # ROC
    ax = axes[1, 1]
    ax.set_title('ROC Curve')
    ax.plot(roc['fpr'], roc['tpr'], label=f"ROC Curve (AUC = {roc['auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()

    for axis in axes.flatten():
        axis.grid(True, alpha=0.3)
        axis.tick_params(labelbottom=True, labelleft=True)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.close(fig)


# todo also compute time in warning as metric
def predict_and_eval_ptnt(
        pdir: PatientDir,
        models: tuple[str] = ('CNN', 'ensemble'),
        optimization_metric: str = 'roc',
):
    """
    Make model predictions for a patient and evaluate them.
    First, use the training clips to find an optimal threshold for predictions. Then issue predictions for train and
    test clips using this threshold.
    :param optimization_metric: Which metric to optimize the threshold for.
    """
    clips = pd.read_pickle(pickle_path(pdir.clips_table))
    clips = clips[clips['valid']]
    split_idx = pd.read_pickle(pickle_path(pdir.train_test_split))['segment_index']

    clips_per_split = {
        'train': clips[clips['end_seg'] <= split_idx],
        'test': clips[clips['end_seg'] > split_idx],
    }

    for model in models:
        metric_bundles = {}
        for split, split_clips in clips_per_split.items():  # split_clips: clips for this data split
            results_dir = _ensure_results_dir(pdir, model, split)

            # Extract y_scores (predicted probabilities per clip)
            y_scores = split_clips[f'{model}_probability'].values

            # Compute metrics for different thresholds
            y_true = split_clips['preictal'].values
            metrics_bundle = compute_per_threshold_metrics(y_true, y_scores)
            metric_bundles[split] = metrics_bundle
            plot_threshold_metrics(metrics_bundle, output_path=results_dir / 'metrics_plot.png')
            save_dataframe_multiformat(metrics_bundle['per_threshold'],
                                       results_dir / 'threshold_analysis.csv',
                                       csv_index=True)

        # Find the optimal threshold using only the train (!) set
        best_thresh, best_score = select_best_threshold(optimization_metric, metric_bundles['train'])

        # Issue binary predictions based on threshold
        y_scores = clips[f'{model}_probability'].values
        # todo is it > or >= ?
        # todo save predictions
        y_pred = y_scores > best_thresh

        # todo html report


def main(pdirs: list[PatientDir] = PATHS.patient_dirs()):
    """Evaluate every provided patient directory."""
    for pdir in pdirs:
        print(f"======== Processing patient: {pdir.name}")
        predict_and_eval_ptnt(pdir)


if __name__ == "__main__":
    main([
        PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-01-MINIFAKE'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-01-FAKE'),
        # PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-1'),
        # PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-2'),
        # PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-3'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-01'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-03'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-04'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-05'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-07'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-12'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-15'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-16'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-17')
    ])
