"""
Calculate, plot, and store various metrics
"""
import logging
import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.metrics import roc_curve, auc, precision_recall_curve, PrecisionRecallDisplay, \
    RocCurveDisplay, precision_score, recall_score, roc_auc_score

from config import PatientDir, PATHS, MultiPath, save_dataframe_multiformat, pickle_path
from model_eval.event_based_metrics import load_data_per_split, ensure_results_dir
from utils.utils import timeit


def plot_threshold_metrics(
        y_true: np.ndarray,
        data_per_model: dict,
        best_thresh_per_model: dict[str, float],
        title: str = '',
        output_path: Path = None,
):
    """

    :param y_true:
    :param data_per_model: dict with data per model
    :param output_path:
    :return:
    """
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(21, 14))

    if title:
        fig.suptitle(title, fontsize=30)

    for model, data in data_per_model.items():
        y_scores = data['y_scores']
        ebms = data['event_based_metrics']

        #### Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        # Remove the final values from precision and recall because they are not based on a real threshold.
        precision, recall = precision[:-1], recall[:-1]
        pr_disp = PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name=model)
        pr_disp.plot(ax=axes[0, 0])

        #### F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
        f1_max_thresh = pr_thresholds[np.argmax(f1_score)]
        line, = axes[0, 1].plot(pr_thresholds, f1_score, label=model)
        axes[0, 1].axvline(f1_max_thresh, linestyle='--', alpha=0.5, color=line.get_color())
        # Plot best thresh given
        axes[0, 1].axvline(best_thresh_per_model[model], linestyle='--', alpha=0.5, color='k')

        #### ROC
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name=model)
        roc_disp.plot(ax=axes[0, 2])

        #### Event-based sensitivity vs. Time in Correct Warning (TICW)
        ticw = 1 - ebms['rel_tifw']
        line, = axes[1, 0].plot(ebms['rel_szrs_predicted'].values, ticw.values, label=model)
        # Mark the point that corresponds to the best threshold
        bt = best_thresh_per_model[model]
        axes[1, 0].scatter(ebms['rel_szrs_predicted'].loc[bt], ticw.loc[bt], marker='x', color=line.get_color())

        #### Event-based f1 score vs. thresholds
        line, = axes[1, 1].plot(ebms['event_based_f1'], label=model)
        axes[1, 1].axvline(best_thresh_per_model[model], linestyle='--', alpha=0.5, color=line.get_color())

    #### Settings per axis
    axes[0, 0].set_title('Precision vs. Recall')

    axes[0, 1].set(
        title='F1 Score Across Thresholds',
        xlabel='Threshold',
        ylabel='F1 Score',
    )

    axes[0, 2].set_title('ROC Curve')
    # Plot random classifier for ROC
    axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')

    axes[1, 0].set(
        title='Event-based sensitivity vs. Relative Time in Correct Warning (TICW)',
        xlabel='Event-based sensitivity',
        ylabel='Relative Time in Correct Warning',
    )

    axes[1, 1].set(
        title='Event-based F1 score vs. thresholds',
        xlabel='Threshold',
        ylabel='Event-based F1 score',
    )

    # Shared settings
    # Hide the unused subplot
    axes[1, 2].axis("off")
    for axis in axes.flatten()[:5]:
        axis.grid(True, alpha=0.3)
        axis.legend()
        axis.tick_params(labelbottom=True, labelleft=True)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.close(fig)


@timeit(kwarg_names=['pdir'])
def eval_ptnt(
        pdir: PatientDir,
        models: tuple[str] = ('CNN', 'ensemble'),
):
    """Evaluate both train and test splits for models."""
    logging.info(f"[{pdir.name}] Evaluating models: {models}")

    clips_per_split = load_data_per_split(pdir)['clips']

    #### Retrieve Event-based metrics (ebms), which we optimize the threshold for
    ebms_per_split = {}  # keys: (split, model, metric)
    for split, split_clips in clips_per_split.items():  # split_clips: clips for this data split
        ebms_per_split[split] = {
            model: pd.read_pickle(pickle_path(pdir.model_eval_dir / split / model / 'event_based_metrics')) for
            model in models
        }

    # Calculate the optimal threshold
    best_thresh_per_model = {}
    for model in models:
        scores: Series = ebms_per_split['train'][model]['event_based_f1']
        best_thresh_per_model[model] = scores.idxmax()

    # Save data, including various metrics for the optimal threshold
    for split, split_clips in clips_per_split.items():
        for model in models:
            best_thresh = best_thresh_per_model[model]
            y_scores = split_clips[f'{model}_probability']
            y_pred = y_scores >= best_thresh
            y_true = split_clips['preictal'].values

            ebms = ebms_per_split[split][model]
            metrics_to_save = pd.Series({
                'model': model,
                'data_split': split,
                'total_clips': len(y_true),
                'preictal_clips': int(y_true.sum()),
                'non_preictal_clips': int(len(y_true) - y_true.sum()),
                'best_threshold': best_thresh,
                'rel_tifw': ebms['rel_tifw'].loc[best_thresh],
                'rel_szrs_predicted': ebms['rel_szrs_predicted'].loc[best_thresh],
                'event_based_f1': ebms['event_based_f1'].loc[best_thresh],
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'roc_auc': roc_auc_score(y_true, y_scores),
            }, name=pdir.name)

            results_dir = ensure_results_dir(pdir, split, model)
            save_dataframe_multiformat(metrics_to_save, MultiPath(results_dir, 'metrics'), save_index=True)

    # Plot various metrics
    for split, split_clips in clips_per_split.items():
        results_dir = ensure_results_dir(pdir, split)

        data_per_model = {}
        for model in models:
            data_per_model[model] = {
                # Extract predicted probabilities per clip (y_scores)
                'y_scores': split_clips[f'{model}_probability'].values,
                'event_based_metrics': ebms_per_split[split][model],
            }

        plot_threshold_metrics(
            y_true=split_clips['preictal'].values,
            data_per_model=data_per_model,
            best_thresh_per_model=best_thresh_per_model,
            title=f'{pdir.name} - {split}',
            output_path=results_dir / f'metrics_plot.png'
        )


def main(
        pdirs: list[PatientDir] = PATHS.patient_dirs(),
        serial_processing: bool = False,
):
    """Evaluate every provided patient directory."""
    if serial_processing:
        for pdir in pdirs:
            eval_ptnt(pdir)
    else:
        with multiprocessing.Pool() as p:
            p.map(eval_ptnt, pdirs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    main(
        pdirs=PATHS.patient_dirs(),
        serial_processing=False,
    )
