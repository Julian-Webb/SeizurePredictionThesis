"""
Calculate, plot, and store various metrics
"""
import logging
import multiprocessing
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.metrics import roc_curve, auc, precision_recall_curve, PrecisionRecallDisplay, \
    RocCurveDisplay, precision_score, recall_score

from config.paths import PatientDir, PATHS
from event_based_metrics import event_based_metrics
from utils.io import pickle_path, save_dataframe_multiformat
from utils.utils import timeit

SUBSELECT_THRESHOLDS_GRANULARITY: float = 0.005  # todo 0.005


def _ensure_results_dir(pdir: PatientDir, split: str, model: str | None = None) -> Path:
    """Return pdir.model_eval_dir/<split>/<model>, creating it if necessary."""
    results_dir = pdir.model_eval_dir / split
    if model is not None:
        results_dir = results_dir / model
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def subselect_thresholds(thresholds: np.ndarray, granularity: float = SUBSELECT_THRESHOLDS_GRANULARITY):
    # Round to granularity
    rounded = np.round(thresholds / granularity) * granularity
    unique = np.sort(np.unique(rounded))
    return unique


def plot_threshold_metrics(
        y_true: np.ndarray,
        data_per_model: dict,
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

        #### ROC
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name=model)
        roc_disp.plot(ax=axes[0, 2])

        #### Event-based sensitivity vs. Time in Correct Warning (TICW)
        ticw = 1 - ebms['relative_tifw']
        axes[1, 0].plot(ebms['event_based_sensitivity'].values, ticw.values, label=model)

        #### Event-based f1 score vs. thresholds
        line, = axes[1, 1].plot(ebms['event_based_f1'], label=model)
        axes[1, 1].axvline(ebms['event_based_f1'].idxmax(), linestyle='--', alpha=0.5, color=line.get_color())

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


@timeit
def eval_ptnt(
        pdir: PatientDir,
        models: tuple[str] = ('CNN', 'ensemble'),
        precomputed_event_based_metrics: bool = False,
):
    """Evaluate both train and test splits for models."""
    logging.info(f"==== Processing patient: {pdir.name}")

    #### Load patient data
    edfs = pd.read_pickle(pickle_path(pdir.edf_files_table))
    szr_starts = pd.read_pickle(pickle_path(pdir.valid_szr_starts_file))['start_mtz'].values
    clips = pd.read_pickle(pickle_path(pdir.clips_table))
    split_idx = pd.read_pickle(pickle_path(pdir.train_test_split))['segment_index']

    clips = clips[clips['valid']]
    clips_per_split = {
        'train': clips[clips['end_seg'] <= split_idx],
        'test': clips[clips['end_seg'] > split_idx],
    }

    #### Calculate Event-based metrics, which we optimize the threshold for
    # Use thresholds based on the training data
    ebm_threshs_per_model = {}
    for model in models:
        all_threshs = clips_per_split['train'][f'{model}_probability'].unique()
        ebm_threshs_per_model[model] = subselect_thresholds(all_threshs)

    # Calculate event-based metrics for each split and model
    ebms_per_split = {}  # ebm: event-based metrics
    for split, split_clips in clips_per_split.items():  # split_clips: clips for this data split
        if precomputed_event_based_metrics:
            ebms_per_split[split] = {
                model: pd.read_pickle(pickle_path(pdir.model_eval_dir / split / model / 'event_based_metrics')) for
                model in models
            }
        else:
            ebms_per_split[split], _ = event_based_metrics(
                split_clips,
                edfs,
                szr_starts,
                thresholds_per_model=ebm_threshs_per_model,
                models=models,
                logging_info=f'[{pdir.name} - {split}]'
            )

    # Calculate the optimal threshold
    best_thresh_per_model = {}
    for model in models:
        scores: Series = ebms_per_split['train'][model]['event_based_f1']
        best_thresh_per_model[model] = scores.idxmax()

    # Save data, including various metrics for the optimal threshold
    for split, split_clips in clips_per_split.items():
        for model in models:
            best_thresh = best_thresh_per_model[model]
            y_pred = split_clips[f'{model}_probability'] >= best_thresh
            y_true = split_clips['preictal'].values

            ebms = ebms_per_split[split][model]
            metrics_to_save = pd.Series({
                'model': model,
                'data_split': split,
                'total_clips': len(y_true),
                'preictal_clips': int(y_true.sum()),
                'non_preictal_clips': int(len(y_true) - y_true.sum()),
                'best_threshold': best_thresh,
                'relative_tifw': ebms['relative_tifw'].loc[best_thresh],
                'event_based_sensitivity': ebms['event_based_sensitivity'].loc[best_thresh],
                'event_based_f1': ebms['event_based_sensitivity'].loc[best_thresh],
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
            }, name=pdir.name)

            results_dir = _ensure_results_dir(pdir, split, model)

            save_dataframe_multiformat(ebms, results_dir / f'event_based_metrics', csv_index=True)
            save_dataframe_multiformat(metrics_to_save, results_dir / 'metrics', csv_index=True)

    # Plot various metrics
    for split, split_clips in clips_per_split.items():
        results_dir = _ensure_results_dir(pdir, split)

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
            title=f'{pdir.name} - {split}',
            output_path=results_dir / f'metrics_plot.png'
        )


def main(
        pdirs: list[PatientDir] = PATHS.patient_dirs(),
        serial_processing: bool = False,
        precomputed_event_based_metrics: bool = False
):
    """Evaluate every provided patient directory."""
    eval_partial = partial(eval_ptnt, precomputed_event_based_metrics=precomputed_event_based_metrics)

    if serial_processing:
        for pdir in pdirs:
            eval_partial(pdir)
    else:
        with multiprocessing.Pool() as p:
            p.map(eval_partial, pdirs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    main([
        PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-01-MINIFAKE'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-01-FAKE'),
        PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-1'),
        PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-2'),
        PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-3'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-01'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-03'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-04'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-05'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-07'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-12'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-15'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-16'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-17'),
    ],
        serial_processing=False,
        precomputed_event_based_metrics=True,
    )
