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
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, PrecisionRecallDisplay, \
    RocCurveDisplay, precision_score, recall_score, roc_auc_score

from config import PatientDir, PATHS, save_dataframe_multiformat, pickle_path
from preprocessing.dataset_partitioning import partition_dataframe
from utils.utils import timeit


# todo not sure if this code is doing what it should.
# todo shouldn't samples be independent for this test?
def hanley_mcneil_test(N1, N2, auc):
    """
    Hanley-McNeil method for ROC-AUC significance/better than chance classification performance.
    """
    # for chance predictor
    a = 0.5
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)

    if not isinstance(N1, (list, tuple, np.ndarray)):
        n1 = N1
        n2 = N2
        std_auc = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))
        p = norm.sf(auc, loc=a, scale=std_auc)

        if auc < 0.5 and p < 0.5:
            p = 1 - p
    else:
        # auc=auc.flatten()
        p = []
        for n1, n2, auc_ in zip(N1, N2, auc):
            std_auc = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))
            p_ = norm.sf(auc_, loc=a, scale=std_auc)

            if auc_ < 0.5 and p_ < 0.5:
                p_ = 1 - p_
                p.append(p_)
        p = np.array(p)
    return p


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
        best_thresh = data['best_thresh']

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
        axes[0, 1].axvline(best_thresh, linestyle='--', alpha=0.5, color='k')

        #### ROC
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name=model)
        roc_disp.plot(ax=axes[0, 2])

        #### Event-based sensitivity vs. 1 - Time in False Warning (TIFW)
        tifw_inv = 1 - ebms['rel_tifw']
        line, = axes[1, 0].plot(ebms['rel_szrs_pred'].values, tifw_inv.values, label=model)
        # Mark the point that corresponds to the best threshold
        axes[1, 0].scatter(ebms['rel_szrs_pred'].loc[best_thresh], tifw_inv.loc[best_thresh], marker='x',
                           color=line.get_color())

        #### Event-based f1 score vs. thresholds
        line, = axes[1, 1].plot(ebms['event_based_f1'], label=model)
        axes[1, 1].axvline(best_thresh, linestyle='--', alpha=0.5, color=line.get_color())

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
        title='Event-based sensitivity vs. 1 - Relative Time in False Warning (TIFW)',
        xlabel='Event-based sensitivity',
        ylabel=' 1 -Relative Time in False Warning',
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
def eval_models_for_pdir(
        pdir: PatientDir,
        models: tuple[str] = ('CNN', 'ensemble'),
):
    """Evaluate both train and test splits for models."""
    logging.info(f"[{pdir.name}] Evaluating models: {models}")
    clip_scores = pd.read_pickle(pdir.clip_scores_table.pickle)
    clip_scores = clip_scores[clip_scores['valid']]
    clip_scores_per_split = partition_dataframe(clip_scores, pdir)

    # Iterate through splits and models to process
    for split, split_clips in clip_scores_per_split.items():
        data_per_model = {}  # for plotting
        y_true = split_clips['preictal'].values

        for model in models:
            model_subdir = pdir.model_eval_subdir(split, model)
            ebms = pd.read_pickle(model_subdir.ebm_table.pickle)  # Retrieve Event-based metrics (ebms)
            best_thresh = ebms['event_based_f1'].idxmax()  # Optimize the threshold

            y_scores = split_clips[f'{model}_score']
            y_pred: Series = y_scores >= best_thresh

            # todo check this with Sot - the results are super sketchy
            # Hanley-McNeil test for ROC-AUC significance/better than chance classification performance.
            roc_auc = roc_auc_score(y_true, y_scores)
            n_pos = y_true.sum()
            n_neg = (~y_true).sum()
            p_hanley_mcneil = hanley_mcneil_test(n_pos, n_neg, roc_auc)
            print(f'p_hanley_mcneil: {p_hanley_mcneil:.5f} [{pdir.name} {split} {model}] ')

            metrics_to_save = pd.Series({
                'model': model,
                'data_split': split,
                'total_clips': len(y_true),
                'preictal_clips': int(y_true.sum()),
                'non_preictal_clips': int(len(y_true) - y_true.sum()),
                'best_threshold': best_thresh,
                'rel_tifw': ebms['rel_tifw'].loc[best_thresh],
                'rel_szrs_pred': ebms['rel_szrs_pred'].loc[best_thresh],
                'event_based_f1': ebms['event_based_f1'].loc[best_thresh],
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'roc_auc': roc_auc,
                'p_hanley_mcneil': p_hanley_mcneil,
            }, name=pdir.name)

            save_dataframe_multiformat(metrics_to_save, model_subdir.metrics_table, save_index=True)
            data_per_model[model] = {'y_scores': y_scores.values, 'event_based_metrics': ebms,
                                     'best_thresh': best_thresh}

        # Plot various metrics
        plot_threshold_metrics(
            y_true,
            data_per_model,
            title=f'{pdir.name} - {split}',
            output_path=pdir.model_eval_subdir(split).metrics_plot,
        )


def eval_models_for_pdirs(
        pdirs: list[PatientDir] = PATHS.patient_dirs(),
        serial_processing: bool = False,
):
    """Evaluate every provided patient directory."""
    if serial_processing:
        for pdir in pdirs:
            eval_models_for_pdir(pdir)
    else:
        with multiprocessing.Pool() as p:
            p.map(eval_models_for_pdir, pdirs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    eval_models_for_pdirs(
        pdirs=PATHS.patient_dirs(),
        serial_processing=False,
    )
