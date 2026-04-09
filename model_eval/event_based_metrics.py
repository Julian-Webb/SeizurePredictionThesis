import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Timedelta

from config import PatientDir, PATHS, save_dataframe_multiformat
from config.intervals import INTERVENTION, SPH
from preprocessing.create_segments import check_szrs_in_sphs
from preprocessing.dataset_partitioning import partition_dataframe
from utils.utils import FunctionTimer


def check_szrs_predicted(
        preictal_clips: DataFrame,
        szr_starts: np.ndarray,
):
    # For the true positive clips, check which seizures were predicted.
    tps = preictal_clips[preictal_clips['y_pred']]

    # Matrix that shows which SPHs contain which seizures (n_szrs, n_clips)
    in_sph = check_szrs_in_sphs(tps['sph_start'].values, tps['sph_end'].values, szr_starts)

    # Sanity check: All preictal clip's SPH should contain a seizure
    preictal_clip_has_szr = in_sph.any(axis=0)
    assert preictal_clip_has_szr.all(), "All preictal clip's SPH should contain a seizure, but some don't."

    szr_pred = Series(in_sph.any(axis=1), index=szr_starts, name='predicted')
    return szr_pred


def non_overlapping_interval_duration(
        starts: Series,
        ends: Series,
):
    """
    Calculates the total duration and detailed intervals for non-overlapping time intervals.

    Note: The starts and ends must be sorted in ascending order!

    This function processes two Pandas Series: `starts` (start times) and `ends`
    (end times) of intervals. It ensures that overlapping intervals are adjusted
    to become non-overlapping by truncating their end times. The function then
    calculates the total duration of these adjusted intervals and returns the
    detailed intervals alongside their respective durations.

    Parameters
    ----------
    starts : Series
        A Pandas Series representing the start times of the intervals.
    ends : Series
        A Pandas Series representing the end times of the intervals.

    Returns
    -------
    tuple
        A tuple containing the following:
        - total_duration : Timedelta
            The total duration of all non-overlapping intervals.
        - ivs : DataFrame
            A Pandas DataFrame with the following columns:
            - `start`: Start times of the intervals.
            - `end`: End times of the non-overlapping intervals.
            - `duration`: Durations of the non-overlapping intervals.
    """
    next_start = starts.shift(-1)
    overlaps_next = (ends > next_start).fillna(False)
    ends_non_overlap = ends.where(~overlaps_next, next_start)

    # Sanity check
    remaining_overlaps = ends_non_overlap > next_start
    assert not remaining_overlaps.any(), "There are still overlaps"

    iv_durations = ends_non_overlap - starts
    total_duration: Timedelta = iv_durations.sum()
    ivs = DataFrame({'start': starts, 'end': ends_non_overlap, 'duration': iv_durations})

    return total_duration, ivs


def event_based_metrics_for_ptnt(
        clips: DataFrame,  # correct number of szrs
        szr_starts: np.ndarray,
        score_col: str,
        thresholds: Iterable[float] = None,
        intervention_duration: Timedelta = INTERVENTION.exact_dur,
        sph_duration: Timedelta = SPH.exact_dur,
):
    """
    Calculate event-based metrics for seizure prediction in a dataset.

    Note: The function assumes that there's no seizure after the last clip for at least the length of the
    intervention + SPH.

    This function evaluates the performance of predictive models for detecting
    seizures by calculating event-based metrics across multiple thresholds. It
    determines the number of seizures predicted, computes the time in false warning (TIFW),
    and calculates relative metrics such as sensitivity and TIFW. Additionally, it computes the event-based F1
    score for each threshold value.

    Parameters
    ----------
    clips : DataFrame
        A DataFrame containing prediction clips with columns relevant to the
        analysis ('end_mtz', 'preictal', and the score column).
    szr_starts : np.ndarray
        An array of seizure start times.
    score_col : str
        Name of the column in the `clips` DataFrame that contains prediction scores from the model.
    thresholds : Iterable[float], optional
        A collection of threshold values for evaluating the model's
        performance. If not provided, the thresholds will be derived from
        unique values in the `score_col` of the DataFrame.
    intervention_duration : Timedelta
        The duration of the intervention period after each clip's end time.
    sph_duration : Timedelta
        The duration of the seizure prediction horizon (SPH).

    Returns
    -------
    metrics : DataFrame
        A DataFrame indexed by thresholds with the following columns:
            - 'n_szrs_pred': The number of seizures predicted for a given
              threshold.
            - 'abs_tifw': The total absolute TIFW for a given threshold.
            - 'rel_szrs_pred': Relative number of seizures predicted, calculated
              as the fraction of seizures identified out of the total seizure
              count.
            - 'rel_tifw': Relative TIFW for a given threshold.
            - 'event_based_f1': The event-based F1 score combining sensitivity
              and false warning time.

    szrs_pred_series : Series
        A Pandas Series indexed by the threshold and seizure start time,
        containing a boolean value indicating whether each seizure was predicted
        for the corresponding threshold.
    """
    # ---- Preprocessing
    # clips: keep only relevant columns and make a copy
    if 'valid' in clips.columns:
        clips = clips[clips['valid']]
    else:
        logging.warning("No 'valid' column in clips DataFrame. Assuming all clips are valid.")
    clips = clips[['end_mtz', 'preictal', score_col]].copy()

    if thresholds is None:
        thresholds = np.sort(clips[score_col].unique())

    # ---- Determine SPHs for clips
    clips['sph_start'] = clips['end_mtz'] + intervention_duration
    clips['sph_end'] = clips['sph_start'] + sph_duration

    # ---- Split preictal and non-preictal clips
    # noinspection PyTypeChecker
    preict: DataFrame = clips[clips['preictal']].copy()
    nonpre: DataFrame = clips[~clips['preictal']].copy()

    metrics_rows = []
    szrs_pred_per_threshold = {}

    for thresh in thresholds:
        preict['y_pred'] = preict[score_col] >= thresh
        nonpre['y_pred'] = nonpre[score_col] >= thresh

        # ---- Check which seizures were predicted
        szrs_pred = check_szrs_predicted(preict, szr_starts)
        szrs_pred_per_threshold[thresh] = szrs_pred

        # ---- Calculate the time in false warning (TIFW)
        fp_clips = nonpre[nonpre['y_pred']]  # false positives
        abs_tifw = non_overlapping_interval_duration(fp_clips['sph_start'], fp_clips['sph_end'])[0]

        metrics_rows.append({
            'threshold': thresh,
            'n_szrs_pred': szrs_pred.sum(),
            'abs_tifw': abs_tifw,
        })

    # ---- Create per-threshold results
    metrics = DataFrame(metrics_rows).set_index('threshold')

    n_szrs = len(szr_starts)
    rel_szrs_pred = metrics['n_szrs_pred'] / n_szrs  # Event-based sensitivity
    metrics['rel_szrs_pred'] = rel_szrs_pred

    # only non-preictal (negative) clips can contribute to TIFW, since only they can be false positives
    max_tifw = non_overlapping_interval_duration(nonpre['sph_start'], nonpre['sph_end'])[0]
    metrics['rel_tifw'] = metrics['abs_tifw'] / max_tifw

    # Calculate event-based F1 Score with invested TIFW and sensitivity
    rel_tifw_inv = 1 - metrics['rel_tifw']
    metrics['event_based_f1'] = 2 * (rel_szrs_pred * rel_tifw_inv) / (rel_szrs_pred + rel_tifw_inv)

    szrs_pred_series = pd.concat(szrs_pred_per_threshold, names=['threshold', 'seizure_start'])
    return metrics, szrs_pred_series


def event_based_metrics_for_pdir(task: tuple[PatientDir, str, str]) -> tuple[str, str, str]:
    pdir, split, model = task
    log_prefix = f"{pdir.name} {split} {model}"
    logging.info(f"[{log_prefix}] 🎬 Calculating Event-Based Metrics...")

    with FunctionTimer(f"[{log_prefix}] ✅ Completed Event-Based Metrics"):
        # Load once per task
        clip_scores = partition_dataframe(pd.read_pickle(pdir.clip_scores_table.pickle), pdir)[split]
        clip_scores = clip_scores[clip_scores['valid']]
        szrs = partition_dataframe(pd.read_pickle(pdir.valid_szr_starts_file.pickle), pdir)[split]

        ebms, szrs_pred = event_based_metrics_for_ptnt(
            clip_scores,
            szrs['start_mtz'].values,
            score_col=f'{model}_score',
        )

        res_dir = pdir.model_eval_subdir(split, model)
        save_dataframe_multiformat(ebms, res_dir.ebm_table, save_index=True)
        save_dataframe_multiformat(szrs_pred, res_dir.szr_pred_table, save_index=True)

    return pdir.name, split, model


def event_based_metrics_for_pdirs(
        pdirs: list[PatientDir],
        splits: Iterable[str] = ('train', 'test'),
        models: Iterable[str] = ('CNN', 'ensemble'),
        serial_processing: bool = False,
        max_workers: int = 48,
):
    tasks = list(product(pdirs, splits, models))
    logging.info(f'Starting {len(tasks)} EBM tasks.')

    if serial_processing:
        for task in tasks:
            event_based_metrics_for_pdir(task)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            exe.map(event_based_metrics_for_pdir, tasks, chunksize=1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    pdirs_ = PATHS.patient_dirs()
    event_based_metrics_for_pdirs(pdirs_, serial_processing=False)
