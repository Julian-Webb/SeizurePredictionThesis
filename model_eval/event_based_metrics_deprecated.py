"""
Calculate event-based metrics: Time in False Warning and Seizures predicted.
"""
import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from typing import Iterable, Optional, Any, Tuple

import numpy as np
import pandas as pd
import portion as P
from pandas import DataFrame, Timedelta, Series

from config import PATHS, PatientDir, save_dataframe_multiformat
from config.intervals import Interval, INTERVENTION, SPH
from preprocessing.dataset_partitioning import partition_dataframe
from utils.utils import timeit

SUBSELECT_THRESHOLDS_GRANULARITY: float = 0.005


def subselect_thresholds(thresholds: np.ndarray, granularity: float = SUBSELECT_THRESHOLDS_GRANULARITY):
    # Round to granularity
    rounded = np.round(thresholds / granularity) * granularity
    unique = np.sort(np.unique(rounded))
    # Prepend a value slightly lower than the lowest if it's not already 0
    if unique[0] > 0:
        low = max(0, unique[0] - granularity)
        unique = np.insert(unique, 0, low)
    return unique


def missing_recording_intervals(
        edfs: DataFrame,
        minimum_duration_missing: Timedelta = Timedelta(seconds=0),
        last_interval_duration: Optional[Timedelta] = None,
) -> DataFrame:
    """

    :param edfs: The patient's edf DataFrame
    :param minimum_duration_missing: How much time needs to be missing for it to be considered a missing interval.
    :param last_interval_duration: If specified, will give the last interval by this duration. Otherwise, it doesn't have an end.
    :return:
    """
    # The ends of recordings until the start of the next recording are the missing intervals
    missing_ivs = DataFrame({
        'previous_file': edfs['file_name'].values,
        'next_file': edfs['file_name'].iloc[1:].tolist() + [None],
        'start': edfs['end_mtz'].values,
        # shift times up by one row to align with the start of the next recording
        'end': edfs['start_mtz'].iloc[1:].tolist() + [None],
    })

    # Extend the last interval (the end of the last recording), in case the last seizure prediction horizon extends
    #  beyond it.
    if last_interval_duration is not None:
        last_i = missing_ivs.index[-1]
        missing_ivs.loc[last_i, 'end'] = \
            missing_ivs.loc[last_i, 'start'] + last_interval_duration

    # remove intervals with 0 duration
    missing_ivs['duration'] = missing_ivs['end'] - missing_ivs['start']
    missing_ivs = missing_ivs[missing_ivs['duration'] > minimum_duration_missing]

    return missing_ivs


def subtract_intervals(
        base_ivs: DataFrame,
        subtract_ivs: P.Interval,
        start_col: str = 'start',
        end_col: str = 'end',
) -> DataFrame:
    """Subtract inclusive intervals from base intervals, splitting as needed. Inputs must have columns 'start' and 'end'
    :param base_ivs: Intervals to subtract from
    :param subtract_ivs: Intervals to subtract - combined into a single portion Interval
    """
    result = []
    for _, base in base_ivs.iterrows():
        # Convert type
        base_iv = P.closed(base[start_col], base[end_col])
        # Subtract all intervals
        remaining = base_iv - subtract_ivs
        # remaining may contain multiple pieces
        if len(remaining) == 0:
            logging.debug(
                f"Base interval {base[start_col]} - {base[end_col]} was fully covered by subtract intervals and was removed.")
        for iv in remaining:
            result.append((iv.lower, iv.upper))

    return DataFrame(result, columns=[start_col, end_col])


def merge_intervals(intervals: DataFrame, start_col: str = 'start', end_col: str = 'end'):
    """Merge inclusive intervals. Input must have columns 'start' and 'end'"""
    union = dataframe2portion_interval(intervals, start_col, end_col)
    # Convert back to DataFrame
    merged = [[iv.lower, iv.upper] for iv in union]
    return DataFrame(merged, columns=['start', 'end'])


def dataframe2portion_interval(df: DataFrame, start_col: str = 'start', end_col: str = 'end') -> P.Interval:
    """Convert a DataFrame with start and end columns to a portion library interval"""
    ivs = df.apply(lambda row: P.closed(row[start_col], row[end_col]), axis=1)

    union = P.empty()
    for iv in ivs.values:
        union |= iv
    return union


def safe_dataframe_concat(objs: list, concat_kwargs) -> DataFrame:
    include = [obj for obj in objs if not len(obj)]
    if len(include) > 0:
        return pd.concat(include, **concat_kwargs)
    else:
        return DataFrame(objs[0])


# todo it's possible to seizures to not be predicted at the beginning of the test set, because we're not including the clips that would predict it (1:05h before the split)
# todo The TIFW can never equal 1 because we're using the total recording time. The duration that should be in warning should be subtracted.
@timeit(kwarg_names=['logging_info'])
def event_based_metrics(
        clips: DataFrame,
        edfs: DataFrame,
        szr_starts: np.ndarray,
        thresholds_per_model: Optional[dict[str, Iterable[float]]] = None,
        models: Iterable[str] = ("CNN", "ensemble"),
        intervention_iv: Interval = INTERVENTION,
        sph_iv: Interval = SPH,
        logging_info: str = '[unknown patient - split]',
) -> Tuple[dict[str, DataFrame], dict[str, dict[float, dict[str, Any]]]]:
    """
    Takes a set of predictions and calculates the time in false warning (TIFW) and relative number of seizures predicted.

    Seizure predictions are issued at the warning time (WT). After that, there's the intervention interval/time (IT),
    followed by the seizure prediction horizon (SPH) in which the seizure should occur. If no seizure occurs within the
    SPH, the SPH is added to the TIFW.

    The SPH is, in a way, the preictal interval, when looking forwards in time from the prediction, instead of backwards
    from the seizure. Thus, the SPH has the same duration as the preictal interval.

    This function conceptually works as follows:

    1. From the clip predictions, the WTs are determined as the end time of the clips.
    2. The SPHs are calculated from the WTs. The SPHs will (almost certainly) be overlapping.
    3. SPHs are verified by checking if they contain a seizure.
    4. It is also checked which seizures were predicted.
    5. False SPHs are merged to make them non-overlapping
    6. Intervals with missing EEG data are subtracted from the merged SPHs because it's unknown whether a seizure
       occurred during them.
    7. The absolute TIFW is the total duration of the remaining SPHs
    8. The relative TIFW is the absolute TIFW / total recording duration.
    9. The relative number of seizures predicted is #seizures predicted / #total seizures

    :param clips: DataFrame with columns '{model}_score', 'end_mtz', 'valid' (optional)
    :param edfs: The patient's EDF DataFrame
    :param szr_starts: The patient's seizure starts
    :param thresholds_per_model: If specified, will use these thresholds per model instead of all thresholds in clips.
    :return: Two nested dicts summary_metrics and intermediary_results with keys (model, metric, threshold)
    """
    clips = clips.copy()
    if 'valid' in clips.columns:
        clips = clips[clips['valid']]
    else:
        logging.warning("No 'valid' column in clips DataFrame. Assuming all clips are valid.")
    # Keep only relevant columns and make a copy
    clips.drop(columns=['segs_in_clip', 'full', 'n_existing', 'sufficient_data', 'valid'],
               errors='ignore', inplace=True)

    #### Calculate all possible SPHs so that we can later select the processed SPHs per threshold
    all_wts: Series = clips['end_mtz']
    all_sphs = DataFrame({'types': clips['types']})
    all_sphs['start'] = all_wts + intervention_iv.exact_dur
    all_sphs['end'] = all_sphs['start'] + sph_iv.exact_dur

    # Check whether each SPH is correct: Loop through seizures and mark SPHs that contain at least one as correct.
    all_sphs['correct'] = False
    for szr in szr_starts:
        # Note: The end must be treated as exclusive here, otherwise the first inter_pre clip before a seizure would
        #  be marked as correct, since its end is exactly the seizure start.
        szr_in_sphs_mask = (all_sphs['start'] <= szr) & (szr < all_sphs['end'])
        all_sphs.loc[szr_in_sphs_mask, 'correct'] = True

    # Calc values which are the same for all thresholds and models
    missing_ivs = missing_recording_intervals(
        edfs,
        minimum_duration_missing=Timedelta(seconds=1),
        last_interval_duration=2 * (intervention_iv.exact_dur + sph_iv.exact_dur)
    )
    missing_ivs = dataframe2portion_interval(missing_ivs)

    total_recording_time = edfs['duration'].sum()

    metrics_per_model = {}  # Metrics with a single number that can be turned into a DataFrame
    intermediate_results_per_model = {}

    for model in models:
        logging.info(f"{logging_info} [{model}] Processing event-based metrics...")
        score_col = f'{model}_score'

        # Use this model's thresholds if specified, otherwise use all unique thresholds in the clips for this model
        if thresholds_per_model is None:
            model_thresholds = np.sort(clips[score_col].unique())
        else:
            model_thresholds = thresholds_per_model[model]

        # Per threshold dicts:
        metrics = {k: {} for k in ['abs_tifw', 'n_szrs_pred']}
        intermediate_results = {k: {} for k in ['szrs_pred', 'sphs']}

        for thresh in model_thresholds:
            logging.debug(f'{logging_info} [{model}] Processing threshold {thresh}...')
            # Select SPHs for this threshold
            y_pred = clips[score_col] >= thresh
            sphs = all_sphs[y_pred]

            # Check which seizures were predicted
            szrs_pred = Series(False, index=szr_starts, name='szr_pred')
            if len(sphs) > 0:
                # Vectorized: for each seizure, check if it's in any SPH
                # Shape: (n_seizures, n_sphs)
                # Note: The end must be treated as exclusive here, otherwise the first inter_pre clip before a seizure
                # would be marked as correct, since its end is exactly the seizure start.
                in_sph = (
                        (szr_starts[:, None] >= sphs['start'].values) &
                        (szr_starts[:, None] < sphs['end'].values)
                )

                szrs_pred[:] = in_sph.any(axis=1)

            # Merge correct and incorrect SPHs separately and subtract missing intervals
            def process_sphs(sphs_: DataFrame) -> DataFrame:
                merged = merge_intervals(sphs_[['start', 'end']])
                remaining = subtract_intervals(merged, missing_ivs)
                remaining['duration'] = remaining['end'] - remaining['start']
                return remaining

            correct_sphs = process_sphs(sphs[sphs['correct']])
            incorrect_sphs = process_sphs(sphs[~sphs['correct']])

            # Combine SPHs again to save
            correct_sphs['correct'] = True
            incorrect_sphs['correct'] = False
            processed_sphs = safe_dataframe_concat([correct_sphs, incorrect_sphs], {'ignore_index': True})
            processed_sphs.sort_values(by='start', ignore_index=True, inplace=True)

            # Calculate & save metrics
            metrics['abs_tifw'][thresh] = incorrect_sphs['duration'].sum()
            metrics['n_szrs_pred'][thresh] = szrs_pred.sum()
            intermediate_results['szrs_pred'][thresh] = szrs_pred
            intermediate_results['sphs'][thresh] = processed_sphs

        # Calculate further metrics for this model in bulk
        metrics = DataFrame(metrics)
        metrics['rel_tifw'] = pd.to_timedelta(metrics['abs_tifw']) / total_recording_time

        rel_szrs_pred = metrics['n_szrs_pred'] / len(szr_starts)
        metrics['rel_szrs_pred'] = rel_szrs_pred

        # Calculate event-based F1 Score with "Time in Correct warning (ticw) and sensitivity
        relative_ticw = 1 - metrics['rel_tifw']
        metrics['event_based_f1'] = 2 * (rel_szrs_pred * relative_ticw) / (rel_szrs_pred + relative_ticw)

        metrics_per_model[model] = metrics
        intermediate_results_per_model[model] = intermediate_results

    return metrics_per_model, intermediate_results_per_model


def _load_data_per_split(pdir: PatientDir):
    """
    :return: data per split (type[edfs, clips, szr_starts], split[train, test])
    """
    #### Select correct EDFs per split and split the EDF that contain the split (if any)
    edfs = pd.read_pickle(pdir.edf_files_table.pickle)
    partition = pd.read_pickle(pdir.dataset_partition.pickle)
    test_start_mtz = partition.loc['start_mtz', 'test']
    edfs = edfs[['file_name', 'start_mtz', 'end_mtz']].copy()

    # Select train and test EDFs that don't contain the split
    train_edfs = edfs[edfs['end_mtz'] <= test_start_mtz].copy()
    test_edfs = edfs[test_start_mtz <= edfs['start_mtz']].copy()

    # Split the EDF that spans the boundary (if any)
    spans_split = (edfs['start_mtz'] < test_start_mtz) & (edfs['end_mtz'] > test_start_mtz)
    if spans_split.any():
        idx = spans_split.idxmax()
        edf = edfs.loc[idx]
        before, after = edf.copy(), edf.copy()
        before['end_mtz'], after['start_mtz'] = test_start_mtz, test_start_mtz
        train_edfs = pd.concat([train_edfs, DataFrame(before).T])
        test_edfs = pd.concat([DataFrame(after).T, test_edfs])

    for split, edfs in (['train', train_edfs], ['test', test_edfs]):
        edfs['duration'] = edfs['end_mtz'] - edfs['start_mtz']

    # Get clip scores and szr starts
    clip_scores = pd.read_pickle(pdir.clip_scores_table.pickle)
    clip_scores = clip_scores[clip_scores['valid']]
    clip_scores = partition_dataframe(clip_scores, test_start_mtz=test_start_mtz)

    szr_starts = pd.read_pickle(pdir.all_szr_starts_file.pickle)['start_mtz'].values

    per_split = {
        'edfs': {'train': train_edfs, 'test': test_edfs},
        'clips': clip_scores,
        'szr_starts': {
            'train': szr_starts[szr_starts < test_start_mtz],
            'test': szr_starts[szr_starts >= test_start_mtz],
        }
    }
    return per_split


def calc_ptnt_split_metrics(args):
    """Process a single patient-split combination."""
    pdir, split, models = args
    logging_info = f'[{pdir.name} - {split}]'
    logging.info(f'{logging_info} Calculating event based metrics...')
    per_split = _load_data_per_split(pdir)

    threshs_per_model = {}
    for model in models:
        all_threshs = per_split['clips']['train'][f'{model}_score'].unique()
        threshs_per_model[model] = subselect_thresholds(all_threshs)

    # noinspection PyTypeChecker
    ebms, _ = event_based_metrics(
        clips=per_split['clips'][split],
        edfs=per_split['edfs'][split],
        szr_starts=per_split['szr_starts'][split],
        thresholds_per_model=threshs_per_model,
        models=models,
        logging_info=logging_info
    )

    for model in models:
        res_dir = pdir.model_eval_subdir(split, model)
        save_dataframe_multiformat(ebms[model], res_dir.ebm_table, save_index=True)


def calc_metrics(pdirs: list[PatientDir],
                 splits: tuple[str] = ('train', 'test'),
                 models: tuple[str] = ('CNN', 'ensemble'),
                 serial_processing: bool = False):
    if serial_processing:
        for pdir, split in product(pdirs, splits):
            calc_ptnt_split_metrics((pdir, split, models))
    else:
        tasks = [(pdir, split, models) for pdir, split in product(pdirs, splits)]
        with ProcessPoolExecutor() as p:
            p.map(calc_ptnt_split_metrics, tasks)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    pdirs_ = PATHS.patient_dirs()
    calc_metrics(pdirs_, serial_processing=True)
