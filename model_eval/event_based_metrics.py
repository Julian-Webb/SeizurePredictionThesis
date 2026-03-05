import logging
from typing import Iterable, Optional, Any, Tuple

import numpy as np
import pandas as pd
import portion as P
from pandas import DataFrame, Timedelta, Series

from config.intervals import Interval, INTERVENTION, SPH
from config.paths import PATHS
from utils.io import pickle_path
from utils.utils import timeit


# todo try this for all patients with minimum duration missing super low (minus 1 year or something) to find file overlaps
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


@timeit
def event_based_metrics(
        clips: DataFrame,
        edfs: DataFrame,
        szr_starts: np.ndarray,
        thresholds_per_model: Optional[dict[str, Iterable[float]]] = None,
        models: Iterable[str] = ("CNN", "ensemble"),
        intervention_iv: Interval = INTERVENTION,
        sph_iv: Interval = SPH,
        logging_info: str = 'unknown patient',
) -> Tuple[dict[str, DataFrame], dict[str, dict[float, dict[str, Any]]]]:
    """
    Takes a set of predictions and calculates the time in false warning (TIFW) and event_based_sensitivity.

    Seizure predictions are issued at the warning time (WT). After that, there's the intervention interval/time (IT),
    followed by the seizure prediction horizon (SPH) in which the seizure should occur. If no seizure occurs within the
    SPH, the SPH is added to the TIFW.

    The SPH is, in a way, the preictal interval, when looking forwards in time from the prediction, instead of backwards
    from the seizure. Thus, the SPH has the same duration as the preictal interval.

    This function works as follows:

    1. From the clip predictions, the WTs are determined as the end time of the clips.
    2. The SPHs are calculated from the WTs. The SPHs will (almost certainly) be overlapping.
    3. Intervals with missing EEG data will be subtracted from the SPHs because we don't know if a seizure occurred
       during them.
    4. For each seizure, we check if it's contained in an SPH. We save which SPHs contain a seizure and which seizures
       were detected.
    5. The false SPHs will be merged to make them non-overlapping.
    6. The duration of the remaining SPHs is the absolute TIFW.
    7. The absolute TIFW / total duration time is the relative TIFW.

    :param clips: DataFrame with columns '{model}_probability', 'end_time', 'valid' (optional)
    :param edfs: The patient's EDF DataFrame
    :param szr_starts: The patient's seizure starts
    :param thresholds_per_model: If specified, will use these thresholds per model instead of all thresholds in clips.
    :param subselect_thresholds_granularity: If specified, will subsample the thresholds to this granularity
    :return: Two nested dicts summary_metrics and intermediary_results with keys model, metric, threshold
    """
    clips = clips.copy()

    if 'valid' in clips.columns:
        clips = clips[clips['valid']]
    else:
        logging.warning("No 'valid' column in clips DataFrame. Assuming all clips are valid.")

    # Calc values which are the same for all thresholds and models
    missing_intervals = missing_recording_intervals(
        edfs,
        minimum_duration_missing=Timedelta(seconds=1),
        last_interval_duration=2 * (intervention_iv.exact_dur + sph_iv.exact_dur)
    )
    missing_intervals = dataframe2portion_interval(missing_intervals)
    total_recording_time = edfs['duration'].sum()

    summary_metrics = {}  # Metrics with a single number that can be turned into a DataFrame
    intermediary_results = {}

    for model in models:
        logging.info(f"{logging_info} Processing event-based metrics for model {model}...")

        prob_col = f'{model}_probability'

        # Use this model's thresholds if specified, otherwise use all unique thresholds in the clips for this model
        if thresholds_per_model is None:
            model_thresholds = np.sort(clips[prob_col].unique())
        else:
            model_thresholds = thresholds_per_model[model]

        model_summary_metrics = {m: {} for m in
                                 ['absolute_tifw', 'relative_tifw', 'event_based_sensitivity', 'event_based_f1']}
        model_intermediary_results = {m: {} for m in ['sphs', 'szrs_detected']}

        for thresh in model_thresholds:
            logging.debug(f'Processing threshold {thresh}...')
            #### Warning Times
            y_pred = clips[prob_col] >= thresh
            # noinspection PyTypeChecker
            warn_times: Series = clips.loc[y_pred, 'end_time']

            #### Seizure prediction horizons
            sphs = DataFrame({'types': clips.loc[warn_times.index, 'types']}, index=warn_times.index)
            sphs['start'] = warn_times + intervention_iv.exact_dur
            sphs['end'] = sphs['start'] + sph_iv.exact_dur

            #### Subtract missing EEG intervals
            remaining_sphs = subtract_intervals(sphs, missing_intervals)

            #### Check whether each SPH is correct
            # Loop through seizures and mark SPHs that contain at least one as correct.
            # Also save which seizures were detected.
            remaining_sphs['correct'] = False
            szrs_detected = Series(False, index=szr_starts, name='szr_detected')

            for szr in szr_starts:
                # Note: The end must be treated as exclusive here, otherwise the first inter_pre clip before a seizure would
                #  be marked as correct, since its end is exactly the seizure start.
                szr_in_sphs_mask = (remaining_sphs['start'] <= szr) & (szr < remaining_sphs['end'])
                remaining_sphs.loc[szr_in_sphs_mask, 'correct'] = True
                if szr_in_sphs_mask.any():
                    szrs_detected.loc[szr] = True

            correct_sphs = remaining_sphs[remaining_sphs['correct']]
            incorrect_sphs = remaining_sphs[~remaining_sphs['correct']]

            #### Merge SPHs
            merged_correct_sphs = merge_intervals(correct_sphs[['start', 'end']])
            merged_incorrect_sphs = merge_intervals(incorrect_sphs[['start', 'end']])

            #### Calculate absolute TIFW
            merged_correct_sphs['duration'] = merged_correct_sphs['end'] - merged_correct_sphs['start']
            merged_incorrect_sphs['duration'] = merged_incorrect_sphs['end'] - merged_incorrect_sphs['start']
            absolute_tifw = Timedelta(merged_incorrect_sphs['duration'].sum())

            #### Calculate relative TIFW, event-based sensitivity, and their harmonic mean (event-based F1 Score)
            relative_tifw = absolute_tifw / total_recording_time
            event_based_sensitivity = szrs_detected.sum() / len(szr_starts)
            relative_ticw = 1 - relative_tifw
            event_based_f1 = 2 * (event_based_sensitivity * relative_ticw) / (event_based_sensitivity + relative_ticw)

            #### Add results to output
            model_summary_metrics['absolute_tifw'][thresh] = absolute_tifw
            model_summary_metrics['relative_tifw'][thresh] = relative_tifw
            model_summary_metrics['event_based_sensitivity'][thresh] = event_based_sensitivity
            model_summary_metrics['event_based_f1'][thresh] = event_based_f1

            merged_correct_sphs['correct'] = True
            merged_incorrect_sphs['correct'] = False
            processed_sphs = pd.concat([merged_correct_sphs, merged_incorrect_sphs], ignore_index=True)
            processed_sphs.sort_values(by='start', ignore_index=True, inplace=True)

            model_intermediary_results['sphs'][thresh] = processed_sphs
            model_intermediary_results['szrs_detected'][thresh] = szrs_detected

        summary_metrics[model] = DataFrame(model_summary_metrics)
        intermediary_results[model] = model_intermediary_results

    return summary_metrics, intermediary_results


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    pdir = PATHS.patient_dirs()[0]

    metrics = event_based_metrics(
        pd.read_pickle(pickle_path(pdir.clips_table)),
        pd.read_pickle(pickle_path(pdir.edf_files_table)),
        pd.read_pickle(pickle_path(pdir.valid_szr_starts_file))['start_mtz'].values,
        subselect_thresholds_granularity=0.2
    )
