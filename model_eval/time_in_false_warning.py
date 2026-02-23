import logging
from typing import Iterable, Optional

import pandas as pd
from pandas import DataFrame, Timedelta
from config.intervals import Interval, INTERVENTION, SPH
from config.paths import PATHS
from utils.io import pickle_path


# todo try this for all patients with minimum duration missing super low (minus 1 year or something)
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


def time_in_false_warning(
        clips: DataFrame,
        edfs: DataFrame,
        prediction_threshold: float,
        models: Iterable[str] = ("CNN", "ensemble"),
        intervention_iv: Interval = INTERVENTION,
        sph_iv: Interval = SPH,
):
    """
    Takes a set of predictions and calculates the time in false warning (TIFW).
    Seizure predictions are issued at the warning time (WT). After that, there's the intervention interval/time (IT),
    followed by the seizure prediction horizon (SPH) in which the seizure should occur. If no seizure occurs within the
    SPH, the SPH is added to the TIFW.

    The SPH is, in a way, the preictal interval, when looking forwards in time from the prediction, instead of backwards
    from the seizure. Thus, the SPH has the same duration as the preictal interval.

    This function works as follows:
    1. From the clip predictions, the WTs are determined as the end time of the clips.
    2. The SPHs are calculated from the WTs. The SPHs will (almost certainly) be overlapping.
    3. Intervals with missing EEG data will be subtracted from the SPHs because we don't know if a seizure occurred
       during them
    4. For each SPH, it is checked whether it is correct, i.e., whether a seizure starts within it.
    5. The false SPHs will be merged to make them non-overlapping.
    6. The duration of the remaining SPHs is the absolute TIFW.
    7. The absolute TIFW / total duration time is the relative TIFW.

    :param clips: DataFrame with columns '{model}_probability', 'end_time', 'valid' (optional)
    :param edfs: The patient's EDF DataFrame
    :return:
    """
    # todo Regarding step 6:
    #  should total duration time be the time of the clips / valid intervals / total recording time?
    #  Because we're filtering out artifacts (-> valid intervals), but then the clips sometimes still contain artifacts
    #  (invalid intervals), but not always...
    clips = clips.copy()

    if 'valid' in clips.columns:
        clips = clips[clips['valid']]
    else:
        logging.warning("No 'valid' column in clips DataFrame. Assuming all clips are valid.")

    for model in models:
        # todo greater than or geq to threshold?
        #### Warning Times
        y_pred = clips[f'{model}_probability'] > prediction_threshold
        warn_times = clips.loc[y_pred, 'end_time']

        #### Seizure prediction horizons
        sphs = DataFrame(index=warn_times.index)
        sphs['start'] = warn_times + intervention_iv.exact_dur
        sphs['end'] = sphs['start'] + sph_iv.exact_dur

        #### Subtract missing EEG intervals
        missing_ivs = missing_recording_intervals(
            edfs,
            minimum_duration_missing=Timedelta(seconds=1),
            last_interval_duration=2 * (intervention_iv.exact_dur + sph_iv.exact_dur)
        )

        

        ...  # todo delete


if __name__ == '__main__':
    pdir = PATHS.patient_dirs()[0]
    time_in_false_warning(
        pd.read_pickle(pickle_path(pdir.clips_table)),
        pd.read_pickle(pickle_path(pdir.edf_files_table)),
        0.5,
    )
