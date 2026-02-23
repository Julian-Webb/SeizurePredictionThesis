import logging
from pathlib import Path
from typing import Tuple, Iterable

import pandas as pd
from pandas import DataFrame, Timedelta

import config.intervals as intervals
from config.constants import MIN_VALID_SEIZURES_PER_PATIENT, MIN_RATIO_RECORDED_TO_BE_VALID
from config.paths import PATHS, PatientDir
from utils.io import save_dataframe_multiformat, pickle_path


def ptnt_valid_szrs(szrs: DataFrame) -> Tuple[DataFrame, DataFrame, dict]:
    """:return: valid_szrs, szrs, ptnt_szr_info"""
    # find the time difference of a seizure to the *previous* one
    if szrs.empty:
        return szrs, szrs, {'total_seizures': len(szrs), 'valid_seizures': 0, 'enough_valid_seizures': False}

    diff = szrs['start_mtz'].diff()

    min_diff = intervals.PREICTAL.exact_dur + intervals.INTERVENTION.exact_dur
    valid = diff > min_diff
    valid.iloc[0] = True  # the first seizure is always valid

    n_valid = valid.value_counts()[True]
    enough_valid_szrs = n_valid >= MIN_VALID_SEIZURES_PER_PATIENT

    valid_szrs = szrs[valid]
    szrs['valid'] = valid

    # noinspection PyTypeChecker
    return valid_szrs, szrs, {'total_seizures': len(szrs), 'valid_seizures': n_valid,
                              'enough_valid_seizures': enough_valid_szrs}


def find_lead_szrs(szrs: DataFrame):
    """Finds the lead seizures.
    :param szrs: DataFrame containing seizure starts"""
    # A seizure is lead if it's within more than a certain time (LEAD.exact_dur) after a previous szr.
    if szrs.empty:
        return szrs
    diff = szrs['start_mtz'].diff()
    lead = diff > intervals.LEAD.exact_dur
    lead.iloc[0] = True  # First szr is always lead
    szrs = szrs.copy()  # ensure we're not writing to a view
    szrs['lead'] = lead
    return szrs


def ptnt_timespan_info(edfs: DataFrame) -> dict[str, dict]:
    """
    Compute information about the recording timespans
    :param ptnt_dir:
    :return: exact information, human-readable information
    """
    if not edfs.empty:
        first_start = edfs.iloc[0]['start_mtz']
        last_end = edfs.iloc[-1]['end_mtz']
        timespan = last_end - first_start

        duration_recorded = edfs['duration'].sum()
        duration_not_recorded = timespan - duration_recorded
        ratio_recorded = duration_recorded / timespan
        valid_ratio_recorded = ratio_recorded >= MIN_RATIO_RECORDED_TO_BE_VALID
    else:
        first_start, last_end = None, None
        timespan = Timedelta(seconds=0)
        duration_recorded, duration_not_recorded = Timedelta(seconds=0), Timedelta(seconds=0)
        ratio_recorded = 0
        valid_ratio_recorded = False

    exact_info = {'recordings_start': first_start,
                  'recordings_end': last_end,
                  'timespan': timespan,
                  'duration_recorded': duration_recorded,
                  'duration_not_recorded': duration_not_recorded,
                  'ratio_recorded': ratio_recorded,
                  'valid_ratio_recorded': valid_ratio_recorded,
                  }

    readable_info = {'recordings_start': first_start.strftime('%Y-%m-%d') if first_start else '',
                     'recordings_end': last_end.strftime('%Y-%m-%d') if last_end else '',
                     'timespan [days]': timespan.days,
                     'duration_recorded [days]': duration_recorded.days,
                     'duration_not_recorded [days]': duration_not_recorded.days,
                     'ratio_recorded': f"{round(ratio_recorded * 100)} %",
                     'valid_ratio_recorded': valid_ratio_recorded,
                     }

    return {'exact': exact_info, 'readable': readable_info}


def move_ptnt_dir(pdir: Path):
    """Move a patient dir to the invalid patient dir."""
    invalid_dataset_dir = PATHS.invalid_patients_dir / pdir.parent.name
    new_ptnt_dir = invalid_dataset_dir / pdir.name
    if pdir != new_ptnt_dir:  # Check if it was already moved because of previous code execution
        invalid_dataset_dir.mkdir(parents=True, exist_ok=True)
        pdir.rename(invalid_dataset_dir / pdir.name)


def validate_patients(pdirs: Iterable[PatientDir], move_invalid_ptnt_dirs: bool) -> None:
    """Find valid seizures for all patients. Save the valid seizures info, and the patient timespan info to files."""
    # patients are grouped by dataset
    ptnt_infos = {'exact': {}, 'readable': {}}

    for pdir in pdirs:
        try:
            szrs = pd.read_pickle(pickle_path(pdir.all_szr_starts_file))
        except FileNotFoundError:
            logging.warning(f'All seizure starts file not found for {pdir.name}')
            szrs = DataFrame()

        valid_szrs, szrs, ptnt_szr_info = ptnt_valid_szrs(szrs)
        valid_szrs = find_lead_szrs(valid_szrs)

        if 'valid' in valid_szrs.columns:
            valid_szrs.drop(columns=['valid'], inplace=True)
        if not valid_szrs.empty:
            save_dataframe_multiformat(valid_szrs, pdir.valid_szr_starts_file)
        save_dataframe_multiformat(szrs, pdir.all_szr_starts_file)

        try:
            edfs = pd.read_pickle(pickle_path(pdir.edf_files_table))
        except FileNotFoundError:
            logging.warning(f'EDF files sheet not found for {pdir.name}')
            edfs = DataFrame()

        ptnt_time_info = ptnt_timespan_info(edfs)
        ptnt_valid = ptnt_szr_info['enough_valid_seizures'] and ptnt_time_info['exact']['valid_ratio_recorded']

        # Add the exact and readable patient info into the patient info dict
        dataset = pdir.parent.name
        for k in ptnt_infos.keys():
            ptnt_infos[k][(dataset, pdir.name)] = {'valid': ptnt_valid, **ptnt_szr_info, **ptnt_time_info[k]}

        if move_invalid_ptnt_dirs and not ptnt_valid:
            move_ptnt_dir(pdir)

    # Save patient infos
    PATHS.patient_info_dir.mkdir(parents=True, exist_ok=True)

    for k, ptnt_info in ptnt_infos.items():
        index = pd.MultiIndex.from_tuples(ptnt_info.keys(), names=['dataset', 'patient'])
        ptnt_info = DataFrame(ptnt_info.values(), index=index)
        ptnt_info.sort_values(by=['valid', 'dataset', 'patient'], inplace=True, ascending=[False, True, True])
        if k == 'readable':
            ptnt_info.to_csv(PATHS.patient_info_readable.with_suffix('.csv'))
        elif k == 'exact':
            save_dataframe_multiformat(ptnt_info, PATHS.patient_info_exact, csv_index=True)


if __name__ == '__main__':
    validate_patients(PATHS.patient_dirs(include_invalid_ptnts=True), move_invalid_ptnt_dirs=True)
