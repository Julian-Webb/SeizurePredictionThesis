from pathlib import Path
from typing import Tuple, Iterable

import pandas as pd
from pandas import DataFrame

import config.intervals as intervals
from config.constants import MIN_VALID_SEIZURES_PER_PATIENT
from config.paths import PATHS, PatientDir


def validate_patient(szrs: DataFrame) -> Tuple[DataFrame, DataFrame, dict]:
    """:return: valid_szrs, szrs, ptnt_szr_info"""
    # find the time difference of a seizure to the *previous* one
    diff = szrs['start'].diff()

    min_diff = intervals.PREICTAL.exact_dur + intervals.HORIZON.exact_dur
    valid = diff > min_diff
    valid.iloc[0] = True  # the first seizure is always valid

    n_valid = valid.value_counts()[True]
    valid_ptnt = n_valid >= MIN_VALID_SEIZURES_PER_PATIENT

    valid_szrs = szrs[valid]
    szrs['valid'] = valid

    # noinspection PyTypeChecker
    return valid_szrs, szrs, {'total_seizures': len(szrs), 'valid_seizures': n_valid, 'valid': valid_ptnt}


def find_lead_szrs(szrs: DataFrame):
    """Finds the lead seizures.
    :param szrs: DataFrame containing seizure starts"""
    # A seizure is lead if it's within more than a certain time (LEAD.exact_dur) after a previous szr.
    diff = szrs['start'].diff()
    lead = diff > intervals.LEAD.exact_dur
    lead.iloc[0] = True  # First szr is always lead
    szrs = szrs.copy()  # ensure we're not writing to a view
    szrs['lead'] = lead
    return szrs


def ptnt_timespan_info(ptnt_dir: PatientDir) -> dict[str, dict]:
    """
    Compute information about the recording timespans
    :param ptnt_dir:
    :return: exact information, human-readable information
    """
    edfs = pd.read_csv(ptnt_dir.edf_files_sheet, parse_dates=['start', 'end'],
                       converters={'duration_hours': pd.to_timedelta})
    first_start = edfs.iloc[0]['start']
    last_end = edfs.iloc[-1]['end']
    timespan = last_end - first_start

    duration_recorded = edfs['duration_hours'].sum()
    duration_not_recorded = timespan - duration_recorded
    ratio_recorded = duration_recorded / timespan

    exact_info = {'recordings_start': first_start,
                  'recordings_end': last_end,
                  'timespan': timespan,
                  'duration_recorded': duration_recorded,
                  'duration_not_recorded': duration_not_recorded,
                  'ratio_recorded': ratio_recorded,
                  }

    readable_info = {'recordings_start': first_start.strftime('%Y-%m-%d'),
                     'recordings_end': last_end.strftime('%Y-%m-%d'),
                     'timespan': f"{timespan.days} days",
                     'duration_recorded': f"{duration_recorded.days} days",
                     'duration_not_recorded': f"{duration_not_recorded.days} days",
                     'ratio_recorded': f"{round(ratio_recorded * 100)} %",
                     }

    return {'exact': exact_info, 'readable': readable_info}


def move_ptnt_dir(ptnt_dir: Path):
    """Move a patient dir to the invalid patient dir."""
    invalid_dataset_dir = PATHS.invalid_patients_dir / ptnt_dir.parent.name
    new_ptnt_dir = invalid_dataset_dir / ptnt_dir.name
    if ptnt_dir != new_ptnt_dir:  # Check if it was already moved because of previous code execution
        invalid_dataset_dir.mkdir(parents=True, exist_ok=True)
        ptnt_dir.rename(invalid_dataset_dir / ptnt_dir.name)


def validate_patients(ptnt_dirs: Iterable[PatientDir], move_invalid_ptnt_dirs: bool) -> None:
    """Find valid seizures for all patients. Save the valid seizures info, and the patient timespan info to files."""
    # patients are grouped by dataset
    ptnt_infos = {'exact': {}, 'readable': {}}

    for ptnt_dir in ptnt_dirs:
        szrs = pd.read_csv(ptnt_dir.all_szr_starts_file, parse_dates=['start'], index_col=0)
        valid_szrs, szrs, ptnt_szr_info = validate_patient(szrs)

        valid_szrs = find_lead_szrs(valid_szrs)
        if 'valid' in valid_szrs.columns:
            valid_szrs.drop(columns=['valid'], inplace=True)

        valid_szrs.to_csv(ptnt_dir.valid_szr_starts_file)
        szrs.to_csv(ptnt_dir.all_szr_starts_file)

        time_info = ptnt_timespan_info(ptnt_dir)
        dataset = ptnt_dir.parent.name

        # Add the exact and readable patient info into the patient info dict
        for k in ptnt_infos.keys():
            ptnt_infos[k][(dataset, ptnt_dir.name)] = {**ptnt_szr_info, **time_info[k]}

        if move_invalid_ptnt_dirs and not ptnt_szr_info['valid']:
            move_ptnt_dir(ptnt_dir)

    # Save patient infos
    PATHS.patient_info_dir.mkdir(parents=True, exist_ok=True)

    for k, ptnt_info in ptnt_infos.items():
        index = pd.MultiIndex.from_tuples(ptnt_info.keys(), names=['dataset', 'patient'])
        ptnt_info = DataFrame(ptnt_info.values(), index=index)
        ptnt_info.sort_values(by=['valid', 'dataset', 'patient'], inplace=True, ascending=[False, True, True])
        if k == 'readable':
            ptnt_info.to_csv(PATHS.patient_info_readable_csv)
        elif k == 'exact':
            ptnt_info.to_csv(PATHS.patient_info_exact_csv)
            ptnt_info.to_pickle(PATHS.patient_info_exact_pkl)



if __name__ == '__main__':
    validate_patients(PATHS.patient_dirs(include_invalid_ptnts=True), move_invalid_ptnt_dirs=True)
