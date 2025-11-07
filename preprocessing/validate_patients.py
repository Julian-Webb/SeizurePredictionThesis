from pathlib import Path
from typing import Tuple, Iterable

import pandas as pd

from config.constants import MIN_VALID_SEIZURES_PER_PATIENT
import config.intervals as intervals
from config.paths import PATHS, PatientDir, Dataset


def _validate_patient(patient_dir: PatientDir) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """:return: valid_szrs, seizures, patient_info"""
    szrs = pd.read_csv(patient_dir.all_szr_starts_file, parse_dates=['start'], index_col=0)

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


def move_ptnt_dir(ptnt_dir: Path):
    """Move a patient dir to the invalid patient dir."""
    invalid_dataset_dir = PATHS.invalid_patients_dir / ptnt_dir.parent.name
    invalid_dataset_dir.mkdir(parents=True, exist_ok=True)
    ptnt_dir.rename(invalid_dataset_dir / ptnt_dir.name)


def validate_patients(move_patient_dirs: bool, ptnt_dirs: Iterable[PatientDir]) -> None:
    """Find valid seizures for all patients. Save the valid seizures and the patient info to files."""
    # patients are grouped by dataset
    ptnts = {}

    for ptnt_dir in ptnt_dirs:
        valid_szrs, szrs, ptnt_info = _validate_patient(ptnt_dir)
        valid_szrs.to_csv(ptnt_dir.valid_szr_starts_file)
        szrs.to_csv(ptnt_dir.all_szr_starts_file)

        dataset = ptnt_dir.parent.name
        ptnts[(dataset, ptnt_dir.name)] = ptnt_info

        if move_patient_dirs and not ptnt_info['valid']:
            move_ptnt_dir(ptnt_dir)

    index = pd.MultiIndex.from_tuples(ptnts.keys(), names=['dataset', 'patient'])
    ptnts = pd.DataFrame(ptnts.values(), index=index)
    ptnts.sort_index(inplace=True)
    ptnts.to_csv(PATHS.patient_info_file)


if __name__ == '__main__':
    validate_patients(move_patient_dirs=True, ptnt_dirs=PATHS.patient_dirs(Dataset.for_mayo))
