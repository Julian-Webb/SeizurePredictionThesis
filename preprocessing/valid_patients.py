# Find the valid participants.
# They should have at least 10 seizures. To make sure seizures are distinct, a successive seizure should be at least
# 1 hour after the previous seizure.
import pandas as pd

from config import PATHS, Durations, Constants
from utils.paths import PatientDir


def _validate_patient(patient_dir: PatientDir):
    """:return: valid_szrs, seizures, patient_info"""
    szrs = pd.read_csv(patient_dir.all_szr_starts_file, parse_dates=['start'], index_col=0)

    # find the time difference of a seizure to the *previous* one
    diff = szrs['start'].diff()

    valid = diff >= Durations.PREICTAL_INTERVAL
    valid.iloc[0] = True  # the first seizure is always valid

    n_valid = valid.value_counts()[True]
    valid_ptnt = n_valid >= Constants.MIN_VALID_SEIZURES_PER_PATIENT

    valid_szrs = szrs[valid]
    szrs['valid'] = valid

    return valid_szrs, szrs, {'total_seizures': len(szrs), 'valid_seizures': n_valid, 'valid': valid_ptnt,
                              # 'dir': str(patient_dir)
                              }


def valid_patients():
    """Find valid seizures for all patients. Save the valid seizures and the patient info to files."""
    # patients are grouped by dataset
    patients = {}

    for patient_dir in PATHS.patient_dirs():
        valid_szrs, szrs, ptnt_info = _validate_patient(patient_dir)
        valid_szrs.to_csv(patient_dir.valid_szr_starts_file)
        szrs.to_csv(patient_dir.all_szr_starts_file)

        dataset = patient_dir.parent.name
        patients[(dataset, patient_dir.name)] = ptnt_info

    index = pd.MultiIndex.from_tuples(patients.keys(), names=['dataset', 'patient'])
    patients = pd.DataFrame(patients.values(), index=index)
    patients.sort_index(inplace=True)
    patients.to_csv(PATHS.patient_info_file)


if __name__ == '__main__':
    valid_patients()
