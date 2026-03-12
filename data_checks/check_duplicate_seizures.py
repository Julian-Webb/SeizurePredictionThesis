from pathlib import Path
from typing import List

import pandas as pd

from config import PATHS, PatientDir
from config.paths import pickle_path


def check_file_for_duplicates(ann_path_pkl: Path, column_names: List[str], patient: str):
    seizures = pd.read_pickle(ann_path_pkl)

    # check for duplicates
    seizures.sort_values(column_names, inplace=True)  # make sure it's sorted
    duplicates = seizures[seizures.duplicated(column_names, keep=False)]
    if duplicates.shape[0] > 0:
        print(f"Duplicates found in {patient}:")
        print(duplicates[column_names])
    else:
        print(f"No duplicates found in {patient}")


# find duplicate seizures
def check_duplicate_seizures():
    """Check seizure annotation files for duplicate seizures."""
    for patient_dir in PATHS.patient_dirs():
        check_file_for_duplicates(pickle_path(patient_dir.valid_szr_starts_file), column_names=['start'],
                                  patient=patient_dir.name)


def check_all_szrs():
    pdirs = [pdir for pdir in PATHS.patient_dirs() if not 'fake' in pdir.name.lower()]
    szrs_per_ptnt = {pdir.name : pd.read_pickle(pickle_path(pdir.all_szr_starts_file)) for pdir in pdirs}

    szrs = pd.concat(szrs_per_ptnt, names=['patient', 'seizure'])

    dups = szrs[szrs['start_mtz'].duplicated(keep=False)]

    if dups.empty:
        print('No duplicates found')
    else:
        print('DUPLICATES FOUND:')
        print(dups)

    return





if __name__ == '__main__':
    check_all_szrs()
