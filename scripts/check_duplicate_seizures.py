from pathlib import Path
from typing import List

import pandas as pd

from config.paths import PATHS


def check_file_for_duplicates(annotation_path: Path, column_names: List[str], patient: str):
    seizures = pd.read_csv(annotation_path)

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
        check_file_for_duplicates(patient_dir.valid_szr_starts_file, column_names=['start'], patient=patient_dir.name)


if __name__ == '__main__':
    check_duplicate_seizures()
