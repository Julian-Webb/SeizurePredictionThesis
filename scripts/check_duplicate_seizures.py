from pathlib import Path

import pandas as pd

from config import PATHS
from utils.paths import Dataset

PATIENT_ANNOTATION_FILES = {
    'B52K3P3G': 'B52K3P3G_CONSENSUS_corrected.csv',
    'E85L95P2H': 'E85L95P2H.csv',
    'G39B4L9E': 'G39B4L9E.csv',
    'K37N36L4D': 'K37N36L4D.csv',
    'M39K4B3C': 'M39K4B3C.csv',
    'P73M2F6H': 'P73M2F6H.csv',
    'A4RW34Z5B': 'combined_annotations.csv',
    'D63Q51K2N': 'combined_annotations.csv',
    'E15T65H3Z': 'combined_annotations.csv',
    'F5TW95P3X': 'combined_annotations.csv',
    'K53T36N7F': 'combined_annotations.csv',
    'L3GS57K2T': 'combined_annotations.csv',
    'P4Hk23M7L': 'combined_annotations.csv',
    'P1': 'seizure_starts.csv',
    'P2': 'seizure_starts.csv',
    'P3': 'seizure_starts.csv',
}


def check_file_for_duplicates(annotation_path: Path, column_name: str, patient: str):
    seizures = pd.read_csv(annotation_path)

    # check for duplicates
    seizures.sort_values(column_name, inplace=True)  # make sure it's sorted
    duplicates = seizures[seizures.duplicated(column_name, keep=False)]
    if duplicates.shape[0] > 0:
        print(f"Duplicates found in {patient}:")
        print(duplicates[column_name])
    else:
        print(f"No duplicates found in {patient}")


# find duplicate seizures
def check_duplicate_seizures():
    """Check seizure annotation files for duplicate seizures."""
    for patient_dir in PATHS.patient_dirs(Dataset.for_mayo):
        check_file_for_duplicates(
            patient_dir.szr_anns_original_dir / PATIENT_ANNOTATION_FILES[patient_dir.name],
            column_name='single_marker',
            patient=patient_dir.name,
        )

    for patient_dir in PATHS.patient_dirs(Dataset.uneeg_extended):
        check_file_for_duplicates(
            patient_dir.szr_anns_dir / PATIENT_ANNOTATION_FILES[patient_dir.name],
            column_name='single_marker',
            patient=patient_dir.name,
        )

    for patient_dir in PATHS.patient_dirs(Dataset.competition):
        check_file_for_duplicates(
            patient_dir.szr_anns_dir / PATIENT_ANNOTATION_FILES[patient_dir.name],
            column_name='start',
            patient=patient_dir.name,
        )

if __name__ == '__main__':
    check_duplicate_seizures()

