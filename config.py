from pathlib import Path
from typing import Generator

# This is contained in the patient's folder
BASE_DIR = Path('/data/home/webb/UNEEG_data_4')
DATA_CLEANING_LOG_FOLDER = BASE_DIR / 'data_cleaning_logs'

PROBLEMATIC_EDFS_FOLDER = DATA_CLEANING_LOG_FOLDER / 'problematic_edf_files'
PROBLEMATIC_EDFS_FILE = PROBLEMATIC_EDFS_FOLDER / 'problematic_edf_files.csv'
REMAINING_DUPLICATES_FILE = DATA_CLEANING_LOG_FOLDER / 'remaining_duplicates.txt'

SEIZURE_ANNOTATIONS_FOLDER_NAME = Path('seizure_annotations')
SEIZURE_ANNOTATIONS_FILE_NAME = Path('seizure_annotations.csv')
# The directory containing the original edf files for the competition dataset, before they're renamed
ORIGINAL_EDF_DIR_NAME = Path('original_edf_data')
EDF_DIR_NAME = Path('edf_data')  # The directory containing the edf files
# The name of the sheet containing the edf file names and their metadata for each patient
EDF_FILES_SHEET_NAME = Path('edf_files.csv')

# dataset folders
FOR_MAYO_DIR = BASE_DIR / '20240201_UNEEG_ForMayo'
UNEEG_EXTENDED_DIR = BASE_DIR / '20250217_UNEEG_Extended'
COMPETITION_DIR = BASE_DIR / '20250501_SUBQ_SeizurePredictionCompetition_2025final'


def get_patient_dirs() -> Generator[Path, None, None]:
    """Generator for all patient folders"""
    for parent_folder in [FOR_MAYO_DIR, UNEEG_EXTENDED_DIR, COMPETITION_DIR]:
        for patient_folder in parent_folder.iterdir():
            if patient_folder.is_dir():
                yield patient_folder
