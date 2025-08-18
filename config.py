from pathlib import Path
from typing import Generator

# This is contained in the patient's folder
BASE_PATH = Path('/data/home/webb/UNEEG_data_3')
DATA_CLEANING_LOG_FOLDER = BASE_PATH / 'data_cleaning_logs'

PROBLEMATIC_EDFS_FOLDER = DATA_CLEANING_LOG_FOLDER / 'problematic_edf_files'
PROBLEMATIC_EDFS_FILE = PROBLEMATIC_EDFS_FOLDER / 'problematic_edf_files.csv'
REMAINING_DUPLICATES_FILE = DATA_CLEANING_LOG_FOLDER / 'remaining_duplicates.txt'

SEIZURE_ANNOTATIONS_FOLDER_NAME = Path('seizure_annotations')
SEIZURE_ANNOTATIONS_FILE_NAME = Path('seizure_annotations.csv')
EDF_DATA_FOLDER_NAME = Path('edf_data')

# dataset folders
FOR_MAYO_DIR = BASE_PATH / '20240201_UNEEG_ForMayo'
UNEEG_EXTENDED_DIR = BASE_PATH / '20250217_UNEEG_Extended'
COMPETITION_DIR = BASE_PATH / '20250501_SUBQ_SeizurePredictionCompetition_2025final'


def get_patient_folders() -> Generator[Path, None, None]:
    """Generator for all patient folders"""
    for parent_folder in [FOR_MAYO_DIR, UNEEG_EXTENDED_DIR, COMPETITION_DIR]:
        for patient_folder in parent_folder.iterdir():
            if patient_folder.is_dir():
                yield patient_folder
