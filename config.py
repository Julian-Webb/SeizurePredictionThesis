from pathlib import Path
from typing import List


class Paths:
    BASE_DIR = Path('/data/home/webb/UNEEG_data')
    DATA_CLEANING_LOG_DIR = BASE_DIR / 'data_cleaning_logs'

    PROBLEMATIC_EDFS_DIR = DATA_CLEANING_LOG_DIR / 'problematic_edf_files'
    PROBLEMATIC_EDFS_FILE = PROBLEMATIC_EDFS_DIR / 'problematic_edf_files.csv'
    REMAINING_DUPLICATES_FILE = DATA_CLEANING_LOG_DIR / 'remaining_duplicates.txt'

    # dataset folders
    FOR_MAYO_DIR = BASE_DIR / '20240201_UNEEG_ForMayo'
    UNEEG_EXTENDED_DIR = BASE_DIR / '20250217_UNEEG_Extended'
    COMPETITION_DIR = BASE_DIR / '20250501_SUBQ_SeizurePredictionCompetition_2025final'

    @staticmethod
    def seizure_annotations_dir(patient_dir: Path):
        return patient_dir / 'seizure_annotations'

    @staticmethod
    def seizure_annotations_file(patient_dir: Path):
        return Paths.seizure_annotations_dir(patient_dir) / 'seizure_annotations.csv'

    @staticmethod
    def original_edf_dir(patient_dir: Path):
        """The directory containing the original edf files for the competition dataset, before they're renamed"""
        return patient_dir / 'original_edf_data'

    @staticmethod
    def edf_dir(patient_dir: Path):
        """The directory containing the edf files"""
        return patient_dir / 'edf_data'

    @staticmethod
    def edf_files_sheet(patient_dir: Path):
        """The name of the sheet containing the edf file names and their metadata for each patient"""
        return patient_dir / 'edf_files.csv'

    @staticmethod
    def patient_dirs() -> List[Path]:
        """Generator for all patient folders"""
        patient_dirs = []
        for parent_dir in [Paths.FOR_MAYO_DIR, Paths.UNEEG_EXTENDED_DIR, Paths.COMPETITION_DIR]:
            for patient_dir in parent_dir.iterdir():
                if patient_dir.is_dir():
                    patient_dirs.append(patient_dir)
        return patient_dirs


class Constants:
    _SECS_PER_MIN = 60
    # The length of the clips
    CLIP_LENGTH_SEC = 10 * _SECS_PER_MIN
    # The length of the segments which the clips are split into
    SEGMENT_LENGTH_SEC = 15
    # How much time before a seizure onset counts as preictal
    PREICTAL_INTERVAL_SEC = [-65 * _SECS_PER_MIN, -5 * _SECS_PER_MIN]
