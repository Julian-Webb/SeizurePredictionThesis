# This file represents the directory and file structure of the project.

from enum import Enum
from pathlib import Path
from typing import List


class Dataset(Enum):
    """The available datasets"""
    for_mayo = '20240201_UNEEG_ForMayo'
    uneeg_extended = '20250217_UNEEG_Extended'
    competition = '20250501_SUBQ_SeizurePredictionCompetition_2025final'


# Use type(Path()) to get the correct class based on the operating system
class PatientDir(type(Path())):
    def __new__(cls, *args, **kwargs):
        """Represents the directory and file structure of a patient"""
        self = super().__new__(cls, *args, *kwargs)

        ### seizure annotations
        self.szr_anns_dir = self / "seizure_annotations"
        self.szr_anns_original_dir = self.szr_anns_dir / "seizure_annotations_original"
        self.combined_anns_file = self.szr_anns_dir / "combined_annotations.csv"
        self.all_szr_starts_file = self.szr_anns_dir / "seizure_starts_all.csv"
        self.valid_szr_starts_file = self.szr_anns_dir / "seizure_starts_valid.csv"

        ### edf data
        # The directory containing the original edf files for the competition dataset, before they're renamed
        self.original_edf_dir = self / 'original_edf_data'
        # The directory containing the edf files
        self.edf_dir = self / 'edf_data'
        # The name of the sheet containing the edf file names and their metadata for each patient
        self.edf_files_sheet = self / 'edf_files.csv'

        ### other
        # todo delete next two lines
        self.preictal_windows_file = self / 'preictal_windows.csv'
        self.interictal_windows_file = self / 'interictal_windows.csv'

        self.segments_table = self / 'segments.csv'

        return self


class Paths(type(Path())):
    def __new__(cls, *args, **kwargs):
        """Represents the directory and file structure of the project.
        :param *args: The base_dir: str | PathLike[str]
        """
        self = super().__new__(cls, *args, **kwargs)

        # dataset dirs
        self.dataset_dirs = {dataset: Path(self, dataset.value) for dataset in Dataset}
        self.for_mayo_dir = self.dataset_dirs[Dataset.for_mayo]
        self.uneeg_extended_dir = self.dataset_dirs[Dataset.uneeg_extended]
        self.competition_dir = self.dataset_dirs[Dataset.competition]

        # data cleaning logs
        self.data_cleaning_logs_dir = self / "data_cleaning_logs"
        self.problematic_edfs_dir = self.data_cleaning_logs_dir / 'problematic_edf_files'
        self.problematic_edfs_file = self.problematic_edfs_dir / 'problematic_edf_files.csv'
        self.remaining_duplicates_file = self.data_cleaning_logs_dir / 'remaining_duplicates.txt'

        # preprocessing
        self.patient_info_file = self / "patient_info.csv"
        self.invalid_patients_dir = self / "invalid_patients"

        return self

    def patient_dirs(self, *args: List[Dataset]) -> List[PatientDir]:
        """Generator for patient folders of specified datasets (default: all)
        :param args: The datasets to get patient dirs for
        :returns: patient_dirs - a list of PatientDir objects"""
        if not args:
            datasets = list(Dataset)
        else:
            datasets = args

        patient_dirs = []
        for dataset in datasets:
            for patient_dir in self.dataset_dirs[dataset].iterdir():
                if patient_dir.is_dir():
                    patient_dirs.append(PatientDir(patient_dir))
        return patient_dirs

    @property
    def base_dir(self) -> Path:
        """:return: The base directory where the data is stored"""
        # This is just an alias
        return self


# Change base path here
# PATHS = Paths('/data/home/webb/UNEEG_data')
PATHS = Paths('/Users/julian/Developer/SeizurePredictionData')
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^

if __name__ == '__main__':
    for ptnt_dir in Paths('/data/home/webb/UNEEG_data').patient_dirs():
        print(ptnt_dir)
