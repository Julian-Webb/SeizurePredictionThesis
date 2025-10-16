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
        self = super().__new__(cls, *args, *kwargs)

        ### seizure annotations
        self.seizure_annotations_dir = self / "seizure_annotations"
        self.seizure_annotations_original_dir = self.seizure_annotations_dir / "seizure_annotations_original"
        self.combined_annotations_file = self.seizure_annotations_dir / "combined_annotations.csv"
        self.seizure_starts_file = self.seizure_annotations_dir / "seizure_starts.csv"

        ### edf data
        # The directory containing the original edf files for the competition dataset, before they're renamed
        self.original_edf_dir = self / 'original_edf_data'
        # The directory containing the edf files
        self.edf_dir = self / 'edf_data'
        # The name of the sheet containing the edf file names and their metadata for each patient
        self.edf_files_sheet = self / 'edf_files.csv'

        return self


class Paths(type(Path())):
    def __new__(cls, *args, **kwargs):
        """Represents the directory and file structure of the project.
        :param *args: The base_dir: str | PathLike[str]
        """
        self = super().__new__(cls, *args, **kwargs)

        # data cleaning logs
        self.data_cleaning_logs_dir = self / "data_cleaning_logs"
        self.problematic_edfs_dir = self.data_cleaning_logs_dir / 'problematic_edf_files'
        self.problematic_edfs_file = self.problematic_edfs_dir / 'problematic_edf_files.csv'
        self.remaining_duplicates_file = self.data_cleaning_logs_dir / 'remaining_duplicates.txt'

        # dataset dirs
        self.dataset_dirs = {dataset: self / dataset.value for dataset in Dataset}
        self.for_mayo_dir = self.dataset_dirs[Dataset.for_mayo]
        self.uneeg_extended_dir = self.dataset_dirs[Dataset.uneeg_extended]
        self.competition_dir = self.dataset_dirs[Dataset.competition]

        return self

    def patient_dirs(self, *args: List[Dataset]) -> List[PatientDir]:
        """Generator for patient folders of specified datasets (default: all)
        :*args: The datasets to get patient dirs for"""
        if not args:
            datasets = list(Dataset)
        else:
            datasets = args
        dataset_dirs = [self.dataset_dirs[dataset] for dataset in datasets]

        patient_dirs = []
        for dataset_dir in dataset_dirs:
            for patient_dir in dataset_dir.iterdir():
                if patient_dir.is_dir():
                    patient_dirs.append(PatientDir(patient_dir))
        return patient_dirs

    @property
    def base_dir(self) -> Path:
        # This is just an alias
        return self
