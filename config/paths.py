# This file represents the directory and file structure of the project.

from enum import Enum
from pathlib import Path
from typing import List, Optional


class Dataset(Enum):
    """The available datasets"""
    for_mayo = '20240201_UNEEG_ForMayo'
    uneeg_extended = '20250217_UNEEG_Extended'
    competition = '20250501_SUBQ_SeizurePredictionCompetition_2025final'


# Use type(Path()) to get the correct class based on the operating system
class PatientDir(type(Path())):
    def __new__(cls, *args, **kwargs):
        """Represents the directory and file structure of a patient"""
        dataset_arg = kwargs.pop('dataset', None)
        self = super().__new__(cls, *args, *kwargs)
        self.dataset = dataset_arg

        ### seizure annotations
        self.szr_anns_dir = Path(self, "seizure_annotations")
        self.szr_anns_original_dir = Path(self.szr_anns_dir, "seizure_annotations_original")
        self.combined_anns_file = Path(self.szr_anns_dir, "combined_annotations")
        self.all_szr_starts_file = Path(self.szr_anns_dir, "seizure_starts_all")
        self.valid_szr_starts_file = Path(self.szr_anns_dir, "seizure_starts_valid")

        ### edf data
        # The directory containing the original edf files for the competition dataset, before they're renamed
        self.original_edf_dir = Path(self, 'original_edf_data')
        # The directory containing the edf files
        self.edf_dir = Path(self, 'edf_data')
        # The name of the sheet containing the edf file names and their metadata for each patient
        self.edf_files_sheet = Path(self, 'edf_files')

        ### Preprocessing
        self.segments_table = Path(self, 'segments')
        self.segments_plot = Path(self, 'segments_plot.png')
        self.train_test_split = Path(self, 'train_test_split')

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
        self.data_cleaning_logs_dir = Path(self, "data_cleaning_logs")
        self.problematic_edfs_dir = Path(self.data_cleaning_logs_dir, 'problematic_edf_files')
        self.problematic_edfs_file = Path(self.problematic_edfs_dir / 'problematic_edf_files.csv')
        self.remaining_duplicates_file = Path(self.data_cleaning_logs_dir / 'remaining_duplicates.txt')

        # preprocessing
        self.patient_info_dir = Path(self, "patient_info")
        self.patient_info_exact = Path(self.patient_info_dir, "patient_info_exact")
        self.patient_info_readable = Path(self.patient_info_dir, "patient_info_readable")

        self.invalid_patients_dir = Path(self, "invalid_patients")
        return self

    def patient_dirs(self, datasets: Optional[List[Dataset]] = None, include_invalid_ptnts: bool = False) -> List[
        PatientDir]:
        """
        Return a list of patient directories of the given datasets
        :param datasets: The datasets to get patient dirs for (default: all)
        :param include_invalid_ptnts: Whether to include invalid patient dirs
        :returns: ptnt_dirs - a list of PatientDir objects
        """
        if datasets is None:
            datasets = list(Dataset)

        base_dirs = [self.base_dir]
        if include_invalid_ptnts:
            base_dirs.append(self.invalid_patients_dir)

        ptnt_dirs = []
        for base_dir in base_dirs:
            for dataset in datasets:
                dataset_path = base_dir / dataset.value
                if dataset_path.is_dir():
                    for ptnt_dir in dataset_path.iterdir():
                        if ptnt_dir.is_dir():
                            ptnt_dirs.append(PatientDir(ptnt_dir, dataset=dataset))

        return ptnt_dirs

    @property
    def base_dir(self) -> Path:
        """:return: The base directory where the data is stored"""
        # This is just an alias
        return self


# Change base path here
PATHS = Paths('/data/home/webb/UNEEG_data')
# PATHS = Paths('/Users/julian/Developer/SeizurePredictionData')
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^

if __name__ == '__main__':
    for ptnt_dir in Paths('/data/home/webb/UNEEG_data').patient_dirs(include_invalid_ptnts=True):
        print(ptnt_dir)
