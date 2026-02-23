# This file represents the directory and file structure of the project.

from enum import Enum
from pathlib import Path
from typing import List, Optional


class Dataset(Enum):
    """The available datasets"""
    competition = 'competition'
    ultra2 = 'ultra2'


# Use type(Path()) to get the correct class based on the operating system
class PatientDir(type(Path())):
    def __new__(cls, *args, **kwargs):
        """Represents the directory and file structure of a patient"""
        dataset_arg = kwargs.pop('dataset', None)
        self = super().__new__(cls, *args, *kwargs)

        ### Info
        self.dataset = dataset_arg

        ### seizure annotations
        self.szr_anns_dir = Path(self, "seizure_annotations")
        self.szr_anns_original_dir = Path(self.szr_anns_dir, "original")
        self.szr_starts_naive_file = Path(self.szr_anns_original_dir, "seizure_starts_naive")
        self.all_szr_starts_file = Path(self.szr_anns_dir, "seizure_starts_all")
        self.valid_szr_starts_file = Path(self.szr_anns_dir, "seizure_starts_valid")

        ### edf data
        # The directory containing the original edf files for the competition dataset, before they're renamed
        self.original_edf_dir = Path(self, 'original_edf_data')
        # The directory containing the edf files
        self.edf_dir = Path(self, 'edf_data')
        # The name of the table containing the edf file names and their metadata for each patient
        self.edf_files_table = Path(self, 'edf_files')
        self.valid_edf_intervals = Path(self, 'edf_intervals_valid')
        self.invalid_edf_intervals = Path(self, 'edf_intervals_invalid')

        ### Preprocessing
        self.segments_table = Path(self, 'segments')
        self.segments_plot = Path(self, 'segments_plot.png')
        self.train_test_split = Path(self, 'train_test_split')

        ### ML models
        self.models_dir = Path(self, 'models')
        self.ensemble_model = Path(self.models_dir, 'ensemble.keras')
        self.feature_scaler = Path(self.models_dir, 'feature_scaler.pkl')
        self.cnn_model = Path(self.models_dir, 'CNN.keras')
        self.cnn_history = Path(self.models_dir, 'CNN_training_history.csv')

        ### Predictions
        # todo change this to be one sheet per patient (segment_probabilities) which includes probabilities for all models (see calc_segment_probabilities.py and clips.py)
        self.predictions_dir = Path(self, 'predictions')
        self.segment_probabilities_table = self.predictions_dir / 'segment_probabilities'
        self.clips_table = self.predictions_dir / 'clips'

        # Model Evaluation
        self.model_eval_dir = Path(self, 'model_evaluation')

        return self


class Paths(type(Path())):
    def __new__(cls, *args, **kwargs):
        """Represents the directory and file structure of the project.
        :param *args: The base_dir: str | PathLike[str]
        """
        self = super().__new__(cls, *args, **kwargs)

        # dataset dirs
        self.datasets_dir = Path(self, "datasets")  # The dir that contains the datasets
        self.ultra2_dir = Path(self.datasets_dir, Dataset.ultra2.value)
        self.competition_dir = Path(self.datasets_dir, Dataset.competition.value)

        # data cleaning logs
        self.data_cleaning_logs_dir = Path(self, "data_cleaning_logs")
        self.problematic_edfs_dir = Path(self.data_cleaning_logs_dir, 'problematic_edfs')
        self.remaining_duplicates_file = Path(self.data_cleaning_logs_dir / 'remaining_duplicates.txt')

        # preprocessing
        self.patient_info_dir = Path(self, "patient_info")
        self.basic_patient_info = Path(self.patient_info_dir, "basic_patient_info.xlsx")
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

        base_dirs = [self.datasets_dir]
        if include_invalid_ptnts:
            base_dirs.append(self.invalid_patients_dir)

        ptnt_dirs = []
        for base_dir in base_dirs:
            for dataset in datasets:
                dataset_path = base_dir / dataset.value
                if dataset_path.is_dir():
                    for ptnt_dir in sorted(dataset_path.iterdir()):
                        if ptnt_dir.is_dir():
                            ptnt_dirs.append(PatientDir(ptnt_dir, dataset=dataset))

        return ptnt_dirs

    @property
    def root(self) -> Path:
        """:return: The root directory of the dataset"""
        # This is just an alias
        return Path(self)


# Change base path here
PATHS = Paths('/data/home/webb/UNEEG')
# PATHS = Paths('/Users/julian/Developer/SeizurePredictionData')
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^

if __name__ == '__main__':
    for ptnt_dir in PATHS.patient_dirs(include_invalid_ptnts=True):
        print(ptnt_dir)

