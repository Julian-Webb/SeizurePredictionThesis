"""
This file represents the directory and file structure of the project.
Uses type(Path()) to get the correct class based on the operating system
"""
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from pandas import DataFrame, Series


class Dataset(Enum):
    """The available datasets"""
    competition = 'competition'
    ultra2 = 'ultra2'


def pickle_path(path: Path):
    path = path.with_suffix('.pkl')
    return path.with_name('.' + path.name)  # Hide it with the .


class MultiPath(type(Path())):
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, *kwargs)

        p = Path(self)
        self.pickle = pickle_path(p)
        self.pickle_visible = p.with_suffix('.pkl')
        self.csv = p.with_suffix('.csv')
        self.ods = p.with_suffix('.ods')
        self.xlsx = p.with_suffix('.xlsx')
        return self


def save_dataframe_multiformat(
        df: Union[DataFrame, Series],
        path: MultiPath,
        formats: tuple[str, ...] = ('csv', 'pickle'),
        save_index: bool = False,
        make_parent_dir: bool = True,
        float_format: str = None,
):
    """
    Save a pd.DataFrame in multiple formats (csv, pickle).
    :param formats: The formats to save the DataFrame in. The names must correspond to the properties of the ``MultiPath`` class.
    :param save_index: Whether to save the index of the DataFrame for formats that support not saving it.
    """
    if make_parent_dir:
        path.parent.mkdir(parents=True, exist_ok=True)

    for f in formats:
        match f:
            case 'pickle':
                df.to_pickle(path.pickle)
            case 'pickle_visible':
                df.to_pickle(path.pickle_visible)
            case 'csv':
                df.to_csv(path.csv, index=save_index, float_format=float_format)
            case 'ods':
                df.to_excel(path.ods, index=save_index, float_format=float_format)
            case 'xlsx':
                df.to_excel(path.xlsx, index=save_index, float_format=float_format)
            case _:
                raise ValueError(f"Unknown format: {f}")


class ModelEvalSubdir(type(Path())):
    def __new__(cls, *args, **kwargs):
        """
        Represents the directory and file structure of a patient's model evaluation dir.

        Parameters
        ----------
        for_single_model, bool
            Must specify!
        """
        for_single_model: bool = bool(kwargs.pop('for_single_model'))
        self = super().__new__(cls, *args, *kwargs)

        if for_single_model:
            self.metrics_table = MultiPath(self, 'metrics')
            self.ebm_table = MultiPath(self, 'event_based_metrics')
            self.szr_pred_table = MultiPath(self, 'szrs_predicted')
        else:  # It's for a split and both models
            self.metrics_plot = Path(self, 'metrics_plot.pdf')

        return self


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
        szr_starts_naive_name = f'{self.name}_Consensus' if self.dataset == Dataset.ultra2 else 'seizure_starts_naive'
        self.szr_starts_naive_file = MultiPath(self.szr_anns_original_dir, szr_starts_naive_name)
        self.all_szr_starts_file = MultiPath(self.szr_anns_dir, "seizure_starts_all")
        self.valid_szr_starts_file = MultiPath(self.szr_anns_dir, "seizure_starts_valid")

        ### edf data
        # The directory containing the original edf files for the competition dataset, before they're renamed
        self.original_edf_dir = Path(self, 'original_edf_data')
        # The directory containing the edf files
        self.edf_dir = Path(self, 'edf_data')
        # The name of the table containing the edf file names and their metadata for each patient
        self.edf_files_table = MultiPath(self, 'edf_files')
        self.valid_edf_intervals = MultiPath(self, 'edf_intervals_valid')
        self.invalid_edf_intervals = MultiPath(self, 'edf_intervals_invalid')

        ### Preprocessing
        self.segments_table = MultiPath(self, 'segments')
        self.clips_table = MultiPath(self, 'clips')
        self.segments_plot = Path(self, 'segments_plot.png')
        self.dataset_partition = MultiPath(self, 'dataset_partition')

        ### ML models
        self.models_dir = Path(self, 'models')
        self.ensemble_model = Path(self.models_dir, 'ensemble.keras')
        self.feature_scaler = Path(self.models_dir, 'feature_scaler.pkl')
        self.cnn_model = Path(self.models_dir, 'CNN.keras')
        self.cnn_history = MultiPath(self.models_dir, 'CNN_training_history')
        self.mlp_history = MultiPath(self.models_dir, 'FB_MLP_training_history')

        ### Predictions
        self.predicted_scores_dir = Path(self, 'predicted_scores')
        self.segment_scores_table = MultiPath(self.predicted_scores_dir, 'segment_scores')
        self.clip_scores_table = MultiPath(self.predicted_scores_dir, 'clip_scores')

        # Model Evaluation
        self.model_eval_dir = Path(self, 'model_evaluation')

        # Cycles
        self.cycle_extraction_dir = Path(self, 'cycle_extraction')
        self.filled_features_for_segs = MultiPath(self.cycle_extraction_dir, 'filled_features_for_segs')
        self.cycle_extraction_results_table = MultiPath(self.cycle_extraction_dir, 'cycle_extraction_metrics')
        self.circular_comparison_table = MultiPath(self.cycle_extraction_dir, 'circular_comparison_results')
        self.filtered_feature_plots_dir = Path(self.cycle_extraction_dir, 'filtered_feature_plots')
        self.circular_histograms_dir = Path(self.cycle_extraction_dir, 'circular_histograms')

        return self

    def model_eval_subdir(self, split: str, model: Optional[str] = None) -> ModelEvalSubdir:
        path = self.model_eval_dir / split
        for_single_model = model is not None
        if for_single_model:
            path /= model
        path.mkdir(parents=True, exist_ok=True)
        return ModelEvalSubdir(path, for_single_model=for_single_model)


class Paths(type(Path())):
    def __new__(cls, *args, **kwargs):
        """Represents the directory and file structure of the project.
        :param *args: The base_dir: str | PathLike[str]
        """
        self = super().__new__(cls, *args, **kwargs)

        # General
        self.logs_dir = Path(self, "logs")

        # dataset dirs
        self.datasets_dir = Path(self, "datasets")  # The dir that contains the datasets
        self.ultra2_dir = Path(self.datasets_dir, Dataset.ultra2.value)
        self.competition_dir = Path(self.datasets_dir, Dataset.competition.value)

        # data cleaning logs
        self.data_cleaning_logs_dir = Path(self, "data_cleaning_logs")
        self.problematic_edfs_dir = MultiPath(self.data_cleaning_logs_dir, 'problematic_edfs')
        self.remaining_duplicates_file = Path(self.data_cleaning_logs_dir / 'remaining_duplicates.txt')

        # preprocessing
        self.patient_info_dir = Path(self, "patient_info")
        self.basic_patient_info = Path(self.patient_info_dir, "basic_patient_info.xlsx")
        self.patient_info_exact = MultiPath(self.patient_info_dir, "patient_info_exact")
        self.patient_info_readable = Path(self.patient_info_dir, "patient_info_readable.csv")
        self.invalid_patients_dir = Path(self, "invalid_patients")
        self.partition_info_table = MultiPath(self.patient_info_dir, "partition_info")

        # model comparison, cycle extraction
        self.statistical_results_dir = Path(self, "statistical_results")
        self.per_patient_comparison_table = MultiPath(self.statistical_results_dir, "per_patient")
        self.per_model_comparison_table = MultiPath(self.statistical_results_dir, "per_model")
        self.cycle_extraction_results_table = MultiPath(self.statistical_results_dir, "cycle_extraction")

        return self

    def patient_dirs(
            self, datasets: Optional[List[Dataset]] = None,
            include_invalid_ptnts: bool = False,
            include_fake_ptnts: bool = True,
    ) -> List[PatientDir]:
        """
        Return a list of patient directories of the given datasets
        :param datasets: The datasets to get patient dirs for (default: all)
        :param include_invalid_ptnts: Whether to include invalid patient dirs
        :param include_fake_ptnts: Whether to include patients with "FAKE" in their name
        :returns: pdirs - a list of PatientDir objects
        """
        if datasets is None:
            datasets = list(Dataset)

        base_dirs = [self.datasets_dir]
        if include_invalid_ptnts:
            base_dirs.append(self.invalid_patients_dir)

        pdirs = []
        for base_dir in base_dirs:
            for dataset in datasets:
                dataset_path = base_dir / dataset.value
                if dataset_path.is_dir():
                    for pdir in sorted(dataset_path.iterdir()):
                        if pdir.is_dir():
                            pdirs.append(PatientDir(pdir, dataset=dataset))

        if not include_fake_ptnts:
            pdirs = [pdir for pdir in pdirs if "FAKE" not in pdir.name]

        return pdirs

    @property
    def root(self) -> Path:
        """:return: The root directory of the dataset"""
        # This is just an alias
        return Path(self)


if __name__ == '__main__':
    from config import PATHS

    for pdir_ in PATHS.patient_dirs(include_invalid_ptnts=True):
        print(pdir_)
