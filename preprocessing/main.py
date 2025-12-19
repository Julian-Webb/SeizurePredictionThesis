import logging

from config.paths import PATHS
from preprocessing.estimate_seizure_starts import estimate_seizure_starts
from preprocessing.segment_tables import segment_tables
from preprocessing.train_test_split import find_ptnt_splits
from preprocessing.validate_patients import validate_patients
from preprocessing.extract_features import extract_features
from utils.utils import FunctionTimer


def preprocessing(ask_confirm: bool = True):
    logging.info("==== Preprocessing ====")

    if ask_confirm:
        input(f"Preprocessing for {PATHS.base_dir}. Press enter to continue.")

    with FunctionTimer('Total Preprocessing'):
        logging.info(f"===== Estimating seizure starts =====")
        with FunctionTimer('estimate_seizure_starts'):
            estimate_seizure_starts()

        logging.info("===== Validating Patients and moving invalid patient dirs =====")
        with FunctionTimer('validate_patients'):
            validate_patients(PATHS.patient_dirs(include_invalid_ptnts=True), move_invalid_ptnt_dirs=True)

        logging.info("===== Creating segment tables =====")
        with FunctionTimer('segment_tables'):
            segment_tables(PATHS.patient_dirs(include_invalid_ptnts=False))

        logging.info("Splitting data into train and test")
        with FunctionTimer('split_train_test'):
            find_ptnt_splits(PATHS.patient_dirs())

        logging.info("Extracting features")
        with FunctionTimer('extract_features'):
            extract_features(PATHS.patient_dirs())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    preprocessing()
