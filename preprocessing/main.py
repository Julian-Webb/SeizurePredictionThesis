from datetime import datetime
import logging

from config import PATHS
from preprocessing.filter_signals import filter_all_edfs
from preprocessing.segment_tables import make_segment_tables
from preprocessing.train_test_allocation import find_ptnt_splits
from preprocessing.validate_patients import validate_patients
from feature_extraction.extract_features import run_feature_extraction
from utils.logging_config import configure_root_logging
from utils.utils import FunctionTimer


def preprocessing(
        ask_confirm: bool = True,
        setup_logging: bool = False,
):
    if setup_logging:
        logging_file = PATHS.logs_dir / f'preprocessing_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log'
        print(f"Logging to: {logging_file}")
        configure_root_logging(log_file=logging_file)

    logging.info("==== Preprocessing ====")

    if ask_confirm:
        input(f"Preprocessing for {PATHS.root}. Press enter to continue.")

    with FunctionTimer('Total Preprocessing'):
        logging.info("===== Validating Patients and moving invalid patient dirs =====")
        with FunctionTimer('validate_patients'):
            validate_patients(PATHS.patient_dirs(include_invalid_ptnts=True), move_invalid_pdirs=True,
                              leave_fake_ptnts=True)

        pdirs = PATHS.patient_dirs()
        logging.info("===== Filtering EDFs =====")
        with FunctionTimer('filter_all_edfs'):
            filter_all_edfs(pdirs)

        logging.info("===== Creating segment tables =====")
        with FunctionTimer('segment_tables'):
            make_segment_tables(pdirs)

        logging.info("Splitting data into train and test")
        with FunctionTimer('split_train_test'):
            find_ptnt_splits(pdirs)

        logging.info("Extracting features")
        with FunctionTimer('extract_features'):
            run_feature_extraction(pdirs)


if __name__ == "__main__":
    preprocessing(setup_logging=True)
