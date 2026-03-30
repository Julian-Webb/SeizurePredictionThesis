from datetime import datetime
import logging

from config import PATHS, PatientDir
from preprocessing.filter_signals import filter_edfs_for_pdirs
from preprocessing.segment_tables import make_segment_tables
from preprocessing.train_test_allocation import find_ptnt_splits
from preprocessing.validate_patients import validate_patients
from feature_extraction.extract_features import run_feature_extraction
from utils.logging_config import configure_root_logging
from utils.utils import FunctionTimer


def preprocessing(
        pdirs: list[PatientDir],
        ask_confirm: bool = True,
        setup_logging: bool = False,
):
    """

    Parameters
    ----------
    pdirs
        The patient directories to perform preprocessing on. Validation will always occur for ALL PATIENTS.
    ask_confirm
    setup_logging
    """
    if setup_logging:
        logging_file = PATHS.logs_dir / f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_preprocessing.log'
        print(f"Logging preprocessing to: {logging_file}")
        configure_root_logging(log_file=logging_file)

    if ask_confirm:
        input(f"Preprocessing for {PATHS.root}. Press enter to continue.")

    logging.info("---- Preprocessing ----")

    with FunctionTimer('Total Preprocessing'):
        logging.info("---- Validating Patients and moving invalid patient dirs ----")
        with FunctionTimer('validate_patients'):
            validate_patients(PATHS.patient_dirs(include_invalid_ptnts=True), move_invalid_pdirs=True,
                              leave_fake_ptnts=True)

        logging.info("---- Filtering EDFs ----")
        with FunctionTimer('filter_all_edfs'):
            filter_edfs_for_pdirs(pdirs)

        logging.info("---- Creating segment tables ----")
        with FunctionTimer('segment_tables'):
            make_segment_tables(pdirs)

        logging.info("---- Splitting data into train and test ----")
        with FunctionTimer('split_train_test'):
            find_ptnt_splits(pdirs)

        logging.info("---- Extracting features ----")
        with FunctionTimer('extract_features'):
            run_feature_extraction(pdirs)


if __name__ == "__main__":
    pdirs_ = PATHS.patient_dirs()
    preprocessing(pdirs_, setup_logging=True)
