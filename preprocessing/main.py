from datetime import datetime
import logging

from config import PATHS, PatientDir
from preprocessing.create_clips import create_clips_for_pdirs
from preprocessing.filter_signals import filter_edfs_for_pdirs
from preprocessing.create_segments import create_segs_for_pdirs
from preprocessing.dataset_partitioning import partition_for_pdirs
from preprocessing.plot_segments import plot_segs_for_pdirs
from preprocessing.validate_patients import validate_patients
from feature_extraction.extract_features import extract_features_for_pdirs
from utils.logging_config import configure_root_logging
from utils.utils import FunctionTimer


def preprocessing(
        pdirs: list[PatientDir],
        ask_confirm: bool = True,
        setup_logging: bool = False,
):
    """
    Run the preprocessing pipeline.
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
            all_pdirs = PATHS.patient_dirs(include_invalid_ptnts=True)
            all_valid_pdirs = validate_patients(all_pdirs, move_invalid_pdirs=True, leave_fake_ptnts=True)

        # Keep just valid pdirs
        valid_pdirs = [pdir for pdir in pdirs if pdir in all_valid_pdirs]
        logging.info(f'Excluded invalid patients.')
        logging.info(f'({len(valid_pdirs)}) valid patients remain: {[p.name for p in valid_pdirs]}')

        logging.info("---- Filtering EDFs ----")
        with FunctionTimer('filter_all_edfs'):
            filter_edfs_for_pdirs(valid_pdirs)

        logging.info("---- Creating segment tables ----")
        with FunctionTimer('create_segs_for_pdirs'):
            create_segs_for_pdirs(valid_pdirs)

        logging.info("---- Creating clips ----")
        with FunctionTimer('create_clips_for_pdirs'):
            create_clips_for_pdirs(valid_pdirs)

        logging.info("---- Partitioning Dataset ----")
        with FunctionTimer('partition_for_pdirs'):
            partition_for_pdirs(valid_pdirs)

        logging.info("---- Creating Segment Plots ----")
        with FunctionTimer('plot_segs_for_pdirs'):
            plot_segs_for_pdirs(valid_pdirs)

        logging.info("---- Extracting features ----")
        with FunctionTimer('run_feature_extraction'):
            extract_features_for_pdirs(valid_pdirs)

        return valid_pdirs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    pdirs_ = PATHS.patient_dirs()
    preprocessing(pdirs_, setup_logging=False)
