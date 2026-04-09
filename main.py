import logging
from datetime import datetime

from cleaning_annotations.localize_annotations import drop_duplicates_and_localize_for_pdirs
from config import PATHS, PatientDir
from cycle_extraction.main import cycle_extraction
from model_eval.main import model_eval
from models.train_models import train_models_for_pdirs
from preprocessing.main import preprocessing
from utils.logging_config import configure_root_logging
from utils.utils import timeit


@timeit
def main(
        pdirs: list[PatientDir],
        available_gpus: list[int],
        ask_confirm: bool = True,
):
    """
    Perform all steps of the data pipeline to process the data from a copy of UNEEG_base
    """
    if ask_confirm:
        print(f"Preprocessing for {PATHS.root}. Patients:")
        for pdir in pdirs:
            print(f"  {pdir.name}")
        input(f"Press enter to continue.")

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
    root_log = PATHS.logs_dir / f"{run_name}_root.log"

    # Default logging for top-level orchestration and console output.
    configure_root_logging(log_file=root_log)
    logging.info(f'Starting pipeline for {PATHS.root} with patients:')
    for pdir in pdirs:
        logging.info(f'    {pdir.name}')

    # ---- Pipeline Start ----------------------------------------------------------------------------------------------
    # Preprocessing
    configure_root_logging(log_file=PATHS.logs_dir / f"{run_name}_preprocessing.log")
    drop_duplicates_and_localize_for_pdirs(pdirs)  # Clean Annotations
    # valid patient directories are returned
    v_pdirs = preprocessing(pdirs, ask_confirm=False, setup_logging=False)

    # Train Models - sets up and manages its own logs.
    configure_root_logging()
    train_models_for_pdirs(v_pdirs, gpus=available_gpus, train_cnn=True, train_ensemble=True, run_name=run_name)

    # Model Evaluation - sets up and manages its own logs.
    configure_root_logging()
    model_eval(v_pdirs, available_gpus, run_name=run_name)

    # Cycle Extraction
    configure_root_logging(log_file=PATHS.logs_dir / f"{run_name}_cycle_extraction.log")
    cycle_extraction(v_pdirs, ask_confirm=False)

    # ---- Pipeline End ------------------------------------------------------------------------------------------------
    # Success message
    configure_root_logging(root_log)
    ptnt_names = [pdirs.name for pdirs in v_pdirs]
    logging.info(f'Completed pipeline for {ptnt_names}.')


if __name__ == "__main__":
    pdirs_ = PATHS.patient_dirs()
    main(
        pdirs_,
        available_gpus=[0, 1, 2, 3],
    )
