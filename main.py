import logging
from datetime import datetime

from cleaning_annotations.localize_annotations import drop_duplicates_and_localize
from config import PATHS, PatientDir
from model_eval.main import model_eval
from models.train_models import train_models
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
        input(f"Preprocessing for {PATHS.root}. Press enter to continue.")

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
    root_log = PATHS.logs_dir / f"{run_name}_root.log"

    # Default logging for top-level orchestration and console output.
    configure_root_logging(log_file=root_log)

    # ---- Pipeline Start ----------------------------------------------------------------------------------------------
    # Clean Annotations
    configure_root_logging(log_file=PATHS.logs_dir / f"{run_name}_clean_annotations.log")
    drop_duplicates_and_localize(pdirs)

    # Preprocessing
    configure_root_logging(log_file=PATHS.logs_dir / f"{run_name}_preprocessing.log")
    preprocessing(pdirs, ask_confirm=False, setup_logging=False)

    # Train Models - sets up and manages its own logs.
    configure_root_logging()
    train_models(pdirs, gpus=available_gpus, train_cnn=True, train_ensemble=True, run_name=run_name)

    # Model Evaluation - sets up and manages its own logs.
    configure_root_logging()
    model_eval(pdirs, available_gpus, run_name=run_name)

    # ---- Pipeline End ------------------------------------------------------------------------------------------------
    # Success message
    configure_root_logging(root_log)
    ptnt_names = [pdirs.name for pdirs in pdirs]
    logging.info(f'Completed pipeline for {ptnt_names}.')


if __name__ == "__main__":
    pdirs_ = PATHS.patient_dirs()
    main(
        pdirs_,
        available_gpus=[0, 1, 2, 3],
    )
