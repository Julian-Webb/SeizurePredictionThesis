from datetime import datetime

from cleaning_annotations.localize_annotations import drop_duplicates_and_localize
from config import PATHS
from model_eval.main import model_eval
from models.train_models import train_models
from preprocessing.main import preprocessing
from utils.logging_config import configure_root_logging


def main(
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

    # Default logging for top-level orchestration and console output.
    configure_root_logging()

    pdirs = PATHS.patient_dirs()

    # Clean Annotations
    configure_root_logging(log_file=PATHS.logs_dir / f"clean_annotations_{run_name}.log")
    drop_duplicates_and_localize(pdirs)

    # Preprocessing
    configure_root_logging(log_file=PATHS.logs_dir / f"preprocessing_{run_name}.log")
    preprocessing(ask_confirm=False, setup_logging=False)

    # train_models sets up and manages its own logs.
    configure_root_logging()
    train_models(pdirs, gpus=available_gpus, train_cnn=True, train_ensemble=True)

    # Model Evaluation
    configure_root_logging(log_file=PATHS.logs_dir / f"model_eval_{run_name}.log")
    model_eval(pdirs, available_gpus)

    configure_root_logging()

if __name__ == "__main__":
    main(
        available_gpus=[0, 1, 2, 3],
    )