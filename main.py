import logging

from cleaning_annotations.localize_annotations import drop_duplicates_and_localize
from config import PATHS
from model_eval.main import model_eval
from models.train_models import train_models
from preprocessing.main import preprocessing


def main(
        available_gpus: list[int],
        ask_confirm: bool = True,
):
    """
    Perform all steps of the data pipeline to process the data from a copy of UNEEG_base
    """
    if ask_confirm:
        input(f"Preprocessing for {PATHS.root}. Press enter to continue.")

    def setup_logging():
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    setup_logging()

    pdirs = PATHS.patient_dirs()

    # Clean Annotations
    drop_duplicates_and_localize(pdirs)

    preprocessing(ask_confirm=False, setup_logging=True)

    train_models(pdirs, gpus=available_gpus, train_cnn=True, train_ensemble=True)

    setup_logging() # just to make sure the logs aren't written to files any more
    model_eval(pdirs, available_gpus)

if __name__ == "__main__":
    main(
        available_gpus=[0, 1, 2, 3],
    )