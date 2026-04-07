import logging
from datetime import datetime

from config import PatientDir, PATHS
from model_eval import calc_scores, event_based_metrics, eval_models
from utils.logging_config import configure_root_logging
from utils.utils import FunctionTimer


def model_eval(
        pdirs: list[PatientDir],
        available_gpus: list[int],
        run_name: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
):
    """
    Run all the model evaluation steps.
    """
    with FunctionTimer('Calculating Scores for Segments and Clips'):
        calc_scores.calc_scores_for_pdirs(
            pdirs,
            serial_processing=False,
            available_gpus=available_gpus,
            merged_log_file=PATHS.logs_dir / f"{run_name}_calc_scores.log",
        )

    # Log the rest to a separate file
    configure_root_logging(log_file=PATHS.logs_dir / f"{run_name}_model_eval.log")

    with FunctionTimer('Computing Event-Based Metrics'):
        event_based_metrics.event_based_metrics_for_pdirs(pdirs)

    with FunctionTimer('Evaluating Models'):
        eval_models.eval_models_for_pdirs(pdirs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    model_eval(
        PATHS.patient_dirs(),
        available_gpus=[0, 1, 2, 3]
    )
