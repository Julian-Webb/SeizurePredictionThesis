import logging
from datetime import datetime

from config import PatientDir, PATHS
from model_eval import calc_segment_probabilities, clips, event_based_metrics, eval_models
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
    with FunctionTimer('Calculating Segment Probabilities'):
        calc_segment_probabilities.main(
            pdirs,
            serial_processing=False,
            available_gpus=available_gpus,
            merged_log_file=PATHS.logs_dir / f"{run_name}_calc_segment_probabilities.log",
        )

    # Log the rest to a separate file
    configure_root_logging(log_file=PATHS.logs_dir / f"{run_name}_model_eval.log")

    with FunctionTimer('Computing Clips'):
        clips.main(pdirs)

    with FunctionTimer('Computing Event-Based Metrics'):
        event_based_metrics.calc_metrics(pdirs)

    with FunctionTimer('Evaluating Models'):
        eval_models.main(pdirs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    model_eval(
        PATHS.patient_dirs(),
        available_gpus=[0, 1, 2, 3]
    )
