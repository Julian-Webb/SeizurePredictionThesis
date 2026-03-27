import logging

from config import PatientDir, PATHS
from model_eval import calc_segment_probabilities, clips, event_based_metrics, eval_models
from utils.utils import FunctionTimer


def model_eval(
        pdirs: list[PatientDir],
        available_gpus: list[int],
):
    """
    Run all the model evaluation steps.
    """
    with FunctionTimer("====== Calculating Segment Probabilities ======="):
        calc_segment_probabilities.main(pdirs, serial_processing=False, available_gpus=available_gpus)

    with FunctionTimer('====== Computing Clips ======'):
        clips.main(pdirs)

    with FunctionTimer('====== Computing Event-Based Metrics ======'):
        event_based_metrics.calc_metrics(pdirs)

    with FunctionTimer('====== Evaluating Models ======'):
        eval_models.main(pdirs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    model_eval(
        PATHS.patient_dirs(),
        available_gpus=[0, 1, 2, 3]
    )
