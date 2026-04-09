from __future__ import annotations

import logging
from datetime import datetime
from typing import List

from config import PATHS, PatientDir
from utils import QueuedCall, run_queued_calls_on_gpus


def _train_cnn_task(pdir: PatientDir):
    # Imported inside worker after CUDA_VISIBLE_DEVICES is set by gpu_multiprocessing
    from models.CNN import create_cnn_and_save_for_pdir
    create_cnn_and_save_for_pdir(pdir)


def _train_ensemble_task(pdir: PatientDir) -> None:
    from models.ensemble import create_ensemble_and_save_for_pdir
    create_ensemble_and_save_for_pdir(pdir)


def train_models_for_pdirs(
        pdirs: List[PatientDir],
        gpus: List[int],
        train_cnn: bool = True,
        train_ensemble: bool = True,
        run_name: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
):
    run_log_dir = PATHS.logs_dir / f"{run_name}_train_models_gpu_logs"
    merged_log_file = PATHS.logs_dir / f"{run_name}_train_models.log"

    PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing per-GPU model-training logs to %s", run_log_dir)
    logging.info("Writing merged model-training log to %s", merged_log_file)

    # Create tasks
    # Add ensembles first, because they take longer to train. If not done this way, some GPUs will shut down while
    # others have just started the long ensemble training process for the last patients
    tasks = []
    if train_ensemble:
        for pdir in pdirs:
            tasks.append(QueuedCall(func=_train_ensemble_task, args=(pdir,), label=f"{pdir.name} - ensemble"))
    if train_cnn:
        for pdir in pdirs:
            tasks.append(QueuedCall(func=_train_cnn_task, args=(pdir,), label=f"{pdir.name} - CNN"))

    run_queued_calls_on_gpus(
        tasks=tasks,
        gpus=gpus,
        log_dir=run_log_dir,
        merged_log_file=merged_log_file,
        keep_gpu_logs=False,
    )

    return merged_log_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    pdirs_ = PATHS.patient_dirs()
    print(f'Training patients: ')
    for pdir_ in pdirs_:
        print(f'    {pdir_.name}')

    train_models_for_pdirs(
        pdirs_,
        gpus=[0, 1, 2, 3],
        train_cnn=True,
        train_ensemble=True,
    )
