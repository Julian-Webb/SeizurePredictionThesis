from __future__ import annotations

from datetime import datetime
import logging
from typing import List

from config import PATHS, PatientDir
from utils import QueuedCall, run_queued_calls_on_gpus


def _train_cnn_task(pdir: PatientDir):
    # Imported inside worker after CUDA_VISIBLE_DEVICES is set by gpu_multiprocessing
    from models.CNN import create_ptnt_cnn_and_save
    create_ptnt_cnn_and_save(pdir)


def _train_ensemble_task(pdir: PatientDir) -> None:
    from models.ensemble import create_ptnt_ensemble_and_save
    create_ptnt_ensemble_and_save(pdir)


def train_models(pdirs: List[PatientDir], gpus: List[int], train_cnn: bool = True, train_ensemble: bool = True):
    tasks = []
    for pdir in pdirs:
        if train_cnn:
            tasks.append(QueuedCall(func=_train_cnn_task, args=(pdir,), label=f"{pdir.name} - CNN"))
        if train_ensemble:
            tasks.append(QueuedCall(func=_train_ensemble_task, args=(pdir,), label=f"{pdir.name} - ensemble"))

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_log_dir = PATHS.logs_dir / f"train_models_{run_name}_gpu_logs"
    merged_log_file = PATHS.logs_dir / f"train_models_{run_name}.log"

    PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing per-GPU model-training logs to %s", run_log_dir)
    logging.info("Writing merged model-training log to %s", merged_log_file)

    run_queued_calls_on_gpus(
        tasks=tasks,
        gpus=gpus,
        log_dir=run_log_dir,
        merged_log_file=merged_log_file,
        keep_gpu_logs=True,
    )

    return merged_log_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(asctime)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    pdirs_ = [
        PatientDir('/data/home/webb/d/UNEEG/datasets/competition/competition-01-MINIFAKE'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/competition/competition-1'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/competition/competition-2'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/competition/competition-3'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-01'),
        PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-01-FAKE'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-03'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-04'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-05'),
        PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-07'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-12'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-15'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-16'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-17')
    ]
    train_models(
        pdirs_,
        gpus=[0, 1, 2, 3],
        train_cnn=True,
        train_ensemble=True,
    )
