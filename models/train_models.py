from __future__ import annotations

from typing import List

from config.paths import PATHS, PatientDir
from utils import QueuedCall, run_queued_calls_on_gpus


def _train_cnn_task(pdir: PatientDir):
    # Imported inside worker after CUDA_VISIBLE_DEVICES is set by gpu_multiprocessing
    from models.CNN import create_ptnt_cnn
    create_ptnt_cnn(pdir)


def _train_mlp_task(pdir: PatientDir) -> None:
    from models.FB_MLP import create_ptnt_ensemble_and_save
    create_ptnt_ensemble_and_save(pdir)


def train_models(pdirs: List[PatientDir], gpus: List[int], train_cnn: bool = True, train_mlp: bool = True):
    tasks = []
    for pdir in pdirs:
        if train_cnn:
            tasks.append(QueuedCall(func=_train_cnn_task, args=(pdir,), label=f"{pdir.name} - CNN"))
        if train_mlp:
            tasks.append(QueuedCall(func=_train_mlp_task, args=(pdir,), label=f"{pdir.name} - FB-MLP"))

    run_queued_calls_on_gpus(tasks=tasks, gpus=gpus)

if __name__ == "__main__":
    train_models(
        PATHS.patient_dirs(),
        gpus=[0, 1, 2, 3],
        train_cnn=True,
        train_mlp=True,
    )
