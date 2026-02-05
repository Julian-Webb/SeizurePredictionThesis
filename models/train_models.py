"""
Python-only launcher to train per-patient models in parallel, one process per GPU.

Behavior:
- Enumerates patients via PATHS.patient_dirs().
- Spawns one worker process per GPU listed in GPUS.
- Each worker pins itself to a single GPU (CUDA_VISIBLE_DEVICES) before importing TF or model code.
- For each patient dequeued, trains CNN first, then FB-MLP.
- Enables TensorFlow memory growth per process.
"""
from __future__ import annotations

import os
import time
from multiprocessing import Process, Queue, set_start_method
from typing import List

from config.paths import PATHS, PatientDir


def _setup_gpu_and_tf(gpu_id: int):
    """Set CUDA visibility and configure TF memory before importing model modules."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    # Import TensorFlow and set memory growth
    import tensorflow as tf  # noqa: WPS433 (runtime import intentional)
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except:
            # Best-effort; ignore if not supported
            print(f"Couldn't set experimental GPU growth for {gpu_id}")


def _worker(gpu_id: int, q: Queue, train_cnn: bool, train_mlp: bool):
    # Important: pin GPU and configure TF BEFORE importing model code
    _setup_gpu_and_tf(gpu_id)

    # Import training functions AFTER setting CUDA visibility
    from models.CNN import create_ptnt_cnn  # noqa: WPS433
    from models.FB_MLP import create_ptnt_ensemble_and_save  # noqa: WPS433

    while True:
        pdir = q.get()
        if pdir is None:
            break
        try:
            print(f"(GPU {gpu_id}) Starting {pdir.name}")
            if train_cnn:
                create_ptnt_cnn(pdir)
                print(f"(GPU {gpu_id}) Finished {pdir.name} - CNN")

            if train_mlp:
                create_ptnt_ensemble_and_save(pdir)
                print(f"(GPU {gpu_id}) Finished {pdir.name} - FB-MLP")

            print(f"(GPU {gpu_id}) Finished {pdir.name}")

        except Exception as e:  # Keep other patients running on error
            print(f"(GPU {gpu_id}) ERROR for {pdir.name}: {e}")


def train_models(pdirs: List[PatientDir], gpus: List[int], train_cnn: bool = True, train_mlp: bool = True):
    """
    Train patient models using multiple GPUs in parallel
    :param pdirs: List of patient dirs
    :param gpus: List of GPU indices to use. Example: [0,1,2,3] or [1] or [0,2]
    """
    start = time.perf_counter()

    # Use spawn to avoid inheriting CUDA/TF state (safer across platforms)
    try:
        set_start_method('spawn')
    except RuntimeError:
        # Already set by previous call/import; ignore
        pass

    if not gpus:
        raise SystemExit("gpus list is empty. Configure gpus = [0,1,2,3] or similar.")

    if not pdirs:
        raise SystemExit("No patients found. Check PATHS configuration and datasets.")

    # Fill the task queue
    q: Queue = Queue()
    for p in pdirs:
        q.put(p)
    # Add sentinel values to stop workers
    for _ in gpus:
        q.put(None)

    # Launch one process per GPU
    procs: List[Process] = []
    for gpu in gpus:
        p = Process(target=_worker, args=(gpu, q, train_cnn, train_mlp), daemon=False)
        p.start()
        procs.append(p)

    # Wait for all processes to finish
    for p in procs:
        p.join()

    elapsed_time = time.perf_counter() - start
    print(f"All patients completed in {elapsed_time / 3600:.2f} hours")


if __name__ == "__main__":
    train_models(
        PATHS.patient_dirs(),
        gpus=[0, 1, 2, 3],
        train_cnn=True,
        train_mlp=True,
    )
