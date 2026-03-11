"""Reusable multi-GPU multiprocessing utilities for TensorFlow workloads.

Design goals:
- Each worker process sees exactly one GPU via ``CUDA_VISIBLE_DEVICES``.
- TensorFlow is configured inside each worker before model code runs.
- Work is defined as a queue of callables plus args/kwargs.
- Workers shut down gracefully via sentinels.

Usage example:
    from config import PATHS
    from models.CNN import create_ptnt_cnn
    from models.FB_MLP import create_ptnt_ensemble_and_save
    from utils.gpu_multiprocessing import QueuedCall, run_queued_calls_on_gpus

    tasks = []
    for pdir in PATHS.patient_dirs():
        tasks.append(QueuedCall(func=create_ptnt_cnn, args=(pdir,), label=f"{pdir.name} - CNN"))
        tasks.append(
            QueuedCall(
                func=create_ptnt_ensemble_and_save,
                args=(pdir,),
                label=f"{pdir.name} - FB-MLP",
            )
        )

    run_queued_calls_on_gpus(tasks=tasks, gpus=[0, 1, 2, 3])
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from multiprocessing import Process, Queue, get_context
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class QueuedCall:
    """A picklable task descriptor for queue-based worker execution."""

    func: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    label: str | None = None


def configure_tf_for_single_visible_gpu(gpu_id: int, tf_log_level: str = "1") -> None:
    """Pin worker process to one GPU and configure TensorFlow memory growth."""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", tf_log_level)

    import tensorflow as tf  # noqa: WPS433 - runtime import is intentional

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            print(f"[GPU {gpu_id}] Could not enable memory growth: {exc}")


def _gpu_worker_loop(gpu_id: int, queue: Queue, continue_on_error: bool) -> None:
    """Consume queued tasks until a sentinel is received."""

    configure_tf_for_single_visible_gpu(gpu_id)

    while True:
        task = queue.get()
        if task is None:
            print(f"[GPU {gpu_id}] Shutdown signal received")
            return

        if not isinstance(task, QueuedCall):
            print(f"[GPU {gpu_id}] Invalid task payload: {type(task).__name__}")
            if continue_on_error:
                continue
            raise TypeError(f"Expected QueuedCall, got {type(task).__name__}")

        task_name = task.label or getattr(task.func, "__name__", "queued_call")
        print(f"[GPU {gpu_id}] Starting {task_name}")
        try:
            task.func(*task.args, **dict(task.kwargs))
            print(f"[GPU {gpu_id}] Finished {task_name}")
        except Exception as exc:
            print(f"[GPU {gpu_id}] ERROR in {task_name}: {exc}")
            if not continue_on_error:
                raise


def run_queued_calls_on_gpus(
    tasks: Iterable[QueuedCall],
    gpus: Sequence[int],
    *,
    continue_on_error: bool = True,
) -> None:
    """Run queue-defined tasks across one worker process per GPU.

    Args:
        tasks: Iterable of ``QueuedCall`` items.
        gpus: Physical GPU ids to use (for example ``[0, 1, 2, 3]``).
        continue_on_error: If ``True``, worker logs task exceptions and keeps going.
            If ``False``, worker raises and exits on first failing task.
    """

    gpus = list(gpus)
    task_list = list(tasks)

    if not gpus:
        raise ValueError("`gpus` is empty. Provide at least one GPU id.")
    if not task_list:
        raise ValueError("`tasks` is empty. Provide at least one QueuedCall.")

    start = time.perf_counter()

    ctx = get_context("spawn")
    queue: Queue = ctx.Queue()

    for task in task_list:
        queue.put(task)
    for _ in gpus:
        queue.put(None)

    workers: list[Process] = []
    for gpu_id in gpus:
        process = ctx.Process(
            target=_gpu_worker_loop,
            args=(gpu_id, queue, continue_on_error),
            daemon=False,
        )
        process.start()
        workers.append(process)

    for process in workers:
        process.join()

    elapsed = time.perf_counter() - start
    print(f"Completed {len(task_list)} task(s) on {len(gpus)} GPU(s) in {elapsed / 3600:.2f} h")

