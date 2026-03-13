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

import logging
import os
import time
from dataclasses import dataclass, field
from multiprocessing import Queue, get_context
from pathlib import Path
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
            logging.warning("Could not enable memory growth: %s", exc)


def _configure_worker_logging(gpu_id: int, log_file: str | None, log_level: int) -> None:
    root = logging.getLogger()
    root.setLevel(log_level)

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt=f"[GPU {gpu_id}:%(levelname)s] %(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file is None:
        handler = logging.StreamHandler()
    else:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")

    handler.setFormatter(formatter)
    root.addHandler(handler)


def _gpu_worker_loop(
        gpu_id: int,
        queue: Queue,
        continue_on_error: bool,
        log_level: int,
        log_file: str | None = None,
) -> None:
    """Consume queued tasks until a sentinel is received."""

    _configure_worker_logging(gpu_id, log_file, log_level)
    configure_tf_for_single_visible_gpu(gpu_id)

    while True:
        task = queue.get()
        if task is None:
            logging.info("Shutdown signal received")
            return

        if not isinstance(task, QueuedCall):
            logging.error("Invalid task payload: %s", type(task).__name__)
            if continue_on_error:
                continue
            raise TypeError(f"Expected QueuedCall, got {type(task).__name__}")

        task_name = task.label or getattr(task.func, "__name__", "queued_call")
        logging.info("[%s] Starting task", task_name)
        try:
            task.func(*task.args, **dict(task.kwargs))
            logging.info("[%s] Finished task", task_name)
        except Exception:
            logging.exception("[%s] ERROR in task", task_name)
            if not continue_on_error:
                raise


def _merge_gpu_logs_by_id(
        gpus: Sequence[int],
        log_dir: Path,
        merged_log_file: Path,
        *,
        keep_gpu_logs: bool,
        header_line: str | None = None,
) -> None:
    merged_log_file.parent.mkdir(parents=True, exist_ok=True)
    with merged_log_file.open("w", encoding="utf-8") as merged_handle:
        if header_line:
            merged_handle.write(f"{header_line}\n\n")
        for idx, gpu_id in enumerate(sorted(gpus)):
            if idx > 0:
                merged_handle.write("\n")
            merged_handle.write(f"GPU {gpu_id}:\n")

            gpu_log_file = log_dir / f"gpu_{gpu_id}.log"
            if gpu_log_file.exists():
                gpu_text = gpu_log_file.read_text(encoding="utf-8")
                merged_handle.write(gpu_text)
                if gpu_text and not gpu_text.endswith("\n"):
                    merged_handle.write("\n")
            else:
                merged_handle.write("[no log lines]\n")

    if not keep_gpu_logs:
        for gpu_id in sorted(gpus):
            gpu_log_file = log_dir / f"gpu_{gpu_id}.log"
            if gpu_log_file.exists():
                gpu_log_file.unlink()


def run_queued_calls_on_gpus(
        tasks: Iterable[QueuedCall],
        gpus: Sequence[int],
        *,
        continue_on_error: bool = True,
        log_dir: str | os.PathLike[str] | None = None,
        merged_log_file: str | os.PathLike[str] | None = None,
        keep_gpu_logs: bool = True,
) -> None:
    """Run queue-defined tasks across one worker process per GPU.

    Args:
        tasks: Iterable of ``QueuedCall`` items.
        gpus: Physical GPU ids to use (for example ``[0, 1, 2, 3]``).
        continue_on_error: If ``True``, worker logs task exceptions and keeps going.
            If ``False``, worker raises and exits on first failing task.
        log_dir: Directory for per-GPU logs. If ``None``, workers log to stdout.
        merged_log_file: If provided with ``log_dir``, writes one merged log ordered by GPU id.
        keep_gpu_logs: Keep intermediate ``gpu_<id>.log`` files after merge.
    """

    gpus = list(gpus)
    task_list = list(tasks)

    if not gpus:
        raise ValueError("`gpus` is empty. Provide at least one GPU id.")
    if not task_list:
        raise ValueError("`tasks` is empty. Provide at least one QueuedCall.")

    worker_log_level = logging.getLogger().getEffectiveLevel()

    start = time.perf_counter()
    log_dir_path = Path(log_dir) if log_dir is not None else None
    merged_log_file_path = Path(merged_log_file) if merged_log_file is not None else None

    if merged_log_file_path is not None and log_dir_path is None:
        raise ValueError("`log_dir` must be provided when `merged_log_file` is set.")

    if log_dir_path is not None:
        log_dir_path.mkdir(parents=True, exist_ok=True)

    ctx = get_context("spawn")
    queue: Queue = ctx.Queue()

    for task in task_list:
        queue.put(task)
    for _ in gpus:
        queue.put(None)

    workers: list[Any] = []
    for gpu_id in gpus:
        gpu_log_file = str(log_dir_path / f"gpu_{gpu_id}.log") if log_dir_path is not None else None
        process = ctx.Process(
            target=_gpu_worker_loop,
            args=(gpu_id, queue, continue_on_error, worker_log_level, gpu_log_file),
            daemon=False,
        )
        process.start()
        workers.append(process)

    for process in workers:
        process.join()

    elapsed = time.perf_counter() - start
    completion_msg = f"Completed {len(task_list)} task(s) on {len(gpus)} GPU(s) in {elapsed / 3600:.2f} h"

    if merged_log_file_path is not None and log_dir_path is not None:
        _merge_gpu_logs_by_id(
            gpus=gpus,
            log_dir=log_dir_path,
            merged_log_file=merged_log_file_path,
            keep_gpu_logs=keep_gpu_logs,
            header_line=completion_msg,
        )
    logging.info(completion_msg)
