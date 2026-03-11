import os
from pathlib import Path

import pytest

from utils.gpu_multiprocessing import QueuedCall, run_queued_calls_on_gpus


def _busy_tf_job(log_path: str, task_id: int, mat_size: int = 192, steps: int = 10) -> None:
    """Run enough TF math to keep a worker busy for a short time and emit logs."""

    import tensorflow as tf  # Runtime import keeps worker setup isolated.

    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    log_file = Path(log_path)

    print(
        f"[worker pid={os.getpid()}] task={task_id} start visible_gpu={visible_gpu}",
        flush=True,
    )

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"task={task_id} phase=start pid={os.getpid()} visible_gpu={visible_gpu}\n")

    left = tf.random.uniform((mat_size, mat_size), dtype=tf.float32)
    right = tf.random.uniform((mat_size, mat_size), dtype=tf.float32)
    accumulator = tf.constant(0.0, dtype=tf.float32)

    for _ in range(steps):
        product = tf.matmul(left, right)
        accumulator = accumulator + tf.reduce_sum(tf.math.tanh(product))
        left, right = right, left

    checksum = float(accumulator.numpy())

    print(
        f"[worker pid={os.getpid()}] task={task_id} end checksum={checksum:.6f}",
        flush=True,
    )

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"task={task_id} phase=end checksum={checksum:.6f}\n")


def _no_op() -> None:
    return


def test_run_queued_calls_on_gpus_executes_tensorflow_work(tmp_path, capfd):
    gpus = [0, 1, 2, 3]

    log_path = tmp_path / "tf_worker_tasks.log"

    print(f"Writing logs to {log_path}")

    task_count = len(gpus) * 2
    tasks = [
        QueuedCall(
            func=_busy_tf_job,
            args=(str(log_path), task_id),
            kwargs={"mat_size": 224, "steps": 10},
            label=f"tf_task_{task_id}",
        )
        for task_id in range(task_count)
    ]

    # Disable pytest capture while workers run so subprocess prints reach Run output.
    disabled_capture = getattr(capfd, "disabled")
    with disabled_capture():
        print("\nRunning worker tasks with live stdout enabled...", flush=True)
        run_queued_calls_on_gpus(tasks=tasks, gpus=gpus, continue_on_error=False)


    log_text = log_path.read_text(encoding="utf-8")
    for task_id in range(task_count):
        assert f"task={task_id} phase=start" in log_text
        assert f"task={task_id} phase=end" in log_text

    assert "visible_gpu=" in log_text


def test_run_queued_calls_on_gpus_validates_input():
    with pytest.raises(ValueError, match="`gpus` is empty"):
        run_queued_calls_on_gpus(tasks=[QueuedCall(func=_no_op)], gpus=[])

    with pytest.raises(ValueError, match="`tasks` is empty"):
        run_queued_calls_on_gpus(tasks=[], gpus=[0])

