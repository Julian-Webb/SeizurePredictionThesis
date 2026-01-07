import logging
import time
from functools import wraps
import tensorflow as tf


def safe_float_to_int(num: float) -> int:
    if num != int(num):
        raise ValueError(f"Number {num} has decimal values")
    return int(num)


class FunctionTimer:
    def __init__(self, label: str = "elapsed"):
        self.label = label
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        logging.info(f"[TIMING] {self.label}: {self.elapsed:.3f}s")


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = time.perf_counter()
        result = f(*args, **kw)
        elapsed = time.perf_counter() - start
        logging.info(f"[TIMING] {f.__name__}: {elapsed:.3f}s")
        return result

    return wrap


class PeriodicalLogger(tf.keras.callbacks.Callback):
    def __init__(self, model_name, interval=100):
        super().__init__()
        self.model_name = model_name
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if (epoch == 1) or (epoch % self.interval == 0):
            # logs is a dict containing the metrics defined in model.compile
            msg = f"[{self.model_name}] Epoch {epoch}/{self.params['epochs']}"
            for metric, value in logs.items():
                msg += f" - {metric}: {value:.4f}"
            logging.info(msg)
