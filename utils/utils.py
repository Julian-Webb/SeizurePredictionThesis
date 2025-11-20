import logging
import time


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


def timeit(label: str = None):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            lbl = label or fn.__name__
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logging.info(f"[TIMING] {lbl}: {elapsed:.3f}s")
            return result

        return wrapper

    return decorator
