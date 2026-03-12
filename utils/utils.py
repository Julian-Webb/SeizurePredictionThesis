import logging
import os
import time
from functools import wraps
from pathlib import Path


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


def timeit(f=None, *, arg_indices: list[int] | None = None, kwarg_names: list[str] | None = None):
    """Decorator that logs the execution time of a function.

    Usage:
        @timeit                                             – shows all args & kwargs
        @timeit(arg_indices=[0, 1])                         – selected positional args by index
        @timeit(kwarg_names=['patient'])                    – selected keyword args by name
        @timeit(arg_indices=[0], kwarg_names=['patient'])   – both
    """
    def decorator(func):
        @wraps(func)
        def wrap(*args, **kw):
            start = time.perf_counter()
            result = func(*args, **kw)
            elapsed = time.perf_counter() - start

            selected = (
                [repr(args[i]) for i in arg_indices if i < len(args)]
                if arg_indices is not None
                else [repr(a) for a in args]
            )
            selected += (
                [f"{k}={repr(kw[k])}" for k in kwarg_names if k in kw]
                if kwarg_names is not None
                else [f"{k}={repr(v)}" for k, v in kw.items()]
            )

            args_str = f"({', '.join(selected)})" if selected else ""
            logging.info(f"[TIMING] {elapsed:.3f}s - {func.__name__}{args_str}: ")
            return result

        return wrap

    # Supports both @timeit and @timeit(...) usage
    if f is not None:
        return decorator(f)
    return decorator


def import_tensorflow_with_available_gpus(available_gpus: list[int]):
    gpus_str = ','.join(map(str, available_gpus))
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_str
    # noinspection PyUnusedImports
    import tensorflow as tf


MAC_PATTERNS = [
    '._.DS_Store',  # Resource fork for .DS_Store
    '.DS_Store',  # Finder metadata
    '._*',  # All resource fork files
    '.AppleDouble',  # Apple Double format directory
    '.LSOverride'  # Finder custom attributes
]


def clean_mac_files(directory: Path):
    """Removes macOS system files, such as .DS_Store and ._ files"""
    # go through all the files in the directory and delete them
    # logging.info('===== Removing macOS system files =====')
    removed_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Check exact matches
            path = Path(root, filename)
            if filename in MAC_PATTERNS:
                try:
                    path.unlink()
                    removed_files.append(path)
                    # logging.info(f"Removed: {Path(root, filename)}")
                except OSError as e:
                    logging.warning(f"Error removing {path}: {e}")

            # Check pattern matches (for ._*)
            elif filename.startswith('._'):
                try:
                    path.unlink()
                    removed_files.append(path)
                    # logging.info(f"Removed: {path}")
                except OSError as e:
                    logging.warning(f"Error removing {path}: {e}")

    logging.info(f"Removed {len(removed_files)} mac system files")
