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


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = time.perf_counter()
        result = f(*args, **kw)
        elapsed = time.perf_counter() - start
        logging.info(f"[TIMING] {f.__name__}: {elapsed:.3f}s")
        return result

    return wrap


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
