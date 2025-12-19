from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from numpy import ndarray
from pyedflib import EdfReader

from config.constants import N_CHANNELS
from config.intervals import SEGMENT


def time_to_index(file_start: datetime, timestamp: datetime, sampling_freq_hz: float) -> float:
    """Based on the specified datetime in a recording, calculate the index of that timestamp.
    Note: the index will be returned as a float so that the reverse conversion can be made accurately.
    If actually used to index, round it."""
    time_dif = timestamp - file_start
    return time_dif.total_seconds() * sampling_freq_hz


def index_to_time(start_time: datetime, index: int, sampling_freq_hz: float):
    """Based on the specified index in a recording, calculate the datetime of that index."""
    time_dif = timedelta(seconds=index / sampling_freq_hz)
    return start_time + time_dif


def load_segmented_sigs(file_path: Path, first_idx: int, n_segs: int) -> ndarray:
    """
    Read signals and segment them.
    :return:
    """
    total_samples = n_segs * SEGMENT.n_samples
    segmented_sigs = np.empty((n_segs, N_CHANNELS, SEGMENT.n_samples))

    with EdfReader(str(file_path)) as edf:
        for chn in range(N_CHANNELS):
            s = edf.readSignal(chn, first_idx, total_samples)
            segmented_sigs[:, chn, :] = s.reshape((n_segs, SEGMENT.n_samples))

    return segmented_sigs
