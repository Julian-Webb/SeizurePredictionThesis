from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import Timestamp, Timedelta
from pyedflib import EdfReader


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


def load_segmented_sigs(file_path: Path, first_idx: int, n_segs: int, channels_last: bool = False) -> np.ndarray:
    """
    Read signals and segment them.
    :param first_idx: The first index of the first segment in that file.
    :param n_segs: The number of segments to read.
    :return: segmented_sigs
    """
    from config.constants import N_CHANNELS
    from config.intervals import SEGMENT

    total_samples = n_segs * SEGMENT.n_samples

    if channels_last:
        segmented_sigs = np.empty((n_segs, SEGMENT.n_samples, N_CHANNELS))
    else:
        segmented_sigs = np.empty((n_segs, N_CHANNELS, SEGMENT.n_samples))

    with EdfReader(str(file_path)) as edf:
        for chn in range(N_CHANNELS):
            s = edf.readSignal(chn, first_idx, total_samples).reshape((n_segs, SEGMENT.n_samples))
            if channels_last:
                segmented_sigs[:, :, chn] = s
            else:
                segmented_sigs[:, chn, :] = s

    return segmented_sigs


def plot_edf_portion(
        file_path: Path,
        start_index: int,
        n_samples: int,
        start_time: Timestamp = None,
        title: str = None,
        subplot_kw=None,
        subplots_kwargs=None,
        plot_kwargs=None,
):
    import matplotlib.pyplot as plt

    if subplots_kwargs is None:
        subplots_kwargs = {'figsize': (20, 8)}
    if plot_kwargs is None:
        plot_kwargs = {'linewidth': 0.5}

    with EdfReader(str(file_path)) as edf:
        n_chn = edf.signals_in_file
        signal_labels = edf.getSignalLabels()
        signals = np.zeros((n_chn, n_samples))
        for chn in range(n_chn):
            signals[chn, :] = edf.readSignal(chn, start_index, n_samples)

    fig, axes = plt.subplots(nrows=n_chn, ncols=1, sharex=True, sharey=True, subplot_kw=subplot_kw, **subplots_kwargs)
    if title:
        fig.suptitle(title)

    sample_period = Timedelta(seconds=1 / edf.getSampleFrequency(0))
    t = pd.date_range(start=start_time, periods=n_samples, freq=sample_period)

    for label, sig, ax in zip(signal_labels, signals, axes):
        ax.plot(t, sig, **plot_kwargs)
        ax.set_title(label)
        ax.set_xlabel('Time')
        ax.set_ylabel('uV')

    return fig
