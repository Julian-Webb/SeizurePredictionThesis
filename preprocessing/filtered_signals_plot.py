import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import Timestamp, Timedelta
from pyedflib import highlevel

from config.constants import SAMPLING_FREQUENCY_HZ
from config import PatientDir, PATHS
from utils.io import pickle_path


def plot_signal_with_bad_regions(
        y: np.ndarray,
        invalid_intervals: np.ndarray = None,
        x: np.ndarray | pd.DatetimeIndex | None = None,
        ax: plt.Axes | None = None,
        signal_kwargs: dict | None = None,
        bad_color: str = "red",
        bad_alpha: float = 0.50,
) -> plt.Axes:
    """
    Plot a 1D signal and shade regions that are in invalid_mask in red.

    Parameters
    ----------
    y : np.ndarray
        1D signal of length N.
    invalid_intervals : np.ndarray
        array of half-open intervals
    x : np.ndarray | None
        Optional x-axis values of length N (e.g., time). If None, uses sample indices.
    ax : matplotlib.axes.Axes | None
        Optional axes to plot on. If None, creates a new figure/axes.
    signal_kwargs : dict | None
        Optional kwargs passed to ax.plot for the signal line.
    bad_color : str
        Color used for shading bad regions.
    bad_alpha : float
        Alpha for shading bad regions.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    if x is None:
        x = np.arange(len(y))
    else:
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")

    if ax is None:
        _, ax = plt.subplots(figsize=(20, 4))

    if signal_kwargs is None:
        signal_kwargs = {"color": "black", "linewidth": 0.2}

    ax.plot(x, y, **signal_kwargs)

    # Shade intervals
    for s, e in invalid_intervals:
        # noinspection PyTypeChecker
        ax.axvspan(x[s], x[e - 1], color=bad_color, alpha=bad_alpha, linewidth=1)

    ax.set_xlim(x[0], x[-1])
    return ax


def plot_signals_with_bad_regions(signals: ndarray, invalid_intervals: ndarray, start_dt: Timestamp):
    n_chn, n_samples = signals.shape
    #x = pd.date_range(start_dt, freq=Timedelta(seconds=1 / SAMPLING_FREQUENCY_HZ), periods=n_samples)
    fig, axs = plt.subplots(nrows=n_chn, ncols=1, sharex=True, sharey=True, figsize=(20, 8))
    for sig, ax in zip(signals, axs):
        plot_signal_with_bad_regions(sig, invalid_intervals,
                                     #x,
                                     ax=ax)
    return fig


def show_examples_per_ptnt(pdirs: list[PatientDir], n_examples_per_ptnt: int = 3):
    for pdir in pdirs:
        invalid_ivs_df = pd.read_pickle(pickle_path(pdir.invalid_edf_intervals))
        edfs = pd.read_pickle(pickle_path(pdir.edf_files_table))
        all_files = invalid_ivs_df['file_name'].unique()
        selected_files = np.random.choice(all_files, n_examples_per_ptnt, replace=False)

        for file in selected_files:
            file_rows = invalid_ivs_df[invalid_ivs_df['file_name'] == file]
            invalid_ivs = np.column_stack([file_rows['start_idx'], file_rows['end_idx']])

            edf_start_mtz = edfs.loc[edfs['file_name'] == file, 'start_mtz'].item()

            signals, sig_headers, header = highlevel.read_edf(str(pdir.edf_dir / file))

            # noinspection PyTypeChecker
            fig = plot_signals_with_bad_regions(signals, invalid_ivs, edf_start_mtz)
            ivs_str = ' '.join([str(iv) for iv in invalid_ivs])
            fig.suptitle(f"{file}\nInvalid intervals: {ivs_str}")
            # Show and wait until the window is closed to continue
            print(f'Showing fig for {file}')
            plt.show(block=True)
            plt.close('all')


def main():
    plt.ion()
    pdirs = PATHS.patient_dirs()
    random.shuffle(pdirs)
    show_examples_per_ptnt(pdirs)


if __name__ == '__main__': main()
