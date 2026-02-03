import logging
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pyedflib import highlevel

from config.constants import SAMPLING_FREQUENCY_HZ
from config.paths import PatientDir, PATHS
from utils.io import pickle_path, save_dataframe_multiformat
from utils.utils import timeit


def eeg_clipping_mask(signals: np.ndarray, min_amp: float, max_amp: float, absolute_tolerance=0.1):
    """
    Returns a mask of where the signal is close to its minimum or maximum
    :param signals: (#channels, n_samples)
    :param min_amp:
    :param max_amp:
    :param absolute_tolerance: How much the signal can deviate from the min/max to still be excluded
    :return:
    """
    close_per_channel_mask = (
            np.isclose(signals, min_amp, atol=absolute_tolerance) |
            np.isclose(signals, max_amp, atol=absolute_tolerance)
    )

    close_mask = close_per_channel_mask.any(axis=0)
    return close_mask


def high_variance_mask(signals: np.ndarray, sfreq: float, window_sec=2.0, step_sec=0.5, var_threshold=40_000):
    """
    Returns a boolean mask of shape (n_samples)

    True  = high-variance artifact (any channel has high variance in a window covering that sample)
    False = variance looks normal
    """
    n_ch, n_samples = signals.shape
    win = max(1, int(round(window_sec * sfreq)))
    step = max(1, int(round(step_sec * sfreq)))

    if win > n_samples:
        return np.zeros(n_samples, dtype=bool)

    # Regular windows (only those that fully fit)
    windows = sliding_window_view(signals, win, axis=1)[:, ::step, :]  # (n_ch, n_windows, win)
    var = np.var(windows, axis=-1)  # (n_ch, n_windows)
    bad_windows = (var > var_threshold).any(axis=0)  # (n_windows,)

    starts = np.arange(0, n_samples - win + 1, step)  # (n_windows,)

    # Build per-sample mask by "painting" bad windows
    marks = np.zeros(n_samples, dtype=np.int32)
    if bad_windows.any():
        marks[starts[bad_windows]] = 1

    # Also consider the trailing region by forcing the last full window (ending at n_samples)
    tail_start = n_samples - win
    if starts.size == 0 or starts[-1] != tail_start:
        tail_var = np.var(signals[:, tail_start:tail_start + win], axis=-1)  # (n_ch,)
        tail_bad = bool((tail_var > var_threshold).any())
        if tail_bad:
            marks[tail_start] = 1

    coverage = np.convolve(marks, np.ones(win, dtype=np.int32), mode="full")[:n_samples]
    return coverage > 0


def mask_to_intervals(valid_mask: np.ndarray, sfreq: float):
    """
    Convert a 1D boolean mask (True=valid, False=invalid) into intervals with index and time.

    Intervals are half-open: [start_sec, end_sec) where end_sec is EXCLUSIVE.
    """
    m = np.asarray(valid_mask, dtype=bool)
    if m.size == 0:
        return [], []

    def _runs_to_intervals(mask_bool: np.ndarray):
        # pad with False so edges are detected as transitions
        padded = np.r_[False, mask_bool, False]
        changes = np.diff(padded.astype(np.int8))

        starts_idx = np.where(changes == 1)[0]  # False -> True
        ends_idx = np.where(changes == -1)[0]  # True -> False (exclusive)

        starts_time = pd.to_timedelta(starts_idx / sfreq, unit='s')
        ends_time = pd.to_timedelta(ends_idx / sfreq, unit='s')

        # noinspection PyTypeChecker
        return {'index': list(zip(starts_idx, ends_idx)), 'time': list(zip(starts_time, ends_time))}

    return {'valid': _runs_to_intervals(m), 'invalid': _runs_to_intervals(~m)}


def filter_edf(path: Path):
    """
    Filter out problematic portions of the file:
    * Artifacts at the start
    * Portions without a change / with the signal limits
    :param path:
    :return: valid_intervals, invalid_intervals
    """
    signals, sig_headers, header = highlevel.read_edf(str(path))

    clipping_mask = eeg_clipping_mask(signals,
                                      min_amp=sig_headers[0]["physical_min"],
                                      max_amp=sig_headers[0]["physical_max"])
    high_var_mask = high_variance_mask(signals, SAMPLING_FREQUENCY_HZ)
    bad_mask = clipping_mask | high_var_mask

    # Turn these masks into intervals
    ivs = mask_to_intervals(~bad_mask, SAMPLING_FREQUENCY_HZ)
    return ivs


@timeit
def filter_ptnt_edfs(pdir: PatientDir):
    """
    For a patient:
      - compute valid/invalid intervals per EDF
      - return two DataFrames (one row per interval) with:
          file_name, start_mtz, end_mtz, start_idx, end_idx
    """
    edfs = pd.read_pickle(pickle_path(pdir.edf_files_sheet))

    valid_ivs = []
    invalid_ivs = []

    def interval_rows_for_file(file_name: str, edf_start_mtz: pd.Timestamp, intervals: dict) -> list[dict]:
        rows = []
        for (s_idx, e_idx), (s_time, e_time) in zip(intervals['index'], intervals['time']):
            rows.append({
                'file_name': file_name,
                'start_mtz': edf_start_mtz + s_time,
                'end_mtz': edf_start_mtz + e_time,
                'start_idx': s_idx,
                'end_idx': e_idx,
            })
        return rows

    for edf in edfs.itertuples(index=False):
        logging.debug(f'Filtering EDF: {edf.file_name}')
        ivs = filter_edf(pdir.edf_dir / edf.file_name)
        valid_ivs.extend(interval_rows_for_file(edf.file_name, edf.start_mtz, ivs['valid']))
        invalid_ivs.extend(interval_rows_for_file(edf.file_name, edf.start_mtz, ivs['invalid']))

    return {'valid': pd.DataFrame(valid_ivs), 'invalid': pd.DataFrame(invalid_ivs)}


def _process_ptnt(pdir: PatientDir):
    logging.info(f'Filtering patient EDFs: {pdir.name}')
    ivs = filter_ptnt_edfs(pdir)
    save_dataframe_multiformat(ivs['valid'], pdir.valid_edf_intervals)
    save_dataframe_multiformat(ivs['invalid'], pdir.invalid_edf_intervals)
    logging.info(f'Done filtering patient EDFs: {pdir.name}')


def filter_all_edfs(pdirs: list[PatientDir], serial_processing: bool = False):
    if serial_processing:
        for pdir in pdirs:
            _process_ptnt(pdir)
    else:
        with multiprocessing.Pool() as pool:
            pool.map(_process_ptnt, pdirs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    pdirs = PATHS.patient_dirs()
    filter_all_edfs(pdirs,
                    serial_processing=False
                    )
