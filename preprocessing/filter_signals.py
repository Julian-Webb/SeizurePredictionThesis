import logging
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import ndarray, array
from numpy.lib.stride_tricks import sliding_window_view
from pandas import Timedelta
from pyedflib import highlevel

from config.constants import SAMPLING_FREQUENCY_HZ
from config.intervals import SEGMENT
from config.paths import PatientDir, PATHS
from utils.io import pickle_path, save_dataframe_multiformat
from utils.utils import timeit


def eeg_clipping_mask(signals: ndarray, min_amp: float, max_amp: float, absolute_tolerance=0.1):
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


def high_variance_mask(signals: ndarray, sfreq: float, window_sec=2.0, step_sec=0.5, var_threshold=20_000):
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


def mask2intervals(mask_bool: ndarray) -> ndarray:
    """
    Convert a 1D boolean mask (True=valid, False=invalid) into intervals with index.

    Intervals are half-open: [start_idx, end_idx).
    """
    padded = np.r_[False, mask_bool, False]
    changes = np.diff(padded.astype(np.int8))
    starts_idx = np.where(changes == 1)[0]  # False -> True
    ends_idx = np.where(changes == -1)[0]  # True -> False (exclusive)
    zipped = np.column_stack((starts_idx, ends_idx))
    return zipped


def intervals2mask(intervals: ndarray, n_samples: int) -> ndarray:
    """
    Convert half-open index intervals into a 1D boolean mask. The intervals will be True and the rest False.
    :param intervals: An array of intervals
    :param n_samples: The number of samples in the mask
    """
    mask = np.zeros(n_samples, dtype=bool)
    for s, e in intervals:
        mask[s:e] = True
    return mask


def merge_intervals_with_gap(intervals: ndarray, min_gap: int) -> ndarray:
    """
    Merge intervals if the gap between consecutive intervals is <= min_gap.
    Intervals are assumed half-open [start, end) (end is exclusive).
    """
    if intervals.size == 0:
        return np.empty((0, 2), dtype=intervals.dtype)

    merged = []
    cur_s, cur_e = intervals[0]

    for s, e in intervals[1:]:
        if s - cur_e <= min_gap:
            cur_e = max(cur_e, e)
        else:
            merged.append([cur_s, cur_e])
            cur_s, cur_e = s, e

    merged.append([cur_s, cur_e])
    return np.asarray(merged)


def invert_intervals(intervals: ndarray, n_samples: int) -> ndarray:
    """
    Invert intervals within [0, n_samples).
    Assumes intervals are [start, end) (end exclusive), non-overlapping, and sorted.
    """
    inverted = []
    prev_end = 0

    for start, end in intervals:
        if start > prev_end:
            inverted.append(array([prev_end, start]))
        prev_end = max(prev_end, end)

    if prev_end < n_samples:
        inverted.append(array([prev_end, n_samples]))

    return array(inverted)


def filter_edf(path: Path,
               lookahead_after_clipped_sec: float = 20.0,
               invalid_interval_padding_sec: float = 0.25,
               ):
    """
    Filter out problematic portions of the file:
    * Portions that are at the signal limits (clipped)
    * Artifacts that are presumably from connecting the device

    :param path: EDF path
    :param lookahead_after_clipped_sec: How far to look ahead for bad signals after clipped portions
    :param invalid_interval_padding_sec: How much padding to add around invalid intervals
    :return: dict with 'valid' and 'invalid' intervals (by index in file)
    """
    signals, sig_headers, header = highlevel.read_edf(str(path))
    n_samples = signals.shape[1]

    # Find intervals at the signal limit
    clipped_mask = eeg_clipping_mask(signals,
                                     min_amp=sig_headers[0]["physical_min"],
                                     max_amp=sig_headers[0]["physical_max"])
    # to mask to intervals and merge them
    clipped_ivs = mask2intervals(clipped_mask)
    clipped_ivs = merge_intervals_with_gap(clipped_ivs, min_gap=SEGMENT.n_samples)

    # Go through clipped intervals and filter out the additional garbage that can comes after them
    invalid_mask = intervals2mask(clipped_ivs, n_samples)
    lookahead = round(lookahead_after_clipped_sec * SAMPLING_FREQUENCY_HZ)

    # Check for high variance after the clipped intervals and start of file (end of 0)
    ends = np.r_[0, clipped_ivs[:, 1]]
    for e in ends:
        hvar_mask = high_variance_mask(signals[:, e: e + lookahead], SAMPLING_FREQUENCY_HZ)
        hvar_ivs = mask2intervals(hvar_mask)
        # Use only the first high variance interval and only if it starts after the clipped period
        hvar_offset = 0
        if hvar_ivs.size > 0:
            if hvar_ivs[0, 0] == 0:
                hvar_offset = hvar_ivs[0, 1]
            # else:
            #     logging.debug(
            #         f"After a clipped period, the high variance period doesn't start immediately after it for "
            #         f"{path.name}: {hvar_ivs=}"
            #     )
        invalid_mask[e: e + hvar_offset] = True

    # Turn these masks into intervals
    invalid_ivs = mask2intervals(invalid_mask)
    # Pad invalid intervals
    padding = round(invalid_interval_padding_sec * SAMPLING_FREQUENCY_HZ)
    invalid_ivs[:, 0] -= padding
    invalid_ivs[:, 1] += padding
    # Clip intervals to file start & end
    invalid_ivs = invalid_ivs.clip(0, n_samples)

    invalid_ivs = merge_intervals_with_gap(invalid_ivs, min_gap=SEGMENT.n_samples)

    valid_ivs = invert_intervals(invalid_ivs, n_samples)

    return {'valid': valid_ivs, 'invalid': invalid_ivs}


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

    def interval_rows_for_file(file_name: str, edf_start_mtz: pd.Timestamp, ivs_idx: ndarray, sfreq: float) -> \
            list[dict]:
        ivs_sec = ivs_idx / sfreq

        rows = []
        for (s_idx, e_idx), (s_sec, e_sec) in zip(ivs_idx, ivs_sec):
            rows.append({
                'file_name': file_name,
                'start_mtz': edf_start_mtz + Timedelta(seconds=s_sec),
                'end_mtz': edf_start_mtz + Timedelta(seconds=e_sec),
                'start_idx': s_idx, 'end_idx': e_idx,
            })
        return rows

    for edf in edfs.itertuples(index=False):
        logging.debug(f'Filtering EDF: {edf.file_name}')

        ivs = filter_edf(pdir.edf_dir / edf.file_name)

        valid_ivs.extend(interval_rows_for_file(edf.file_name, edf.start_mtz, ivs['valid'], SAMPLING_FREQUENCY_HZ))
        invalid_ivs.extend(interval_rows_for_file(edf.file_name, edf.start_mtz, ivs['invalid'], SAMPLING_FREQUENCY_HZ))

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


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    pdirs = PATHS.patient_dirs()
    # root = Path('/data/home/webb/UNEEG/datasets')
    # p1 = root / 'competition' / 'competition-2'
    # p2 = root / 'ultra2' / 'U002-DE01-07'
    # pdirs = [PatientDir(p) for p in [p1, p2]]

    filter_all_edfs(pdirs,
                    serial_processing=False
                    )


if __name__ == '__main__': main()
