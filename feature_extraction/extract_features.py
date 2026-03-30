import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import Series, DataFrame
from scipy.integrate import simpson
from scipy.signal import welch
from statsmodels.tsa import stattools

from config import PatientDir, PATHS
from config import save_dataframe_multiformat
from config.constants import SAMPLING_FREQUENCY_HZ, SPECTRAL_BANDS
from utils.edf_utils import load_segmented_sigs
from utils.utils import timeit


def autocorrelation_function_width(sig: ndarray) -> int:
    """Computes the ACFW (autocorrelation function width) for a signal.
    The ACFW is the lag at which the autocorrelation is half its maximum value.
    :param sig: signal to compute ACFW for
    :return: ACFW width (lag)
    """
    # compute autocorrelation for all lags
    autocorr = stattools.acf(sig, nlags=len(sig), fft=True)
    # assert autocorr.max() == 1, "Autocorrelation max isn't 1"

    # Find the lag (index) where the ACF is closest to its half-max (=0.5).
    # The half-max is 0.5 because the maximum autocorrelation is 1.
    # Subtracting 0.5 makes the values close to 0.5 become close to 0.
    # Taking the absolute values makes values close to 0 be the minimal elements.
    # Then, the argmin is taken to find the value closest to 0 (the half-max index/lag).
    lag = np.abs(autocorr - 0.5).argmin()
    return lag


def bandpowers_vectorized(segmented_sigs: ndarray, sfreq: float, bands: dict):
    """
    Compute the average, absolute bandpower for the signal of each segment and channel for the specified bands.
    :param segmented_sigs: array with shape [#segments, #channels, #samples per segment]
    :param sfreq: sampling frequency.
    :param bands: frequency bands in ascending order of frequency
    :return:
    """
    # Compute the window length for Welch
    lowest_band_freq = list(bands.values())[0][0]
    win_sec = 2 / lowest_band_freq
    # How many samples (indices) per Welch window
    win_n_idx = win_sec * sfreq

    # Compute Welch power spectrum density (psd) for all segments
    freqs, psd = welch(segmented_sigs, sfreq,
                       axis=-1,  # compute per segment
                       nperseg=win_n_idx,
                       noverlap=win_n_idx // 2,  # 50% overlap
                       window='hann',  # tapering
                       detrend='constant',  # Removes the mean which changes due to EEG drift
                       scaling='density',  # Power normalized by the width of the frequency bin (Power / Hz)
                       )
    freq_resolution = freqs[1] - freqs[0]

    # Compute spectral band powers
    n_segs, n_chn, _ = segmented_sigs.shape
    bandpowers = np.empty((n_segs, n_chn, len(bands)))
    for b_i, (low, high) in enumerate(bands.values()):
        band_mask = (low <= freqs) & (freqs <= high)
        # Integral approximation of the spectrum using Simpson's rule
        bandpowers[:, :, b_i] = simpson(psd[:, :, band_mask], dx=freq_resolution, axis=-1)
    return bandpowers


class FeatureNames:
    CORRCOEF = ['corrcoef']
    ACFW = ['acfw_D', 'acfw_P']
    VAR = ['var_D', 'var_P']
    BANDPOWERS = [
        'Delta_D', 'Theta_D', 'Alpha_D', 'Beta_D', 'Gamma_D',
        'Delta_P', 'Theta_P', 'Alpha_P', 'Beta_P', 'Gamma_P',
    ]
    ALL_ORDERED = CORRCOEF + ACFW + VAR + BANDPOWERS
    ENSEMBLE = CORRCOEF + BANDPOWERS  # Features for the ensemble model
    CYCLES = ACFW + VAR


class Features:
    """
    Represents the features for multiple segments.
    Features per segment:
    ---------------------
    Scalar correlation between the channels
    corrcoefs: ndarray. shape = (#segs, 1)

    Autocorrelation function width of each channel
    acfws: ndarray. shape = (#segs, #chn)

    Variances of the channels
    variances: ndarray. shape = (#seg, #chn)

    Bandpowers for each band and channel:
    bandpowers: ndarray. shape = (#segs, #chn, #bands)
    """

    def __init__(self, file_path: Path, start_index: int, n_segs: int):
        """
        Extract features for all segments in a continuous chunk of an EDF file.
        :param start_index: The index of the first segment's start in the EDF file.
        """
        st = time.perf_counter()

        # Read signals and segment them
        # Segmented signals with shape [#segments, #channels, #samples per seg]
        ss = load_segmented_sigs(file_path, start_index, n_segs)
        # Compute Features
        self.corrcoefs = np.expand_dims(
            [np.corrcoef(ss[seg, 0, :], ss[seg, 1, :])[0, 1] for seg in range(n_segs)],
            axis=1)
        self.acfws = np.apply_along_axis(autocorrelation_function_width, axis=-1, arr=ss)
        self.variances = ss.var(axis=-1)
        self.bandpowers = bandpowers_vectorized(ss, SAMPLING_FREQUENCY_HZ, SPECTRAL_BANDS)

        logging.debug(f"Features extracted in {time.perf_counter() - st:.3f} sec for : {file_path.name}")

    def to_array(self) -> ndarray:
        """
        Returns the features as a 2D array of shape (n_segs, n_features).
        n_features = 15.
        The order of features per segment is:
        0: correlation coefficient
        1-2: autocorrelation function width per channel
        3-4: variances per channel
        5-9: bandpowers of first channel
        10-14: bandpowers of second channel
        """
        # Flatten the 3rd dimension of bandpowers
        n_segs = self.bandpowers.shape[0]
        bps_flat = self.bandpowers.reshape(n_segs, -1)
        return np.hstack([
            self.corrcoefs,
            self.acfws,
            self.variances,
            bps_flat
        ])

    def to_series_for_seg(self, seg_idx: int) -> pd.Series:
        return pd.Series(
            self.to_array()[seg_idx],
            index=FeatureNames.ALL_ORDERED
        )


def _extract_chunk_features(chunk_info: np.ndarray) -> Tuple[int, ndarray]:
    """
    Extract features for one continuous chunk of a file and return (chunk_id, features).
    Parameters
    ----------
    chunk_info: np.ndarray
        A tuple with the chunk_id, file_path, start_index, and number of segments in the chunk.
    """
    chunk_id, start_index, n_segs, file_path = chunk_info
    logging.debug("%s, %s", chunk_id, file_path.name)
    arr =  Features(file_path, start_index, n_segs).to_array()
    return chunk_id, arr


def iter_feature_results(chunk_infos: DataFrame, serial_processing: bool) -> List[Tuple[int, ndarray]]:
    """Yield (chunk_id, feature_array) for each continuous chunk.
    Parameters
    ----------
    chunk_infos: DataFrame
        Chunk information per-row with columns: 'chunk_id', 'start_index', 'n_segs', 'file_path',
    """
    # Change to numpy so that it can be pickled for multiprocessing and iterate through the rows
    chunk_infos_numpy = chunk_infos.to_numpy() # shape = (n_chunks, 4)

    if serial_processing:
        return [_extract_chunk_features(c) for c in chunk_infos_numpy]
    else:
        with ProcessPoolExecutor() as exe:
            return list(exe.map(_extract_chunk_features, chunk_infos_numpy, chunksize=512))


def find_continuous_file_chunk_id_for_segs(file_col: Series):
    """
    Find continuous segments that belong to the same file (without NA gaps).
    :param file_col: segs['file']
    :return:
    """
    is_real = file_col.notna()
    prev_is_real = is_real.shift(fill_value=False)
    chunk_start = is_real & ((~prev_is_real) | (file_col != file_col.shift()))
    chunk_id = chunk_start.cumsum()  # all rows get run ids
    continuous_file_chunk_id = chunk_id.where(is_real)  # gaps -> NaN
    continuous_file_chunk_id.name = 'chunk_id'
    return continuous_file_chunk_id


def extract_ptnt_features(segs: DataFrame, edf_dir: Path, serial_processing: bool = False):
    """
    Parameters
    ----------
    segs: DataFrame
        pd.read_pickle(pdir.segments_table.pickle)
    edf_dir: Path
    serial_processing: bool

    Returns
    -------
    segs: DataFrame
        The original DataFrame with the new features.
    """
    continuous_file_chunk_id = find_continuous_file_chunk_id_for_segs(segs['file'])

    # Extract only necessary information to pass to workers
    segs_chunked = segs.groupby(continuous_file_chunk_id, sort=False, dropna=True)
    first_seg_in_chunk = segs_chunked.first()

    # A chunk per row (index: chunk_id)
    chunk_infos = DataFrame({
        'chunk_id': first_seg_in_chunk.index,
        'start_index': first_seg_in_chunk['start_index'],
        'n_segs': segs_chunked.size(),
        'file_path': edf_dir / first_seg_in_chunk['file'],
    })

    # Compute Features and Assign to segs DataFrame
    segs.loc[:, FeatureNames.ALL_ORDERED] = np.nan
    chunk_features = iter_feature_results(chunk_infos, serial_processing)
    for chunk_id, features_arr in chunk_features:
        indices = segs_chunked.indices[chunk_id]
        segs.loc[indices, FeatureNames.ALL_ORDERED] = features_arr

    return segs


@timeit(kwarg_names=['pdir'])
def extract_ptnt_features_and_save_from_pdir(pdir: PatientDir, serial_processing: bool = False):
    logging.info(f"[{pdir.name}] Extracting features...")
    segs = pd.read_pickle(pdir.segments_table.pickle)
    segs = extract_ptnt_features(segs, pdir.edf_dir, serial_processing)
    save_dataframe_multiformat(segs, pdir.segments_table)
    logging.info(f"[{pdir.name}] Finished feature Extraction.")


@timeit
def run_feature_extraction(pdirs: List[PatientDir], serial_processing: bool = False):
    """Extract the features for the segments of a patient."""
    if serial_processing:
        for pdir in pdirs:
            extract_ptnt_features_and_save_from_pdir(pdir, serial_processing)
    else:
        with ProcessPoolExecutor() as exe:
            list(exe.map(extract_ptnt_features_and_save_from_pdir, pdirs))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s', force=True)
    run_feature_extraction(PATHS.patient_dirs(), serial_processing=False)
