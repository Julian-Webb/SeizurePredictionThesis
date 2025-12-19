import logging
import time
# noinspection PyUnusedImports
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from numpy import ndarray
from pyedflib import EdfReader
from scipy.integrate import simpson
from scipy.signal import welch
from statsmodels.tsa import stattools

from config.constants import N_CHANNELS, SAMPLING_FREQUENCY_HZ, SPECTRAL_BANDS
from config.intervals import SEGMENT
from config.paths import PatientDir, PATHS
from utils.io import pickle_path, save_dataframe_multiformat

# How many files to process per file batch
FILE_BATCH_SIZE: int = 128


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


class FeaturesForFile:
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
    ORDERED_FEATURE_NAMES = (
        'corrcoefs',
        'acfw_D', 'acfw_P',
        'var_D', 'var_P',
        'Delta_D', 'Theta_D', 'Alpha_D', 'Beta_D', 'Gamma_D',
        'Delta_P', 'Theta_P', 'Alpha_P', 'Beta_P', 'Gamma_P',
    )

    def __init__(self, file_path: Path, first_idx: int, n_segs: int):
        """
        Extract features for all segments in an EDF file as a batch-operation.
        :param first_idx: The index of the first segment start in the EDF file.
        """
        st = time.perf_counter()

        # Read signals and segment them
        # Segmented signals with shape [#segments, #channels, #samples per seg]
        ss = _load_segmented_sigs(file_path, first_idx, n_segs)
        # Compute Features
        self.corrcoefs = np.expand_dims(
            [np.corrcoef(ss[seg, 0, :], ss[seg, 1, :])[0, 1] for seg in range(n_segs)],
            axis=1)
        self.acfws = np.apply_along_axis(autocorrelation_function_width, axis=-1, arr=ss)
        self.variances = ss.var(axis=-1)
        self.bandpowers = bandpowers_vectorized(ss, SAMPLING_FREQUENCY_HZ, SPECTRAL_BANDS)

        logging.debug(f"Features extracted in {time.perf_counter() - st:.3f} sec for : {file_path.name}")

    @classmethod
    def init_to_array(cls, file_path: Path, first_idx: int, n_segs: int):
        """Initialize and directly return the features as an array"""
        return cls(file_path, first_idx, n_segs).to_array()

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
            index=self.ORDERED_FEATURE_NAMES
        )


def _load_segmented_sigs(file_path: Path, first_idx: int, n_segs: int) -> ndarray:
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


@dataclass
class FileInfo:
    file_path: Path
    first_idx: int
    n_segs: int


def extract_file_batch_features(files_infos: List[FileInfo]):
    logging.info(f"Batch-extracting features for {len(files_infos)} files")
    st = time.perf_counter()

    file_features = {}
    for f in files_infos:
        file_name = f.file_path.name
        # Compute Features
        file_features[file_name] = FeaturesForFile.init_to_array(f.file_path, f.first_idx, f.n_segs)

    logging.info(f"Batch features extracted in {time.perf_counter() - st:.3f} sec for : {len(files_infos)} files")
    return file_features


def extract_ptnt_features(ptnt_dir: PatientDir):
    logging.info(f"Extracting features for {ptnt_dir.name}")
    start_time = time.perf_counter()
    segs = pd.read_pickle(pickle_path(ptnt_dir.segments_table))

    # Iterate through the existing segments based on their file
    # Note: There are typically around 500-2000 EDFs per patient
    file_names = segs['file'].dropna().unique()

    # --- Serial Processing --------------------------------------------------------------------------------------------
    # for file_name in file_names:
    #     # Auxiliary variables
    #     file_mask = segs['file'] == file_name
    #     file_segs = segs[file_mask]
    #     n_segs = file_segs.shape[0]
    #     first_idx = file_segs.iloc[0]['start_index']
    #     file_path = ptnt_dir.edf_dir / file_name
    #     # Compute Features and update segs
    #     segs.loc[file_mask, FeaturesForFile.ORDERED_FEATURE_NAMES] = \
    #         FeaturesForFile.init_to_array(file_path, first_idx, n_segs)
    # ------------------------------------------------------------------------------------------------------------------

    # --- Parallel Processing v2 ---------------------------------------------------------------------------------------
    # Make file info dict
    files_infos = {}
    file_masks = {}
    for file_name in file_names:
        file_path = ptnt_dir.edf_dir / file_name
        file_mask = segs['file'] == file_name
        file_segs = segs[file_mask]
        first_idx = file_segs.iloc[0]['start_index']
        n_segs = file_segs.shape[0]

        files_infos[file_name] = FileInfo(file_path, first_idx, n_segs)
        file_masks[file_name] = file_mask

    # Make batches
    files_infos_list = [files_infos[name] for name in file_names]
    batches = [files_infos_list[i: i + FILE_BATCH_SIZE] for i in range(0, len(files_infos_list), FILE_BATCH_SIZE)]
    logging.info(f"Created {len(batches)} batches for patient {ptnt_dir.name}")

    # Compute Features in batches in parallel
    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(extract_file_batch_features, batch) for batch in batches]

        # Update segs
        for future in as_completed(futures):
            batch_res = future.result()
            for file_name, features_arr in batch_res.items():
                segs.loc[file_masks[file_name], FeaturesForFile.ORDERED_FEATURE_NAMES] = features_arr
    # ------------------------------------------------------------------------------------------------------------------

    save_dataframe_multiformat(segs, ptnt_dir.segments_table)
    logging.info(f"Features extracted for {ptnt_dir.name} in {time.perf_counter() - start_time:.3f} sec.")


def extract_features(ptnt_dirs: List[PatientDir]):
    """Extract the features for the segments of a patient."""
    st = time.perf_counter()

    # --- Serial Processing --------------------------------------------------------------------------------------------
    # for ptnt_dir in ptnt_dirs:
    #     extract_ptnt_features(ptnt_dir)
    # ------------------------------------------------------------------------------------------------------------------

    # --- Parallel Processing ------------------------------------------------------------------------------------------
    with ProcessPoolExecutor() as exe:
        exe.map(extract_ptnt_features, ptnt_dirs)
    # ------------------------------------------------------------------------------------------------------------------

    logging.info(f"[TIMING] Extracted features in {time.perf_counter() - st:.3f} sec")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s', force=True)

    # pdir = PatientDir(PATHS.for_mayo_dir / 'B52K3P3G')
    # extract_features([pdir])

    extract_features(PATHS.patient_dirs())
