from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yasa
from pyedflib import EdfReader
from itertools import groupby

from config.constants import CHANNELS, N_CHANNELS, SAMPLING_FREQUENCY_HZ, SPECTRAL_BANDS
from config.intervals import SEGMENT
from config.paths import PatientDir

from statsmodels.tsa import stattools


def autocorrelation_function_width(sig: np.ndarray) -> int:
    """Computes the ACFW (autocorrelation function width) for a signal.
    The ACFW is the lag at which the autocorrelation is half its maximum value.
    :param sig: signal to compute ACFW for
    :return: ACFW width (lag)
    """
    # compute autocorrelation for all lags
    autocorr = stattools.acf(sig, nlags=len(sig), fft=True)
    # todo delete this after testing
    assert autocorr.max() == 1, "Autocorrelation max isn't 1"

    # Find the lag (index) where the ACF is closest to its half-max (=0.5)
    # The half-max is 0.5 because the maximum autocorrelation is 1
    # Subtracting 0.5 makes the values close to 0.5 become close to 0
    # Taking the absolute values makes values close to 0 be the minimal elements
    # Then, the argmin is taken to find the value closest to 0 (the half-max index/lag)
    lag = np.abs(autocorr - 0.5).argmin()
    return lag


@dataclass
class Features:
    """Represents the features for a segment"""
    variances: np.ndarray  # of the two channels
    acfws: list  # autocorrelation function width
    corrcoef: float  # scalar correlation between the channels
    bandpowers: pd.DataFrame  # bandpowers for each band and channel

    @classmethod
    def from_signals(cls, sigs: np.ndarray) -> 'Features':
        """Compute the features for a single segment.
        :param sigs: A dict with channel names as keys and a numpy array as values."""
        variances = np.var(sigs, axis=1)
        acfws = [autocorrelation_function_width(sig) for sig in sigs]

        # Correlation of signals
        corrcoef = np.corrcoef(sigs[0], sigs[1])[0, 1]

        # Band Power Spectrum
        # It says the length of the sliding window should be at least two times the inverse of the lowest frequency of interest
        win_sec = 2 / SPECTRAL_BANDS[0][0]
        # todo is this doing the right thing? (Also should the window be 'hamming', etc.)?
        bandpowers = yasa.bandpower(sigs, SAMPLING_FREQUENCY_HZ, CHANNELS, win_sec=win_sec, bands=SPECTRAL_BANDS)
        bandpowers.drop(['TotalAbsPow', 'FreqRes', 'Relative'], axis='columns', inplace=True)
        return cls(variances, acfws, corrcoef, bandpowers)

    def to_series(self) -> pd.Series:
        bandpowers = {}
        for i, ch in enumerate(self.bandpowers.index):
            for band in self.bandpowers.columns:
                bandpowers[f'ch{i}_{band}'] = self.bandpowers.loc[ch, band]

        return pd.Series(
            {'var0': self.variances[0], 'var1': self.variances[1], 'acfw0': self.acfws[0], 'acfw1': self.acfws[1],
             'corrcoef': self.corrcoef, **bandpowers}, name='values'
        )

    def to_list(self) -> list:
        """Returns the features in a fixed order as a 1D list which can be used as input to a neural network."""
        return [*self.variances, *self.acfws, self.corrcoef, *self.bandpowers.to_numpy().flatten()]


def extract_ptnt_features(ptnt_dir: PatientDir):
    """Extract the features for all existing segments of a patient."""
    segs = pd.read_csv(ptnt_dir.segments_table)
    segs = segs[segs['exists']]

    # Iterate through the existing segments based on their file
    for file_path, segs_group in groupby(segs.iterrows(), key=lambda row: row[1]['file']):
        # todo could potentially make it faster if I extract all segs from the same file and reshape
        edf = EdfReader(str(ptnt_dir.edf_dir / file_path))

        for _, seg in segs_group:
            # todo make sure that channel 0 and 1 are always the same channel
            sigs = np.zeros((N_CHANNELS, SEGMENT.n_samples))
            for i in range(N_CHANNELS):
                sigs[i] = edf.readSignal(chn=i, start=seg['start_index'], n=SEGMENT.n_samples, digital=False)
            features = Features.from_signals(sigs)
            ser = features.to_series()
            lst = features.to_list()
            breakpoint()

        edf.close()


def extract_features(ptnt_dirs: List[PatientDir]):
    """Extract the features for the segments of a patient."""
    for ptnt_dir in ptnt_dirs:
        extract_ptnt_features(ptnt_dir)


if __name__ == '__main__':
    ptnt_dir = PatientDir(Path('/Users/julian/Developer/SeizurePredictionData/20240201_UNEEG_ForMayo/ptnt1'))
    extract_ptnt_features(ptnt_dir)
