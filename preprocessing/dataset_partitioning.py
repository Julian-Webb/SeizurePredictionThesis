import multiprocessing
from typing import List

import numpy as np
import pandas as pd
from pandas import Series, Timestamp, Timedelta, DataFrame

from config import PATHS, PatientDir, save_dataframe_multiformat
from config.constants import RATIO_OF_TIMESPAN_FOR_TRAINING
from config.intervals import INTERICTAL, CLIP


def find_split_for_ptnt(clips: DataFrame, segs: DataFrame, all_szrs: DataFrame, valid_szrs: DataFrame):
    """
    Computes the time point where the test set starts and builds a DataFrame with information about the partition based
    on that.

    Returns
    -------
    test_start_exact: Timestamp, o: DataFrame with information about the split.
    """
    # ---- Calculate the time point where test set starts
    first_start = clips['start_mtz'].iloc[0]
    last_end = clips['end_mtz'].iloc[-1]
    timespan: Timedelta = last_end - first_start
    # This time point will overlap with the first test clip
    test_start_approx: Timestamp = first_start + timespan * RATIO_OF_TIMESPAN_FOR_TRAINING

    # "Round" the approximate start to nearest clip
    test_clips = clips[test_start_approx < clips['end_mtz']]
    test_start_exact = test_clips['start_mtz'].iloc[0]

    # Sanity checks
    assert np.abs(test_start_approx - test_start_exact) < CLIP.exact_dur, \
        'The approximate test start was rounded too much.'
    first_test_clip: Series = test_clips.iloc[0]
    first_test_seg: Series = segs.iloc[first_test_clip['start_seg']]
    if first_test_seg['type'] != INTERICTAL.label:
        raise NotImplementedError("The first test segment isn't interictal.")
    assert first_test_clip['start_mtz'] == first_test_seg['start_mtz']

    # ---- Create output DataFrame with rich information
    o = pd.DataFrame(columns=['train', 'test', 'overall'])
    o.loc['split_rule_idx'] = ['idx < first test idx', 'idx >= first test idx', '']
    o.loc['split_rule_timestamp'] = ['ts < test start_mtz', 'ts >= test start_mtz', '']

    o.loc['start_mtz'] = [first_start, test_start_exact, first_start]
    o.loc['end_mtz'] = [test_start_exact, last_end, last_end]
    o.loc['duration'] = o.loc['end_mtz'] - o.loc['start_mtz']

    # Sanity checks
    assert o.loc['duration', 'overall'] == timespan
    assert np.isclose(o.loc['duration', 'train'] / timespan, RATIO_OF_TIMESPAN_FOR_TRAINING, atol=0.01)
    assert np.isclose(o.loc['duration', 'test'] / timespan, 1 - RATIO_OF_TIMESPAN_FOR_TRAINING, atol=0.01)

    def _add_idx_rows(df: DataFrame, tag: str):
        train = df[df['start_mtz'] < test_start_exact]
        test = df[df['start_mtz'] >= test_start_exact]
        o.loc[f'first_idx_{tag}'] = [train.index[0], test.index[0], df.index[0]]
        o.loc[f'last_idx_{tag}'] = [train.index[-1], test.index[-1], df.index[-1]]
        o.loc[f'n_{tag}'] = [len(train), len(test), len(df)]

    _add_idx_rows(clips, 'clips')
    _add_idx_rows(segs, 'segs')
    _add_idx_rows(all_szrs, 'all_szrs')
    _add_idx_rows(valid_szrs, 'valid_szrs')

    return test_start_exact, o


def find_split_for_pdir(pdir: PatientDir):
    segs = pd.read_pickle(pdir.segments_table.pickle)
    clips = pd.read_pickle(pdir.clips_table.pickle)
    all_szrs = pd.read_pickle(pdir.all_szr_starts_file.pickle)
    valid_szrs = pd.read_pickle(pdir.valid_szr_starts_file.pickle)

    _, infos = find_split_for_ptnt(clips, segs, all_szrs, valid_szrs)
    save_dataframe_multiformat(infos, pdir.dataset_partition, save_index=True)


def find_splits_for_pdirs(pdirs: List[PatientDir], serial_processing: bool = False):
    if serial_processing:
        for pdir in pdirs:
            find_split_for_pdir(pdir)
    else:
        with multiprocessing.Pool() as pool:
            pool.map(find_split_for_pdir, pdirs)


if __name__ == '__main__':
    pdirs_ = PATHS.patient_dirs()
    find_splits_for_pdirs(pdirs_, serial_processing=True)
