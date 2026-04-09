import logging
import math
import multiprocessing
from math import ceil, floor
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pandas import Series, Timestamp, DataFrame
from pandas.api.types import is_numeric_dtype, is_timedelta64_dtype

from config import PATHS, PatientDir, save_dataframe_multiformat
from config.intervals import INTERICTAL
from utils.utils import autofit_excel_columns

SPLITS = ('train', 'test', 'overall')
# How many of the dataset's seizures should be allocated to training
INTENDED_RATIO_OF_SZRS_FOR_TRAINING = 0.6
# How many of the valid clips should be allocated to each partition at minimum.
MIN_RATIO_VALID_CLIPS_TEST = 0.3
MIN_RATIO_VALID_CLIPS_TRAIN = 0.4
MAX_RATIO_VALID_CLIPS_TRAIN = 1 - MIN_RATIO_VALID_CLIPS_TEST

MANUAL_TEST_START_OVERRIDES = {
    # has a huge gap in his recordings where seizures were excluded
    'U002-DE01-03': Timestamp('2021-11-01 00:00:00'),
}


def partition_dataframe(
        df: DataFrame,
        pdir: PatientDir = None,
        test_start_mtz: Timestamp = None,
        start_col: str = 'start_mtz'
) -> dict[str, DataFrame]:
    """
    Partition a DataFrame into train and test sets based on the start time of the test set.

    Parameters
    ----------
    df
    pdir
    test_start_mtz
        The start time of the test set. If None, it will be read from the patient's partition file.
    start_col

    Returns
    -------
    dict with keys 'train' and 'test'
    """
    if test_start_mtz is None:
        if pdir is None:
            raise ValueError('Either pdir or test_start_mtz must be specified.')
        # noinspection PyTypeChecker
        test_start_mtz: Timestamp = pd.read_pickle(pdir.dataset_partition.pickle).loc['test', 'start_mtz']
    train = df[df[start_col] < test_start_mtz]
    test = df[df[start_col] >= test_start_mtz]
    return {'train': train, 'test': test}


def _round_test_start_to_clip(ts_approx: Timestamp, clips: DataFrame):
    # "Round" the approximate start to nearest clip
    test_clips = clips[ts_approx < clips['end_mtz']]
    ts_exact = test_clips['start_mtz'].iloc[0]
    return ts_exact


def _find_closest_interictal_clip(clips: DataFrame, search_forwards: bool, first_test_idx: int, ptnt: str):
    """Find the closest interictal clip, searching forwards or backwards"""
    is_interictal = clips['types'].map(lambda t: t == [INTERICTAL.label])
    if search_forwards:
        candidates = is_interictal.loc[first_test_idx:]
    else:
        candidates = is_interictal.loc[:first_test_idx].iloc[::-1]
    # Finds the first true value - looking either forwards or backwards
    assert candidates.any(), f'[{ptnt}] Could not find an interictal clip to partition on.'
    first_i = candidates.idxmax()
    return clips.loc[first_i]


def find_test_start_for_ptnt(clips: DataFrame, valid_szrs: DataFrame, ptnt: str) -> Timestamp:
    """
    Computes the time point where the test set starts.

    Returns
    -------
    test_start_exact: Timestamp
    """
    # ts = test start
    # vcs = valid clips
    if ptnt in MANUAL_TEST_START_OVERRIDES:
        return MANUAL_TEST_START_OVERRIDES[ptnt]

    # ---- Calculate the approximate time point where test set starts
    szrs = valid_szrs['start_mtz'].values
    n_train_szrs = math.floor(len(szrs) * INTENDED_RATIO_OF_SZRS_FOR_TRAINING)

    last_train_szr = szrs[n_train_szrs - 1]  # -1 because of 0-indexing
    first_test_szr = szrs[n_train_szrs]
    # Set the approximate start of the test set between the first test szr and the previous one
    ts_approx = last_train_szr + ((first_test_szr - last_train_szr) / 2)  # is contained in first test clip
    ts_rounded = _round_test_start_to_clip(ts_approx, clips)

    # ---- Bound the test start to the valid clip (vc) limits
    vcs = clips[clips['valid']]
    n_vcs = len(vcs)
    vcs_train = vcs[vcs['start_mtz'] < ts_rounded]
    train_ratio = len(vcs_train) / n_vcs

    min_ratio, max_ratio = MIN_RATIO_VALID_CLIPS_TRAIN, MAX_RATIO_VALID_CLIPS_TRAIN  # alias
    # interictal_search_forwards: Whether to search forwards or backwards for the interictal clip later
    if train_ratio < min_ratio:
        n_vcs_train = ceil(n_vcs * min_ratio)
        interictal_search_forwards = True
    elif train_ratio > max_ratio:
        n_vcs_train = floor(n_vcs * max_ratio)
        interictal_search_forwards = False
    else:
        n_vcs_train = len(vcs_train)
        # Search forwards if we're closer to the minimum ratio
        interictal_search_forwards = abs(train_ratio - min_ratio) < abs(train_ratio - max_ratio)

    # Note: because of 0-indexing, the number of valid clips in train corresponds to the first clip in test
    first_test_clip = vcs.iloc[n_vcs_train]
    first_test_idx = first_test_clip.name  # Note: vcs and clips share their index

    # ---- Ensure the partition (test start) falls on an interictal clip
    if first_test_clip['types'] != [INTERICTAL.label]:
        # search direction is based on whether there was too little data in the train or test set
        first_test_clip = _find_closest_interictal_clip(clips, interictal_search_forwards, first_test_idx, ptnt)

    ts_final = first_test_clip['start_mtz']
    return ts_final


def _check_partition(test_start: Timestamp, clips: DataFrame, valid_szrs: DataFrame, ptnt: str):
    vcs = clips[clips['valid']]
    vcs_train = partition_dataframe(vcs, test_start_mtz=test_start)['train']
    vc_ratio_train = len(vcs_train) / len(vcs)

    if vc_ratio_train < MIN_RATIO_VALID_CLIPS_TRAIN:
        logging.error(f'[{ptnt}] x The ratio of training clips is too low: {vc_ratio_train:.2f}.')
    elif vc_ratio_train > MAX_RATIO_VALID_CLIPS_TRAIN:
        logging.error(f'[{ptnt}] x The ratio of training clips is too high: {vc_ratio_train:.2f}.')

    szrs = valid_szrs['start_mtz'].values
    szrs_train = szrs[szrs < test_start]
    szr_ratio_train = len(szrs_train) / len(szrs)
    if not np.isclose(szr_ratio_train, INTENDED_RATIO_OF_SZRS_FOR_TRAINING, atol=0.10):
        logging.warning(f'[{ptnt}] The ratio of training seizures is not as expected: {szr_ratio_train:.2f}.')


def partition_info_for_ptnt(
        test_start: Timestamp,
        clips: DataFrame,
        segs: DataFrame,
        all_szrs: DataFrame,
        valid_szrs: DataFrame,
) -> DataFrame:
    clips = clips.copy()
    # ---- Create output DataFrame with rich information
    o = pd.DataFrame(index=Series(['train', 'test', 'overall'], name='partition'))
    o['split_rule_idx'] = ['idx < first test idx', 'idx >= first test idx', '']
    o['split_rule_timestamp'] = ['ts < test start_mtz', 'ts >= test start_mtz', '']

    # Monitoring Timespan
    first_start = clips['start_mtz'].iloc[0]
    last_end = clips['end_mtz'].iloc[-1]

    o['start_mtz'] = [first_start, test_start, first_start]
    o['end_mtz'] = [test_start, last_end, last_end]
    o['monitoring_timespan'] = o['end_mtz'] - o['start_mtz']

    # Duration of valid clips ≈ Recording Duration
    clips['dur'] = clips['end_mtz'] - clips['start_mtz']
    vclips = clips[clips['valid']]
    train_c = vclips[vclips['start_mtz'] < test_start]
    test_c = vclips[vclips['start_mtz'] >= test_start]
    o['valid_clips_duration'] = [train_c['dur'].sum(), test_c['dur'].sum(), vclips['dur'].sum()]

    # Index Rows
    def _add_idx_rows(df: DataFrame, tag: str):
        train = df[df['start_mtz'] < test_start]
        test = df[df['start_mtz'] >= test_start]
        n_train, n_test, n_total = len(train), len(test), len(df)
        o[f'n_{tag}'] = [n_train, n_test, n_total]
        o[f'ratio_{tag}'] = [n_train / n_total, n_test / n_total, 1]

    _add_idx_rows(vclips, 'valid_clips')
    _add_idx_rows(segs[segs['exists']], 'existing_segs')
    _add_idx_rows(all_szrs, 'all_szrs')
    _add_idx_rows(valid_szrs, 'valid_szrs')

    return o


# noinspection PyTypeChecker
def find_split_for_pdir(pdir: PatientDir):
    # Compute test start
    clips = pd.read_pickle(pdir.clips_table.pickle)
    valid_szrs = pd.read_pickle(pdir.valid_szr_starts_file.pickle)
    test_start = find_test_start_for_ptnt(clips, valid_szrs, pdir.name)

    _check_partition(test_start, clips, valid_szrs, pdir.name)

    # Generate partition info
    segs = pd.read_pickle(pdir.segments_table.pickle)
    all_szrs = pd.read_pickle(pdir.all_szr_starts_file.pickle)
    infos = partition_info_for_ptnt(test_start, clips, segs, all_szrs, valid_szrs)
    save_dataframe_multiformat(infos, pdir.dataset_partition, save_index=True)


def combine_partition_info_for_pdirs(pdirs: list[PatientDir]):
    metrics = [
        'ratio_valid_szrs', 'ratio_valid_clips', 'n_valid_szrs', 'n_valid_clips', 'valid_clips_duration',
        'monitoring_timespan', 'start_mtz', 'end_mtz', 'n_existing_segs', 'ratio_existing_segs',
        'n_all_szrs', 'ratio_all_szrs',
    ]

    info_per_ptnt = {pdir.name: pd.read_pickle(pdir.dataset_partition.pickle)[metrics] for pdir in pdirs}
    info: DataFrame = pd.concat(info_per_ptnt, names=['patient'])
    # Put FAKE patients at the end
    info = info.sort_index(level=0, sort_remaining=False, key=lambda s: s.str.contains('FAKE', na=False))
    return info


def _save_styled_partition_info_xlsx(df: DataFrame, out_path: Path):
    ratio_cols = [c for c in df.columns if str(c).startswith('ratio_')]
    other_cols = [c for c in df.columns if c not in ratio_cols]

    # Use only train/test rows to define gradient bounds for non-ratio columns.
    train_test_mask = df.index.get_level_values(-1).isin(['train', 'test'])

    styler = df.style

    if ratio_cols:
        # Ratios have a global, fixed meaning: 0=bad, 1=good.
        styler = styler.background_gradient(cmap='RdYlGn', subset=ratio_cols, axis=0, vmin=0, vmax=1)

    for col in other_cols:
        s = df[col]
        if is_numeric_dtype(s):
            gmap = s.astype(float)
        elif is_timedelta64_dtype(s):
            gmap = s.dt.total_seconds()
        else:
            continue

        styler = styler.background_gradient(cmap='RdYlGn', subset=[col], axis=0, gmap=gmap, vmin=0,
                                            vmax=gmap[train_test_mask].max())

    styler.to_excel(out_path)


def partition_for_pdirs(pdirs: List[PatientDir], serial_processing: bool = False):
    logging.info(f'🎬 Computing partitions for {len(pdirs)} patients.')
    if serial_processing:
        for pdir in pdirs:
            find_split_for_pdir(pdir)
    else:
        with multiprocessing.Pool() as pool:
            pool.map(find_split_for_pdir, pdirs)

    partition_info = combine_partition_info_for_pdirs(pdirs)
    _save_styled_partition_info_xlsx(partition_info, PATHS.partition_info_table.xlsx)
    autofit_excel_columns(PATHS.partition_info_table.xlsx)
    logging.info(f'✅ Completed partitions for {len(pdirs)} patients.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    pdirs_ = PATHS.patient_dirs()
    partition_for_pdirs(pdirs_, serial_processing=False)
