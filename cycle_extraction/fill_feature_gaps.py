"""
Fill gaps in the features during missing recording intervals of patients.
"""

import multiprocessing
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Timedelta

from config import PATHS, PatientDir, save_dataframe_multiformat
from config.constants import LONG_GAP_MIN_DURATION_FOR_FEATURE_FILLING
from config.intervals import SEGMENT
from feature_extraction.extract_features import FeatureNames


def detect_contiguous_gaps(exists: Series) -> DataFrame:
    """Find contiguous gaps where exists is False; index must increase by 1."""
    idx_diffs = np.diff(exists.index.to_numpy())
    if len(idx_diffs) > 0 and not np.all(idx_diffs == 1):
        raise ValueError('Expected index values to be strictly increasing by 1.')

    missing_rows = exists[~exists]
    if missing_rows.empty:
        return DataFrame(columns=['gap_id', 'start_idx', 'end_idx', 'n_segs'])

    # New group starts when two missing rows are not adjacent in the index.
    group_id = missing_rows.index.to_series().diff().ne(1).cumsum()
    temp = DataFrame({'idx': missing_rows.index.to_numpy(), 'gap_id': group_id.to_numpy()})
    gaps = temp.groupby('gap_id', as_index=False).agg(
        start_idx=('idx', 'min'),
        end_idx=('idx', 'max'),
        n_segs=('idx', 'size'),
    )
    gaps[['start_idx', 'end_idx', 'n_segs']] = gaps[['start_idx', 'end_idx', 'n_segs']].astype(int)
    return gaps


def split_by_long_gaps(df: DataFrame, gaps: DataFrame, long_gap_min_segs: int) -> list[DataFrame]:
    """Split df around long gaps (n_segs >= long_gap_min_segs), excluding those long-gap rows from chunks."""
    if gaps.empty:
        return [df.copy()]

    long_gaps = gaps[gaps['n_segs'] >= long_gap_min_segs].sort_values('start_idx')
    if long_gaps.empty:
        return [df.copy()]

    chunks = []
    cursor = int(df.index.min())
    last_idx = int(df.index.max())

    for gap in long_gaps.itertuples(index=False):
        if cursor <= gap.start_idx - 1:
            chunks.append(df.loc[cursor:gap.start_idx - 1].copy())
        cursor = int(gap.end_idx) + 1

    if cursor <= last_idx:
        chunks.append(df.loc[cursor:last_idx].copy())

    return [chunk for chunk in chunks if not chunk.empty]


def fill_short_gaps_per_column(
        df: DataFrame,
        feature_cols: list[str],
        warn_gap_n_segs_threshold: Optional[int] = None,
        min_donor_n_segs: int = 100,
        max_distortion_pct: float = 0.05,
        random_state: Optional[int] = None,
        patient: str = 'unknown patient',
) -> DataFrame:
    """
    Fill short gaps by sampling donors locally per column.

    For a gap of n_segs=L, donor window is up to W rows before and W rows after,
    where W = max(L, min_donor_n_segs).
    Sampling is with replacement and independently per feature column.
    Filled values are perturbed by a multiplicative factor in
    [1-max_distortion_pct, 1+max_distortion_pct].
    If no valid donors for a column, the gap cells remain NaN.
    """
    if warn_gap_n_segs_threshold is None:
        warn_gap_n_segs_threshold = float('inf')
    if min_donor_n_segs < 0:
        raise ValueError('min_donor_n_segs must be >= 0.')
    if max_distortion_pct < 0:
        raise ValueError('max_distortion_pct must be >= 0.')

    out = df.copy()
    rng = np.random.default_rng(random_state)

    gaps = detect_contiguous_gaps(out['exists'])
    if gaps.empty:
        return out

    for gap in gaps.itertuples(index=False):
        g_start, g_end, g_nsegs = gap.start_idx, gap.end_idx, gap.n_segs
        window_n_segs = max(g_nsegs, min_donor_n_segs)

        if g_nsegs > warn_gap_n_segs_threshold:
            duration: Timedelta = g_nsegs * SEGMENT.exact_dur
            print(
                f'[{patient}] Gap {g_start}-{g_end} has {g_nsegs} segments with duration {duration}; filling anyway.',
                flush=True)

        left_start = max(int(out.index.min()), g_start - window_n_segs)
        left_end = g_start - 1
        right_start = g_end + 1
        right_end = min(int(out.index.max()), g_end + window_n_segs)

        donor_idx = []
        if left_start <= left_end:
            donor_idx.extend(range(left_start, left_end + 1))
        if right_start <= right_end:
            donor_idx.extend(range(right_start, right_end + 1))
        if not donor_idx:
            raise ValueError(f'No valid donors found for gap {g_start}-{g_end}.')

        donors = out.loc[donor_idx]
        donors = donors.loc[donors['exists']]
        if donors.empty:
            raise ValueError(f'No valid donors found for gap {g_start}-{g_end}.')

        gap_index = range(g_start, g_end + 1)
        for col in feature_cols:
            sampled_vals = rng.choice(donors[col].to_numpy(), size=g_nsegs, replace=True)
            if max_distortion_pct > 0:
                distortion = rng.uniform(1 - max_distortion_pct, 1 + max_distortion_pct, size=g_nsegs)
                sampled_vals = sampled_vals * distortion
            out.loc[gap_index, col] = sampled_vals

    return out


def fill_ptnt_gaps(
        segs: DataFrame,
        long_gap_min_segs: int,
        warn_gap_n_segs_threshold: Optional[int] = None,
        min_donor_n_segs: int = 100,
        max_distortion_pct: float = 0.05,
        random_state: Optional[int] = None,
        feature_cols: list[str] = FeatureNames.ALL_ORDERED,
        patient: str = 'unknown patient',
) -> DataFrame:
    """Fill short gaps per chunk and reintegrate into one dataframe.

    Long gaps remain present in the output and keep NaN feature values.
    """
    selected_cols = ['start_mtz', 'exists', *feature_cols]
    base: DataFrame = segs.loc[:, selected_cols].copy()
    gaps: DataFrame = detect_contiguous_gaps(base['exists'])

    out = base.copy()

    chunks = split_by_long_gaps(base, gaps, long_gap_min_segs=long_gap_min_segs)

    for chunk in chunks:
        filled_chunk = fill_short_gaps_per_column(chunk,
                                                  feature_cols,
                                                  warn_gap_n_segs_threshold,
                                                  min_donor_n_segs,
                                                  max_distortion_pct,
                                                  random_state,
                                                  patient)
        out.loc[filled_chunk.index, feature_cols] = filled_chunk[feature_cols]

    return out


def fill_ptnt_gaps_and_save(
        pdir: PatientDir,
        long_gap_min_segs: int,
        warn_gap_n_segs_threshold: Optional[int] = None,
        min_donor_n_segs: int = 100,
        max_distortion_pct: float = 0.05,
        random_state: Optional[int] = None,
        feature_cols: list[str] = FeatureNames.ALL_ORDERED,
):
    print(f'[{pdir.name}] Filling Feature Gaps...', flush=True)
    segs = pd.read_pickle(pdir.segments_table.pickle)
    filled_features = fill_ptnt_gaps(segs,
                                     long_gap_min_segs,
                                     warn_gap_n_segs_threshold,
                                     min_donor_n_segs,
                                     max_distortion_pct,
                                     random_state,
                                     feature_cols,
                                     pdir.name)

    # Save results
    save_dataframe_multiformat(filled_features, pdir.filled_features_for_segs, csv_kwargs={'float_format': '%.3f'})
    print(f'[{pdir.name}] Saved filled features ({len(filled_features)} rows)', flush=True)


def fill_gaps_for_ptnts(
        pdirs: list[PatientDir],
        long_gap_min_duration: Timedelta = LONG_GAP_MIN_DURATION_FOR_FEATURE_FILLING,
        warn_gap_threshold: Optional[Timedelta] = LONG_GAP_MIN_DURATION_FOR_FEATURE_FILLING / 3,
        min_donor_n_segs: int = 100,
        max_distortion_pct: float = 0.05,
        random_state: Optional[int] = None,
        feature_cols: list[str] = FeatureNames.ALL_ORDERED,
        serial_processing: bool = False,
):
    """Fill gaps in all segments of all patients."""
    long_gap_min_segs = long_gap_min_duration // SEGMENT.exact_dur
    warn_gap_n_segs_threshold = None if warn_gap_threshold is None else warn_gap_threshold // SEGMENT.exact_dur

    func = partial(fill_ptnt_gaps_and_save, long_gap_min_segs=long_gap_min_segs,
                   warn_gap_n_segs_threshold=warn_gap_n_segs_threshold,
                   min_donor_n_segs=min_donor_n_segs,
                   max_distortion_pct=max_distortion_pct,
                   random_state=random_state, feature_cols=feature_cols)

    if serial_processing:
        for pdir in pdirs:
            func(pdir)
    else:
        with multiprocessing.Pool() as p:
            p.map(func, pdirs)


if __name__ == '__main__':
    pdirs_ = PATHS.patient_dirs()
    fill_gaps_for_ptnts(pdirs_, serial_processing=False)
