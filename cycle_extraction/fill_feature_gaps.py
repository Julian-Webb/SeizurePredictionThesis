"""
Fill gaps in the features during missing recording intervals of patients.
"""

import multiprocessing
import pickle
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Timedelta

from config import PATHS, PatientDir, MultiPath
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
        exists_col: str = 'exists',
        warn_gap_n_segs_threshold: Optional[int] = None,
        random_state: Optional[int] = None,
        patient: str = 'unknown patient',
) -> DataFrame:
    """
    Fill short gaps by sampling donors locally per column.

    For a gap of n_segs=L, donor window is up to L rows before and L rows after.
    Sampling is with replacement and independently per feature column.
    If no valid donors for a column, the gap cells remain NaN.
    """
    if warn_gap_n_segs_threshold is None:
        warn_gap_n_segs_threshold = float('inf')

    out = df.copy()
    rng = np.random.default_rng(random_state)

    gaps = detect_contiguous_gaps(out[exists_col])
    if gaps.empty:
        return out

    for gap in gaps.itertuples(index=False):
        g_start, g_end, g_nsegs = gap.start_idx, gap.end_idx, gap.n_segs

        if g_nsegs > warn_gap_n_segs_threshold:
            duration: Timedelta = g_nsegs * SEGMENT.exact_dur
            print(
                f'[{patient}] Gap {g_start}-{g_end} has {g_nsegs} segments with duration {duration}; filling anyway.',
                flush=True)

        left_start = max(int(out.index.min()), g_start - g_nsegs)
        left_end = g_start - 1
        right_start = g_end + 1
        right_end = min(int(out.index.max()), g_end + g_nsegs)

        donor_idx = []
        if left_start <= left_end:
            donor_idx.extend(range(left_start, left_end + 1))
        if right_start <= right_end:
            donor_idx.extend(range(right_start, right_end + 1))
        if not donor_idx:
            raise ValueError(f'No valid donors found for gap {g_start}-{g_end}.')

        donors = out.loc[donor_idx]
        donors = donors.loc[donors[exists_col]]
        if donors.empty:
            raise ValueError(f'No valid donors found for gap {g_start}-{g_end}.')

        gap_index = range(g_start, g_end + 1)
        for col in feature_cols:
            sampled_vals = rng.choice(donors[col].to_numpy(), size=g_nsegs, replace=True)
            out.loc[gap_index, col] = sampled_vals

    return out


def fill_ptnt_gaps(
        segs: DataFrame,
        long_gap_min_segs: int,
        warn_gap_n_segs_threshold: Optional[int] = None,
        random_state: Optional[int] = None,
        feature_cols: list[str] = FeatureNames.CYCLES,
        patient: str = 'unknown patient',
) -> list[DataFrame]:
    """Split by long gaps and fill short gaps per resulting chunk."""
    selected_cols = ['start_mtz', 'exists', *feature_cols]
    base: DataFrame = segs.loc[:, selected_cols].copy()
    gaps: DataFrame = detect_contiguous_gaps(base['exists'])
    chunks = split_by_long_gaps(base, gaps, long_gap_min_segs=long_gap_min_segs)

    filled_chunks = []
    for chunk in chunks:
        filled_chunks.append(
            fill_short_gaps_per_column(chunk, feature_cols, 'exists', warn_gap_n_segs_threshold, random_state,
                                       patient)
        )
    return filled_chunks


def fill_ptnt_gaps_and_save(
        pdir: PatientDir,
        long_gap_min_segs: int,
        warn_gap_n_segs_threshold: Optional[int] = None,
        random_state: Optional[int] = None,
        feature_cols: list[str] = FeatureNames.CYCLES,
):
    print(f'[{pdir.name}] Processing...', flush=True)
    segs = pd.read_pickle(pdir.segments_table.pickle)
    filled_chunks = fill_ptnt_gaps(segs, long_gap_min_segs, warn_gap_n_segs_threshold, random_state, feature_cols,
                                   pdir.name)

    # Save as pickle in one file
    path = pdir.filled_features_table
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.pickle, 'wb') as handle:
        pickle.dump(filled_chunks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save as csv for viewing in one file per chunk
    for i, chunk in enumerate(filled_chunks):
        p = MultiPath(path.with_name(f'{path.name}_chunk{i}')).csv
        chunk.to_csv(p, index=False, float_format='%.3f')

    print(f'[{pdir.name}] Saved {len(filled_chunks)} chunks', flush=True)


def fill_gaps_for_ptnts(
        pdirs: list[PatientDir],
        long_gap_min_duration: Timedelta = Timedelta(days=14),
        warn_gap_threshold: Optional[Timedelta] = Timedelta(days=3),
        random_state: Optional[int] = None,
        feature_cols: list[str] = FeatureNames.CYCLES,
        serial_processing: bool = False,
):
    """Fill gaps in all segments of all patients."""
    long_gap_min_segs = long_gap_min_duration // SEGMENT.exact_dur
    warn_gap_n_segs_threshold = None if warn_gap_threshold is None else warn_gap_threshold // SEGMENT.exact_dur

    func = partial(fill_ptnt_gaps_and_save, long_gap_min_segs=long_gap_min_segs,
                   warn_gap_n_segs_threshold=warn_gap_n_segs_threshold,
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
