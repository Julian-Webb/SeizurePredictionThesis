import pickle

import numpy as np
import pandas as pd
from pandas import Timedelta, DataFrame

from config import PatientDir, PATHS
from config.intervals import SEGMENT
from cycle_functions import nc_filter, compute_plv_for_split_signal
from feature_extraction.extract_features import FeatureNames


def average_features_per_clip(feature_chunks: list[DataFrame],
                              clips: DataFrame,
                              feature_names: list[str] = FeatureNames.CYCLES, ):
    """
    Compute per-clip feature means from per-segment features.

    :param feature_chunks: A DataFrame per chunk containing the features per segment.
    :param clips: DataFrame with start_seg and end_seg columns (inclusive boundaries).
    :return: Clips DataFrame with feature columns set to mean values over covered segments.
    """
    out = clips.copy()
    if out.empty:
        return out
    if not feature_chunks:
        out[feature_names] = np.nan
        return out

    if 'start_seg' not in out.columns or 'end_seg' not in out.columns:
        raise KeyError("clips must contain 'start_seg' and 'end_seg' columns.")

    segs = pd.concat(feature_chunks, axis=0)
    missing_features = [name for name in feature_names if name not in segs.columns]
    if missing_features:
        raise KeyError(f'Missing feature columns in chunked segments: {missing_features}')

    seg_idx = segs.index.to_numpy()
    seg_order = np.argsort(seg_idx, kind='stable')
    seg_idx = seg_idx[seg_order]
    seg_values = segs.iloc[seg_order][feature_names]

    clip_starts = out['start_seg'].to_numpy()
    clip_ends = out['end_seg'].to_numpy()
    clip_order = np.argsort(clip_starts, kind='stable')
    starts_sorted = clip_starts[clip_order]
    ends_sorted = clip_ends[clip_order]

    if np.any(starts_sorted[1:] <= ends_sorted[:-1]):
        raise ValueError('Clips overlap. average_features_per_clip expects non-overlapping clip intervals.')

    # Assign each available segment to its containing clip in O(n log m).
    pos = np.searchsorted(starts_sorted, seg_idx, side='right') - 1
    valid = (pos >= 0) & (seg_idx <= ends_sorted[np.clip(pos, 0, len(ends_sorted) - 1)])

    clip_ids_sorted = clip_order
    assigned_clip_ids = clip_ids_sorted[pos[valid]]
    assigned_values = seg_values.iloc[valid].copy()
    assigned_values['_clip_id'] = assigned_clip_ids
    clip_means = assigned_values.groupby('_clip_id', sort=False)[feature_names].mean()

    out[feature_names] = np.nan
    out.loc[clip_means.index, feature_names] = clip_means.to_numpy()
    return out


def split_clips_by_segment_chunks(clips: DataFrame,
                                  feature_chunks: list[DataFrame]) -> list[DataFrame]:
    """
    Split clips by the same large-gap chunking used for segment features.

    A clip is assigned to a chunk only if both start_seg and end_seg lie inside
    the same chunk interval.
    """
    if clips.empty:
        return [clips.copy() for _ in feature_chunks]
    if not feature_chunks:
        return []
    if 'start_seg' not in clips.columns or 'end_seg' not in clips.columns:
        raise KeyError("clips must contain 'start_seg' and 'end_seg' columns.")

    chunk_starts = np.array([int(chunk.index.min()) for chunk in feature_chunks])
    chunk_ends = np.array([int(chunk.index.max()) for chunk in feature_chunks])

    clip_starts = clips['start_seg'].to_numpy()
    clip_ends = clips['end_seg'].to_numpy()

    start_chunk_idx = np.searchsorted(chunk_starts, clip_starts, side='right') - 1
    end_chunk_idx = np.searchsorted(chunk_starts, clip_ends, side='right') - 1

    start_valid = (start_chunk_idx >= 0) & (clip_starts <= chunk_ends[np.clip(start_chunk_idx, 0, len(chunk_ends) - 1)])
    end_valid = (end_chunk_idx >= 0) & (clip_ends <= chunk_ends[np.clip(end_chunk_idx, 0, len(chunk_ends) - 1)])
    valid = start_valid & end_valid & (start_chunk_idx == end_chunk_idx)

    if not np.all(valid):
        n_invalid = int((~valid).sum())
        raise ValueError(
            f'{n_invalid} clip(s) could not be assigned to a single chunk. '
            'This usually means a clip overlaps a removed large-gap interval.'
        )

    assigned_idx = pd.Series(start_chunk_idx, index=clips.index)
    clips_per_chunk = [clips.loc[assigned_idx == i].copy() for i in range(len(feature_chunks))]
    return clips_per_chunk


def map_szrs_to_interval_index(szrs: np.ndarray,
                               interval_starts: np.ndarray,
                               interval_ends: np.ndarray):
    """
    Map seizures to the interval index.
    :param szrs: Seizure timestamps (1D array).
    :param interval_starts: Start timestamps for non-overlapping intervals (1D array).
    :param interval_ends: End timestamps for non-overlapping intervals (1D array).
    :return: Interval index per seizure as ndarray (-1 if no interval contains it).
    """
    # Reshape for broadcasting: (n_seizures, 1) and (1, n_intervals)
    szr_values = szrs[:, np.newaxis]
    starts_broadcast = interval_starts[np.newaxis, :]
    ends_broadcast = interval_ends[np.newaxis, :]

    # Check if seizure is in interval: start <= seizure < end
    in_interval = (starts_broadcast <= szr_values) & (szr_values < ends_broadcast)

    # Find the interval index for each seizure (-1 if not found)
    interval_indices = np.where(in_interval.any(axis=1),
                                in_interval.argmax(axis=1),
                                -1)

    return interval_indices


def cycle_extraction_for_ptnt(
        pdir: PatientDir,
        feature_names: list[str] = FeatureNames.CYCLES,
):
    # Load data and discard irrelevant columns
    szrs = pd.read_pickle(pdir.all_szr_starts_file.pickle)['start_mtz']
    clips = pd.read_pickle(pdir.clips_table.pickle)  # [['start_seg', 'end_seg', 'start_mtz', 'end_mtz']]
    # todo use clips instead of segs?

    # Feature chunks are the existing chunks between large recording gaps
    with open(pdir.filled_features_for_segs.pickle, 'rb') as f:
        feature_chunks = pickle.load(f)

    clips = average_features_per_clip(feature_chunks, clips)
    clips_per_chunk = split_clips_by_segment_chunks(clips, feature_chunks)

    if sum(len(chunk) for chunk in clips_per_chunk) != len(clips):
        raise AssertionError('Not all clips were assigned to chunked clip outputs.')

    segs_per_hour = Timedelta(seconds=3600) / SEGMENT.exact_dur

    filtered_chunks = []
    for feat_chunk in feature_chunks:
        filtered = feat_chunk.copy()  # (n_segs, n_features)
        for feature_name in feature_names:
            filtered[feature_name] = nc_filter(feat_chunk[feature_name], fs=segs_per_hour, type_='multidien',
                                               figure=False)
        filtered_chunks.append(filtered)

    # Assign seizures to chunks
    szr_indices_per_chunk = []
    total_szrs_in_chunks = 0
    for feat_chunk in filtered_chunks:
        # Get starts and ends of the intervals (clips / segments)  todo adjust comment
        starts = feat_chunk['start_mtz'].values
        ends = starts + SEGMENT.exact_dur  # todo adjust
        chunk_start = starts[0]
        chunk_end = ends[-1]

        szrs_in_chunk = szrs[(chunk_start <= szrs) & (szrs <= chunk_end)].values
        szr_indices = map_szrs_to_interval_index(szrs_in_chunk, starts, ends)

        szr_indices_per_chunk.append(szr_indices)
        total_szrs_in_chunks += len(szrs_in_chunk)

    # todo delete
    # szrs['seg_idx'] = map_szrs_to_interval_index(szrs['start_mtz'].values, segs['start_mtz'].values,
    #                                              segs['end_mtz'].values)
    # szr_indices_per_chunk = []
    # total_szrs_in_chunks = 0
    # for feat_chunk in filtered_chunks:
    #     chunk_start: Timestamp = feat_chunk.iloc[0]['start_mtz']
    #     chunk_end: Timestamp = feat_chunk.iloc[-1]['start_mtz'] + SEGMENT.exact_dur  # todo change for clips
    #     szrs_in_chunk = szrs[(chunk_start <= szrs['start_mtz']) & (szrs['start_mtz'] <= chunk_end)]
    #     szr_indices_per_chunk.append(szrs_in_chunk['seg_idx'].values)
    #     total_szrs_in_chunks += len(szrs_in_chunk)

    # todo uncomment
    # assert total_szrs_in_chunks == len(szrs), \
    #     (f"Total number of seizures ({len(szrs)}) does not match the number of seizures in chunks "
    #      f"({total_szrs_in_chunks}). This might mean there are seizures during a large recording gap.")

    # Compute Phase Locking Values (PLV)
    print(f'--- {pdir.name} ---')
    for feat_name in feature_names:
        feature_per_chunk = [filtered_chunk[feat_name].values for filtered_chunk in filtered_chunks]
        plv, mean_angle, mean_angle_deg, event_phases = \
            compute_plv_for_split_signal(feature_per_chunk, szr_indices_per_chunk)

        print(feat_name)
        print(f'PLV: {plv:.3f}')
        print(f'Mean angle: {round(mean_angle_deg)}°')
        print()
    print()

    # Save Results
    # todo report results

    return


def cycle_extraction_for_ptnts(pdirs: list[PatientDir] = PATHS.patient_dirs()):
    for pdir in pdirs:
        cycle_extraction_for_ptnt(pdir)


if __name__ == '__main__':
    # pdirs_ = PATHS.patient_dirs() todo

    pdirs_ = [
        # PatientDir('/data/home/webb/d/UNEEG/datasets/competition/competition-01-MINIFAKE'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/competition/competition-1'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/competition/competition-2'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/competition/competition-3'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-01'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-01-FAKE'),
        PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-03'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-04'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-05'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-07'),
        PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-12'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-15'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-16'),
        # PatientDir('/data/home/webb/d/UNEEG/datasets/ultra2/U002-DE01-17')
    ]

    cycle_extraction_for_ptnts(pdirs_)
