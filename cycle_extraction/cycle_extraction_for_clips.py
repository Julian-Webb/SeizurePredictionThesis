import multiprocessing
from typing import Optional

import numpy as np
import pandas as pd
from pandas import Timedelta, DataFrame

from config import PatientDir, PATHS, save_dataframe_multiformat
from config.intervals import CLIP
from cycle_functions import nc_filter, compute_plv_for_split_signal
from feature_extraction.extract_features import FeatureNames


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


def average_seg_features_per_clip(seg_features: DataFrame, clips: DataFrame):
    clips = clips.copy()
    feature_names = seg_features.columns

    # Groups segs by clip ID and compute averages per clip
    clip_id = np.searchsorted(clips['start_seg'], seg_features.index, side='right') - 1
    averages = seg_features.groupby(clip_id).median()
    clips[feature_names] = averages

    # Exclude clips where any segment is missing (the ones around large gaps)
    # per clip, per feature: does this clip contain any NA in that feature?
    has_na_df = seg_features[feature_names].isna().groupby(clip_id).any()
    # assert all feature columns have identical True/False pattern
    assert has_na_df.eq(has_na_df.iloc[:, 0], axis=0).to_numpy().all(), "NA pattern differs between feature columns"
    has_na = has_na_df.iloc[:, 0]
    clips.loc[has_na, feature_names] = pd.NA

    return clips


def split_dataframe_by_nan_gaps(df: DataFrame, relevant_cols: Optional[list[str]] = None):
    cols = relevant_cols or df.columns
    # gap row = all features NA
    gaps = df[cols].isna().all(axis=1)
    # Assign group IDs to groups of continuous gaps/existing data
    group_id = gaps.ne(gaps.shift(fill_value=False)).cumsum()
    # Include chunks that aren't NA
    chunks = [chunk for _, chunk in df.groupby(group_id) if not chunk[cols].isna().to_numpy().all()]
    return chunks


def cycle_extraction_for_ptnt(
        pdir: PatientDir,
        feature_names: list[str] = FeatureNames.CYCLES,
):
    print(f'[{pdir.name}] Cycle Extraction...')

    # Load data and discard irrelevant columns
    szrs = pd.read_pickle(pdir.all_szr_starts_file.pickle)['start_mtz']
    seg_features = pd.read_pickle(pdir.filled_features_for_segs.pickle)
    clips = pd.read_pickle(pdir.clips_table.pickle)[['start_seg', 'end_seg', 'start_mtz', 'end_mtz']].copy()

    # Average seg features per clip
    clips = average_seg_features_per_clip(seg_features[feature_names], clips)
    # Split clips by long gaps (where the features are still NA)
    chunked_clips = split_dataframe_by_nan_gaps(clips, feature_names)

    # Filter the features
    # todo problem: the clips don't all have the same length, so there's no correct sampling length...
    clips_per_hour = Timedelta(hours=1) / CLIP.exact_dur

    # Create filtered DataFrame per chunk. Reset the index per chunk.
    filtered_chunks = []
    for chunk in chunked_clips:
        filt_chunk = chunk[['start_mtz', 'end_mtz']].reset_index(drop=True)
        for feat_name in feature_names:
            filt_chunk[feat_name] = nc_filter(chunk[feat_name].values, fs=clips_per_hour, type_='multidien',
                                              figure=False)
        filtered_chunks.append(filt_chunk)

    # Assign seizures to chunks and get their relative indices in the chunks
    szr_indices_per_chunk = []
    total_szrs_in_chunks = 0
    for chunk in filtered_chunks:
        # Get the starts and ends of the clips
        starts, ends = chunk['start_mtz'].values, chunk['end_mtz'].values
        chunk_start = starts[0]
        chunk_end = ends[-1]

        szrs_in_chunk = szrs[(chunk_start <= szrs) & (szrs <= chunk_end)].values
        szr_indices = map_szrs_to_interval_index(szrs_in_chunk, starts, ends)
        szr_indices_per_chunk.append(szr_indices)
        total_szrs_in_chunks += len(szrs_in_chunk)

    assert total_szrs_in_chunks == len(szrs), \
        (f"Total number of seizures ({len(szrs)}) does not match the number of seizures in chunks "
         f"({total_szrs_in_chunks}). This might mean there are seizures during a large recording gap.")

    # Compute Phase Locking Values (PLV)
    res = DataFrame(index=feature_names, columns=['PLV', 'Mean angle', 'Mean angle (deg)'], dtype='float64')
    for feat_name in feature_names:
        feat_per_chunk = [filt_chunk[feat_name].values for filt_chunk in filtered_chunks]
        plv, mean_angle, mean_angle_deg, event_phases = \
            compute_plv_for_split_signal(feat_per_chunk, szr_indices_per_chunk)
        res.loc[feat_name] = [plv, mean_angle, mean_angle_deg]

    return res


def cycle_extraction_for_ptnts(pdirs: list[PatientDir] = PATHS.patient_dirs(), serial_processing: bool = False):
    # Compute Metrics
    if serial_processing:
        results = [cycle_extraction_for_ptnt(pdir) for pdir in pdirs]
    else:
        with multiprocessing.Pool() as pool:
            results = pool.map(cycle_extraction_for_ptnt, pdirs)

    # noinspection PyUnboundLocalVariable
    per_patient = {pdir.name: res for pdir, res in zip(pdirs, results)}
    metrics = pd.concat(per_patient, names=['patient', 'feature'])

    save_dataframe_multiformat(metrics, PATHS.cycle_extraction_results_table, save_index=True,
                               csv_kwargs={'float_format': '%.3f'})


if __name__ == '__main__':
    pdirs_ = PATHS.patient_dirs(include_fake_ptnts=False)
    cycle_extraction_for_ptnts(pdirs_)
    print(f"Created cycle extraction results for {len(pdirs_)} patients.")
