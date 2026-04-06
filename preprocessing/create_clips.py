"""Build clip-level tables from segment tables.

Assumptions:
- `segs` index is monotonic and represents sequential segment numbers.
- Clips are aligned to intervention starts (post-preictal) to avoid mixed preictal/other clips.
"""
import logging
import multiprocessing
from typing import Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame

from config import save_dataframe_multiformat, PatientDir, PATHS
from config.constants import MIN_SEGMENTS_PER_CLIP_RATIO
from config.intervals import SEGMENTS_PER_CLIP, SEGMENT
from utils.utils import safe_float_to_int


def map_segs_to_clips(seg_index: pd.Index, all_clip_starts: np.ndarray):
    """Map segments to clips.
    Returns
    -------
    clip_id - For each segment, the index of the clip that contains it.

    """
    clip_id = np.searchsorted(all_clip_starts, seg_index, side='right') - 1
    return clip_id


def create_clips_for_ptnt(
        segs: DataFrame,
        segs_per_clip: int = SEGMENTS_PER_CLIP,
        min_segs_per_clip_ratio: float = MIN_SEGMENTS_PER_CLIP_RATIO,
):
    """
    Create clips from a segment table.

    Parameters
    ----------
        segs: DataFrame with at least `start_mtz`, `type` and `exists` columns and a sequential index.
        segs_per_clip: Target number of segments per clip.
        min_segs_per_clip_ratio: Fraction of segments that must exist for a clip to be valid.

    Returns
    -------
        DataFrame with clip boundaries and aggregated properties.
    """
    min_segs_per_clip = safe_float_to_int(segs_per_clip * min_segs_per_clip_ratio)  # inclusive
    segs = segs.copy()

    # ==================================================================================================================
    # 1. Find Seeds
    # Seeds are the start of intervention intervals (right after preictal intervals), as well as the first and last
    # segment. Continuous clips will be created between the seeds, so that clips are aligned with preictal intervals.
    # ==================================================================================================================
    seeds = []

    # Find starts of intervention intervals
    intervention_segs = segs[segs['type'] == 'intervention']
    intervention_indices = intervention_segs.index.to_numpy()

    if len(intervention_indices) > 0:
        # Find breaks to identify separate intervals
        breaks = np.where(np.diff(intervention_indices) > 1)[0]
        intervention_intervals = np.split(intervention_indices, breaks + 1)
        # Add the beginning of each interval as a seed
        for interval in intervention_intervals:
            seeds.append(interval[0])

    # Add first and final segment as seed, so that clips are generated for the whole range of segments
    seeds = [segs.index[0]] + seeds + [segs.index[-1] + 1]

    # Remove duplicates and sort
    seeds = sorted(set(seeds))

    # ==================================================================================================================
    # 2. Create Clips Between Seeds
    # Loop through seeds and generate clips for each seed's "range" (from the current seed to the next seed)
    # ==================================================================================================================
    all_clip_starts = []
    all_clip_ends = []

    for i in range(len(seeds) - 1):
        range_start = seeds[i]
        range_end = seeds[i + 1]  # exclusive
        range_n_segs = range_end - range_start

        # Use divmod to split into full clips and remainder
        n_full_clips, remainder = divmod(range_n_segs, segs_per_clip)

        # Clip starts come from the first smaller clip (if it exists) and then the full clips.
        # This keeps full clips aligned to the end of the range (preictal adjacent).
        clip_starts = np.arange(range_start + remainder, range_end, segs_per_clip)
        clip_ends = clip_starts + segs_per_clip - 1
        # Insert the remainder clip start at the beginning of the range if it exists:
        if remainder > 0:
            clip_starts = np.insert(clip_starts, 0, range_start)
            clip_ends = np.insert(clip_ends, 0, range_start + remainder - 1)

        all_clip_starts.append(clip_starts)
        all_clip_ends.append(clip_ends)

    # Concatenate all clip starts and ends into a single array
    all_clip_starts = np.concatenate(all_clip_starts)
    all_clip_ends = np.concatenate(all_clip_ends)

    # ==================================================================================================================
    # 3. Calculate Clip Properties
    # Transform into DataFrame and calculate further properties
    # ==================================================================================================================
    clips = DataFrame({'start_seg': all_clip_starts, 'end_seg': all_clip_ends})

    # Calculate start and end datetime
    clips['start_mtz'] = segs.loc[clips['start_seg'], 'start_mtz'].values
    end_segs_start = segs.loc[clips['end_seg'], 'start_mtz'].values
    clips['end_mtz'] = end_segs_start + SEGMENT.exact_dur

    # Check if the clip is full (has all segs_per_clip "theoretical" segments (whether actual recordings exist or not))
    clips['segs_in_clip'] = clips['end_seg'] - clips['start_seg'] + 1
    clips['full'] = clips['segs_in_clip'] == segs_per_clip

    # Assign each segment to the clip that contains it.
    # For each segment, find the rightmost clip start <= segment index
    clip_id = map_segs_to_clips(segs.index, clips['start_seg'].values)
    agg = segs.groupby(clip_id).agg(
        n_existing=('exists', 'sum'),
        types=('type', lambda x: sorted(set(x))),
    )

    # Merge clip properties back into clips DataFrame
    clips = clips.join(agg)

    # Determine preictal clips
    clips['preictal'] = clips['types'].apply(lambda types: 'preictal' in types)
    # Make sure there are no mixed preictal clips
    mixed_preictal = clips['preictal'] & clips['types'].apply(lambda types: len(types) > 1)
    if mixed_preictal.any():
        for _, row in clips.loc[mixed_preictal, ['start_seg', 'end_seg', 'types']].iterrows():
            logging.error(f"Clip with segments {row['start_seg']} - {row['end_seg']} has other types mixed "
                          f"with preictal: {row['types']}")

    clips['sufficient_data'] = clips['n_existing'] >= min_segs_per_clip
    clips['valid'] = clips['full'] & clips['sufficient_data']

    # Sort columns
    clips = clips[['start_seg', 'end_seg', 'start_mtz', 'end_mtz', 'preictal', 'types',
                   'valid', 'segs_in_clip', 'full', 'n_existing', 'sufficient_data']]

    return clips


def create_clips_for_pdir(pdir: PatientDir):
    """Load a patient's segments, compute clips, and save the clip table."""
    logging.info(f'[{pdir.name}] 🎬 Creating Clips ...')
    segs = pd.read_pickle(pdir.segments_table.pickle)[['start_mtz', 'type', 'exists']]
    clips = create_clips_for_ptnt(segs)
    save_dataframe_multiformat(clips, pdir.clips_table)
    logging.info(f'[{pdir.name}] ✅ Completed Clip Creation.')


def create_clips_for_pdirs(pdirs: Iterable[PatientDir], serial_processing: bool = False):
    if serial_processing:
        for pdir in pdirs:
            create_clips_for_pdir(pdir)
    else:
        with multiprocessing.Pool() as pool:
            pool.map(create_clips_for_pdir, pdirs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
    pdirs_ = PATHS.patient_dirs()
    create_clips_for_pdirs(pdirs_)
