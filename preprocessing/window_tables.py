import logging
import math
import multiprocessing
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pandas import Interval, DataFrame, Series

from config.constants import SAMPLING_FREQUENCY_HZ
from config.paths import PATHS
from config.intervals import CLIPS_PER_PREICTAL_INTERVAL, SEGMENTS_PER_CLIP, HORIZON, PREICTAL
from utils.edf_utils import time_to_index
from config.paths import PatientDir


# todo delete
# class WindowTable:
#     """Represent a window table with clips and segments"""
#     index_cols_interictal = ['clip', 'segment']
#     index_cols_preictal = ['seizure_start'] + index_cols_interictal
#
#     @staticmethod
#     def initialize(szr_starts: pd.Series | None):
#         """Initialize a window table from the seizure starts.
#         :param szr_starts: Series or list of start times. Don't specify for interictal tables"""
#         clips = list(range(Durations.CLIPS_PER_PREICTAL_INTERVAL))
#         segments = list(range(Durations.SEGMENTS_PER_CLIP))
#         components = ([szr_starts] if szr_starts is not None else []) + [clips, segments]
#         names = WindowTable.index_cols_interictal if szr_starts is None else WindowTable.index_cols_interictal
#         index = pd.MultiIndex.from_product(components, names=names)
#         return DataFrame(columns=['start', 'end', 'exists', 'file', 'start_index'], index=index)
#
#     @staticmethod
#     def from_csv(csv_path: Path, preictal: bool = False):
#         """Load a window table from a csv file
#         :param preictal: Whether it's preictal or interictal"""
#         dates = ['start', 'end']
#         if preictal:
#             dates = ['seizure_start'] + dates
#         index_cols = WindowTable.index_cols_preictal if preictal else WindowTable.index_cols_interictal
#         return pd.read_csv(csv_path, index_col=index_cols, parse_dates=dates)


class _WindowTable:
    """Represent a window table with clips and segments"""
    index_cols = ['clip', 'segment']
    date_cols = ['start', 'end']

    @classmethod
    def initialize(cls, szr_starts: pd.Series | None):
        """Initialize a window table from the seizure starts.
        :param szr_starts: Series or list of start times. Don't specify for interictal tables"""
        clips = list(range(CLIPS_PER_PREICTAL_INTERVAL))
        segments = list(range(SEGMENTS_PER_CLIP))
        components = ([szr_starts] if szr_starts is not None else []) + [clips, segments]
        index = pd.MultiIndex.from_product(components, names=cls.index_cols)
        return DataFrame(columns=['start', 'end', 'exists', 'file', 'start_index'], index=index)

    @classmethod
    def from_csv(cls, csv_path: Path):
        """Load a window table from a csv file"""
        return pd.read_csv(csv_path, index_col=cls.index_cols, parse_dates=cls.date_cols,
                           dtype={'start_index': 'Int64'})


class PreictalWindowTable(_WindowTable):
    index_cols = ['seizure_start'] + _WindowTable.index_cols
    date_cols = ['seizure_start'] + _WindowTable.date_cols


# todo i think I might need to delete this
class InterictalWindowTable(_WindowTable):
    @staticmethod
    def initialize():
        return super().initialize(szr_starts=None)


def preictal_windows_table(patient_dir: PatientDir) -> DataFrame:
    """Create a table of the clips and segments for each preictal window for a patient.
    It contains the start and end times, whether it exists in the data, and the file name of each segment."""
    szr_starts = pd.read_csv(patient_dir.valid_szr_starts_file, usecols=['start'], parse_dates=['start']).squeeze()

    # Create the table
    windows = PreictalWindowTable.initialize(szr_starts)

    for szr_start in szr_starts:
        # Determine various time points
        preictal_end = szr_start - HORIZON
        preictal_start = preictal_end - PREICTAL

        # Find files that have any overlap with the preictal interval
        edf_files = pd.read_csv(patient_dir.edf_files_sheet,
                                usecols=lambda x: x != 'old_file_name',  # ignore this column
                                parse_dates=['start', 'end'])
        # the formula a_start <= b_end and b_start <= a_end determines overlap between any intervals a and b
        overlapping_interval_mask = (preictal_start <= edf_files['end']) & (edf_files['start'] <= preictal_end)
        matching_edfs = edf_files[overlapping_interval_mask]

        # This finds cases where there are multiple overlapping intervals with a certain time difference (for debugging)
        # if len(matching_edfs) >= 2:
        #     end_0 = matching_edfs.iloc[0]['end']
        #     start_1 = matching_edfs.iloc[1]['start']
        #     time_diff = start_1 - end_0
        #     if time_diff > timedelta(minutes=5):
        #         ...

        # Select the subset of segments that corresponds to this seizure (while maintaining the entire index)
        szr_segs = windows.loc[[szr_start]]

        # Generate a DataFrame with all segments on one index level
        # duplicate this column because interval starts are the same as file starts here (since files are intervals)
        # noinspection PyTypeChecker
        szr_segs = _determine_segments_in_intervals(matching_edfs, preictal_start, preictal_end, file_start_col='start')
        # Reindex, so that segs are split into clips
        clips, segments = divmod(szr_segs.index, SEGMENTS_PER_CLIP)
        szr_segs.index = pd.MultiIndex.from_arrays([[szr_start] * len(segments), clips, segments],
                                                   names=['seizure_start', 'clip', 'segment'])

        windows.update(szr_segs)

    return windows


def _subtract_intervals(base: Interval, exclusions: List[Interval]) -> List[Interval]:
    """Subtract multiple Interval objects from the base Interval.
    :param base: The interval to be subtracted from.
    :param exclusions: a sorted list of non-overlapping Interval's
    :return: ``remaining`` - A sorted list of the remaining Interval's"""

    remaining = []
    cur_left = base.left
    for exc in exclusions:
        if base.right <= exc.left:
            # The remaining excluded intervals are after base
            break
        if cur_left < exc.left:
            # The gap between cur_left and exc.left is still available -> we save it to the results
            remaining.append(Interval(cur_left, min(exc.left, base.right), closed='both'))
        # Move the cursor forward.
        # We use max in case the previous exclusion ended before the current one. In practice, this shouldn't occur
        # because intervals have the same length (for now)
        cur_left = max(cur_left, exc.right)
        if base.right <= cur_left:
            break
    # Add the rest of the base interval
    if cur_left < base.right:
        remaining.append(Interval(cur_left, base.right, closed='both'))

    return remaining


def _determine_exclusions(szr_starts: Series) -> List[Interval]:
    """Determine the exclusion intervals from seizure starts and merge them.
    :return ``merged_exclusions``: a sorted list of non-overlapping Interval's"""
    # todo fix this with updated Durations/invervals
    exclusions = [Interval(start - Durations.BLOCKED_AROUND_SZR, start + Durations.BLOCKED_AROUND_SZR, closed='both')
                  for start in szr_starts]
    # Sort and merge overlapping/adjacent exclusions
    merged = [exclusions[0]]
    for iv in exclusions[1:]:
        if merged[-1].right < iv.left:
            # They don't overlap
            merged.append(iv)
        else:
            # Merge overlapping intervals
            merged[-1] = Interval(merged[-1].left, max(merged[-1].right, iv.right), closed='both')
    return merged


def _calc_remaining_intervals(ptnt_dir: PatientDir) -> DataFrame:
    szr_starts = pd.read_csv(ptnt_dir.valid_szr_starts_file, usecols=['start'], parse_dates=['start']).squeeze()
    # Intervals:
    exclusions = _determine_exclusions(szr_starts)
    available = pd.read_csv(ptnt_dir.edf_files_sheet, usecols=['file_name', 'start', 'end'],
                            parse_dates=['start', 'end'])

    remaining = []
    for i, cur_avail in available.iterrows():
        cur_remain = _subtract_intervals(Interval(cur_avail['start'], cur_avail['end']), exclusions)
        # integrate what remains from the cur_avail interval
        for iv in cur_remain:
            remaining.append({'file_name': cur_avail['file_name'], 'file_start': cur_avail['start'],
                              'start': iv.left, 'end': iv.right,
                              # 'duration_hours': str(iv.right - iv.left)
                              })
    return DataFrame(remaining)


# noinspection PyUnresolvedReferences
def _determine_segments_in_intervals(intervals: DataFrame, first_seg_start: pd.Timestamp,
                                     last_seg_end: pd.Timestamp, file_start_col: str) -> DataFrame:
    """From a DataFrame of intervals, determine the segments that are present in each interval.
    :param intervals: DataFrame of intervals with columns start, end, file_name, file_start
    :param file_start_col: Column containing the start of the file"""
    # Calculate the number of potential segments based on the time span of the intervals
    timespan = last_seg_end - first_seg_start
    n_segs = math.ceil(timespan / Durations.SEGMENT)

    # todo use WindowTable here or delete it
    segs = DataFrame(columns=['start', 'end', 'exists', 'file', 'start_index'], index=np.arange(n_segs))

    # The start is shifted by the duration of a segment per segment
    segs['start'] = first_seg_start + segs.index * Durations.SEGMENT
    segs['end'] = segs['start'] + Durations.SEGMENT

    segs['exists'] = False  # segs that exist are later set to True
    # Iterate over the intervals
    for _, iv in intervals.iterrows():
        # We only want segments completely contained in the interval, because we only want full segments
        iv_segs_mask = (iv['start'] <= segs['start']) & (segs['end'] <= iv['end'])
        segs.loc[iv_segs_mask, 'exists'] = True
        segs.loc[iv_segs_mask, 'file'] = iv['file_name']

        # todo this might be sped up and cleaner if I calculate the starts based on adding the index (rather than from Timestamp)
        # Calculate the start index based on the start of the file
        segs.loc[iv_segs_mask, 'start_index'] = segs.loc[iv_segs_mask, 'start'].apply(
            lambda timestamp: round(time_to_index(file_start=iv[file_start_col], timestamp=timestamp,
                                                  sampling_freq_hz=SAMPLING_FREQUENCY_HZ))
        )

        # Since we converted from time to index, which is slightly messy, I want to assert that the distance between
        #  the start indexes is correct.
        index_diffs = segs.loc[iv_segs_mask, 'start_index'].diff()
        # Ignore the first start entry because there is no previous start and make sure the differences are correct
        assert (index_diffs.iloc[1:] == Durations.SEGMENT_N_SAMPLES).all(), \
            f'The differences between two starts indexes is not {Durations.SEGMENT_N_SAMPLES=}'

    return segs


def interictal_windows_table(ptnt_dir: PatientDir) -> DataFrame:
    """Create a table of the clips and segments for each interictal window for a patient.
    It contains the start and end times, whether it exists in the data, and the file name of each segment.
    """
    # The interictal windows consist of 1h of continuous recording and must be 4h from any seizure
    # Procedure:
    # 1. There are excluded intervals (around the seizures) and available intervals (based on the recordings).
    #    The remaining intervals are calculated
    # 2. The remaining intervals are entirely segmented
    remaining = _calc_remaining_intervals(ptnt_dir)

    first_seg_start = remaining.iloc[0]['start']
    last_seg_end = remaining.iloc[-1]['end']

    segs = _determine_segments_in_intervals(remaining, first_seg_start, last_seg_end, file_start_col='file_start')
    return segs


def _process_patient(ptnt_dir: PatientDir):
    st = time.time()
    try:
        preictal_windows_table(ptnt_dir).to_csv(ptnt_dir.preictal_windows_file)
        interictal_windows_table(ptnt_dir).to_csv(ptnt_dir.interictal_windows_file)
        logging.info(f'{ptnt_dir.name}: {time.time() - st} seconds')
    except Exception as e:
        logging.error(f'Error processing patient {ptnt_dir.name}: {e}')


def window_tables():
    """Make the preictal window tables for all patients."""
    ### parallel:
    ptnt_dirs = PATHS.patient_dirs()
    with multiprocessing.Pool() as pool:
        pool.map(_process_patient, ptnt_dirs)

    ### serial:
    # for patient_dir in PATHS.patient_dirs():
    #     _process_patient(patient_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')
    window_tables()
