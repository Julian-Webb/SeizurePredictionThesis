from pathlib import Path

import pandas as pd

from config import Durations, Constants, PATHS
from utils.edf_utils import time_to_index
from utils.paths import PatientDir


def preictal_windows_table(patient_dir: PatientDir, seizure_annotations_file: Path) -> pd.DataFrame:
    """Create a table of the clips and segments for each preictal window.
    It contains the start and end times, whether it exists in the data, and the file name of each segment."""
    seizure_annotations = pd.read_csv(seizure_annotations_file, parse_dates=['start', 'end', 'single_marker'])

    # Create the table
    szr_times = seizure_annotations['single_marker']
    clips_range = list(range(Durations.CLIPS_PER_PREICTAL_INTERVAL))
    segments_range = list(range(Durations.SEGMENTS_PER_CLIP))
    segments = pd.DataFrame(
        columns=['start', 'end', 'exists', 'file', 'start_index'],
        index=pd.MultiIndex.from_product([szr_times, clips_range, segments_range],
                                         names=['seizure_time', 'clip', 'segment']),
    )

    for _, szr in seizure_annotations.iterrows():
        # Determine various time points
        # TODO there's a problem here: the single marker is used (it's at a random point during the seizure),
        #  but it's being treated as if it was the start when calculating the preictal window.
        szr_time = szr['single_marker']
        preictal_end = szr_time - Durations.PREICTAL_OFFSET
        preictal_start = preictal_end - Durations.PREICTAL_INTERVAL

        # Find files that have any overlap with the preictal interval
        edf_files = pd.read_csv(patient_dir.edf_files_sheet,
                                usecols=lambda x: x != 'old_file_name',  # ignore this column
                                parse_dates=['start', 'end'])
        # the formula a_start <= b_end and b_start <= a_end determines overlap between any intervals a and b
        overlapping_interval_mask = (preictal_start <= edf_files['end']) & (edf_files['start'] <= preictal_end)
        matching_edfs = edf_files[overlapping_interval_mask]

        # todo delete
        # This finds cases where there are multiple overlapping intervals with a certain time difference (for debugging)
        # if len(matching_edfs) >= 2:
        #     end_0 = matching_edfs.iloc[0]['end']
        #     start_1 = matching_edfs.iloc[1]['start']
        #     time_diff = start_1 - end_0
        #     if time_diff > timedelta(minutes=5):
        #         ...

        # Select the subset of segments that corresponds to this seizure (while maintaining the entire index)
        szr_segs = segments.loc[[szr_time]]

        szr_segs['start'] = preictal_start + szr_segs.index.get_level_values(
            1) * Durations.CLIP + szr_segs.index.get_level_values(2) * Durations.SEGMENT
        szr_segs['end'] = szr_segs['start'] + Durations.SEGMENT

        szr_segs['exists'] = False
        # Iterate over the matching edfs
        for _, edf in matching_edfs.iterrows():
            # Set the property "exists" to True for all segments contained in the edf
            segments_in_edf = (edf['start'] <= szr_segs['start']) & (szr_segs['end'] <= edf['end'])
            szr_segs.loc[segments_in_edf, 'exists'] = True
            szr_segs.loc[segments_in_edf, 'file'] = edf['file_name']

            # Calculate the start index based on the start
            szr_segs.loc[segments_in_edf, 'start_index'] = szr_segs.loc[segments_in_edf, 'start'].apply(
                lambda timestamp: round(time_to_index(file_start=edf['start'], timestamp=timestamp,
                                                      sampling_freq_hz=Constants.SAMPLING_FREQUENCY_HZ)))

            # Since we converted from time to index, which is slightly messy, I want to assert that the distance between
            #  the start indexes is correct.
            index_diffs = szr_segs.loc[segments_in_edf, 'start_index'].diff()
            # Ignore the first start entry because there is no previous start and make sure the differences are correct
            assert (index_diffs.iloc[1:] == Durations.SEGMENT_N_SAMPLES).all(), \
                f'The differences between two starts indexes is not {Durations.SEGMENT_N_SAMPLES=}'

        segments.update(szr_segs)

    return segments


if __name__ == '__main__':
    patient_dir = PatientDir(PATHS.for_mayo_dir / 'B52K3P3G')
    seizure_annotation_file = patient_dir.seizure_annotations_dir / 'B52K3P3G_CONSENSUS_corrected.csv'
    preictal_windows_table(patient_dir, seizure_annotation_file)
