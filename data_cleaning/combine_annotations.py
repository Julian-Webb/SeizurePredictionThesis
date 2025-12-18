import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from pandas import NaT

from config.paths import PATHS
from data_cleaning.file_correction import clean_mac_files
from config.paths import PatientDir, Dataset
from utils.io import pickle_path, save_dataframe_multiformat


def combine_annotation_files(paths: List[Path]):
    """Combine multiple annotation files into one. Annotations may be duplicate and single markers will be combined
    with user seizure markers with a start and end."""
    # Procedure
    # 1. combine all files into one dataframe
    # 2. Select rows with a start, marker, and end
    # 3. Add rows with just a start and end (if not duplicated)
    # 4. Add rows with just a marker (if not duplicated)
    #    * Add the marker between a start and end, if it's within the interval
    #    * Add the marker as a new row if it's not within any interval
    # 5. Sort by datetime and save

    # get seizure dataframes from all files
    szr_dfs = [pd.read_pickle(file_path) for file_path in paths]
    szr_dfs = pd.concat(szr_dfs, ignore_index=True)

    # NOTE: NaT = Not a Time
    # rows with start, marker, and end
    start_marker_end_rows = szr_dfs[(
            (szr_dfs['start'] != NaT) &
            (szr_dfs['single_marker'] != NaT) &
            (szr_dfs['end'] != NaT)
    )]

    # rows with just start and end
    start_end_rows = szr_dfs[(
            (szr_dfs['start'] != NaT) &
            (szr_dfs['single_marker'] == NaT) &
            (szr_dfs['end'] != NaT)
    )]

    # rows with just a marker
    marker_rows = szr_dfs[(
            (szr_dfs['start'] == NaT) &
            (szr_dfs['single_marker'] != NaT) &
            (szr_dfs['end'] == NaT)
    )]

    seizures = start_marker_end_rows.copy(deep=True)

    # --- add rows that have just a start and end
    rows_to_add = []
    for i, row in start_end_rows.iterrows():
        if not row['start'] in seizures['start'].values:
            # the seizure should be added
            assert row['end'] not in seizures['end'].values, f"The start is not contained but the end is: {row}"
            rows_to_add.append(i)
            logging.debug(f"Added row: {row}")
        else:
            # row is duplicate
            logging.info(f"Duplicate row: {row}")

    seizures = pd.concat([seizures, start_end_rows.loc[rows_to_add]])

    # --- add rows that have just a marker
    rows_to_add = []
    # convert seizures start/end to datetime for comparison
    seizures_start = pd.to_datetime(seizures['start'])
    seizures_end = pd.to_datetime(seizures['end'])
    for i, row in marker_rows.iterrows():
        # check if it's within the start-end interval of an existing row
        # convert marker to datetime
        marker = datetime.strptime(str(row['single_marker']), "%Y-%m-%d %H:%M:%S.%f")
        within_interval = ((seizures_start <= marker) & (marker <= seizures_end))

        # a list of indices of where the marker is within the interval
        match_indices = seizures.index[within_interval].tolist()

        assert len(match_indices) <= 1, f"Multiple matches for marker: {row}"
        if match_indices:
            # If it's within an interval, we check if there's already a single_marker.
            idx = match_indices[0]
            existing_marker = seizures.loc[idx, 'single_marker']
            if existing_marker == NaT:
                # If there's none, we add the marker
                seizures.loc[idx, 'single_marker'] = row['single_marker']
            elif existing_marker != NaT:
                # If there is one, we assert it's the same
                assert existing_marker == row[
                    'single_marker'], f"Different markers: {existing_marker} and {row['single_marker']}"
        else:
            # We add the marker
            rows_to_add.append(i)
            logging.debug(f"Added row: {row}")

    seizures = pd.concat([seizures, marker_rows.loc[rows_to_add]])

    # --- sort by datetime
    has_single_marker = seizures['single_marker'] != NaT
    # Select the single marker if there is one
    seizures.loc[has_single_marker, 'sort_time'] = seizures.loc[has_single_marker, 'single_marker']
    # Select the start otherwise
    seizures.loc[~has_single_marker, 'sort_time'] = seizures.loc[~has_single_marker, 'start']

    seizures['sort_time'] = pd.to_datetime(seizures['sort_time'])
    seizures.sort_values(by='sort_time', inplace=True, ascending=True)
    # reset index
    seizures.reset_index(drop=True, inplace=True)
    seizures.drop(columns=['sort_time'], inplace=True)
    return seizures


def combine_annotations(patient_dirs: Iterable[PatientDir]):
    for ptnt_dir in patient_dirs:
        ann_files = [file for file in ptnt_dir.szr_anns_original_dir.iterdir() if
                            file.suffix == '.pkl' and not 'all automatic detections' in file.name]
        combined = combine_annotation_files(ann_files)
        save_dataframe_multiformat(combined, ptnt_dir.combined_anns_file)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    clean_mac_files(PATHS.uneeg_extended_dir)
    combine_annotations(PATHS.patient_dirs([Dataset.uneeg_extended]))
