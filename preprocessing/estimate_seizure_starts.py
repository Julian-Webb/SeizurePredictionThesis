# Estimate Seizure Starts
# The single marker falls somewhere within the seizure, not at the start.
# However, the start of the seizure is needed to determine the preictal window.
# The starts, as determined by the reviewer, usually aren't available.
# As a workaround, I will calculate the average time between the single marker and start, in the cases where both exist.
# Then, I will subtract that duration from the single marker to get an estimated start for seizure without an annotated
# start.
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import config.constants
from config.paths import PATHS, Dataset, PatientDir
from utils.io import save_annotations, pickle_path


def _load_all_seizures(ptnt_ann_files: dict[Path, Path]) -> pd.DataFrame:
    """Load all seizures from for_mayo and uneeg_extended."""
    all_seizures = pd.DataFrame()
    for ptnt_dir, ann_file in ptnt_ann_files.items():
        seizures = pd.read_pickle(ann_file)
        seizures['patient'] = ptnt_dir.name
        logging.debug(f'{ptnt_dir.name}: {seizures.shape[0]} seizures.')
        all_seizures = pd.concat([all_seizures, seizures])
    return all_seizures


def _single_marker_start_differences(ptnt_ann_files: dict[Path, Path]) -> Tuple[pd.Timedelta, pd.Timedelta, int]:
    all_seizures = _load_all_seizures(ptnt_ann_files)

    # find seizures where there's a single_marker and a start
    # "marker and start mask"
    ms_mask = all_seizures['single_marker'].notna() & all_seizures['start'].notna()
    ms_seizures = all_seizures[ms_mask]

    # calculate the difference between the single_marker and start
    diff = ms_seizures['single_marker'] - ms_seizures['start']

    # return the average difference and the standard deviation
    # noinspection PyTypeChecker
    return diff.mean(), diff.std(), len(ms_seizures)


def _estimate_seizure_starts_for_patient(single_marker_to_start_shift: pd.Timedelta, ptnt_dir: PatientDir,
                                         ptnt_ann_file: Path):
    seizures = pd.read_pickle(ptnt_ann_file)
    # find seizures where there's no start
    mask = seizures['start'].isna()

    seizures.loc[mask, 'start'] = seizures.loc[mask, 'single_marker'] - single_marker_to_start_shift
    # indicate where the start was estimated
    seizures['start_is_statistically_estimated'] = mask

    # save the updated dataframe
    seizures = seizures.sort_values(by='start').reset_index(drop=True)
    save_annotations(seizures, ptnt_dir.all_szr_starts_file)

    logging.debug(f'{ptnt_dir.name} starts saved.')


def estimate_seizure_starts():
    """Create a new file with estimated starts for seizures for each patient in for_mayo and uneeg_extended.
    The estimates are based on the mean difference between the start and single marker, where present from reviewers."""

    # Create file paths for existing annotations
    ptnt_ann_files = {}
    for ptnt_dir in PATHS.patient_dirs([Dataset.for_mayo], include_invalid_ptnts=True):
        ptnt_ann_files[ptnt_dir] = pickle_path(ptnt_dir.szr_anns_original_dir / ptnt_dir.name)
    for ptnt_dir in PATHS.patient_dirs([Dataset.uneeg_extended], include_invalid_ptnts=True):
        ptnt_ann_files[ptnt_dir] = pickle_path(ptnt_dir.combined_anns_file)

    mean, std, n_ms_seizures = _single_marker_start_differences(ptnt_ann_files)
    logging.info('Difference between single_marker and start:')
    logging.info(f'Mean: {mean.total_seconds()} s')
    logging.info(f'Std: {std.total_seconds()} s')
    logging.info(f'#seizures with both single_marker and start: {n_ms_seizures}')
    config.constants.single_marker_to_start_shift = mean

    ptnt_dirs = PATHS.patient_dirs([Dataset.for_mayo, Dataset.uneeg_extended], include_invalid_ptnts=True)
    for ptnt_dir in ptnt_dirs:
        _estimate_seizure_starts_for_patient(mean, ptnt_dir, ptnt_ann_files[ptnt_dir])
    logging.info(f'Seizure starts saved for patients: {[p.name for p in ptnt_dirs]}')


if __name__ == '__main__':
    logging.basicConfig(level='INFO', format='[%(levelname)s] %(message)s')
    estimate_seizure_starts()
