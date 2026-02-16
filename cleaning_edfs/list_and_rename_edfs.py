import logging
import math
import re
from pathlib import Path

from pandas import Timestamp, DataFrame, Timedelta, Series
from pyedflib import EdfReader

from config.paths import PatientDir, PATHS, Dataset
from utils.io import pickle_path
from utils.timezone import PatientTimezone, timezone_from_edf_annotation
from utils.utils import timeit


def name_file(patient_id: str, sequence_number: int, n_edfs: int, start_mtz: Timestamp):
    datetime = start_mtz.strftime('%Y-%m-%d_%H-%M-%S')
    # Look at the number all this patient's edfs, to determine how the sequence number should be padded
    n_digits = len(str(n_edfs))
    return f"{patient_id}_{sequence_number:0{n_digits}d}_{datetime}.edf"


@timeit
def list_edfs(pdir: PatientDir) -> DataFrame:
    """
    Make a DataFrame of all edf files in a patient dir with timestamps localized to the patient's main timezone.
    :return: DataFrame with columns 'old_file_name', 'file_name', 'start_localized', 'start_mtz', 'end_mtz', 'duration'
    """

    def get_competition_sequence_number(path: Path) -> int:
        """
        Extracts the sequence number from filenames:
        - competition-1_0287.edf -> 287
        - competition-1_0012_2020-04-01_15-04-03.edf -> 12
        """
        seq_match = re.search(r'^competition-\d+_(\d+)', path.stem)
        if seq_match:
            return int(seq_match.group(1))
        raise ValueError(f"Couldn't extract sequence number from {path.name}")

    is_competition = pdir.dataset == Dataset.competition
    edfs = []

    # Read in EDF info
    if not pdir.edf_dir.exists() or list(pdir.edf_dir.iterdir()) == 0:
        logging.warning(f"No edfs found in {pdir}")
        return DataFrame(edfs)

    edf_paths = list(pdir.edf_dir.iterdir())
    for edf_path in edf_paths:
        logging.debug(f"Processing {edf_path}")

        with EdfReader(str(edf_path)) as edf:
            start_naive = edf.getStartdatetime()
            duration_secs = edf.getFileDuration()
            edf_info = {
                "old_file_name": edf_path.name,
                "path": edf_path,
                "duration": Timedelta(seconds=math.floor(duration_secs)),
            }

            if is_competition:
                # Keep naive start for localization later
                edf_info['start_naive'] = start_naive
            else:
                # ultra2: Localize immediately using the file's specific offset
                annotation = edf.read_annotation()
                tz_offset = timezone_from_edf_annotation(annotation)
                # Localize to the specific offset (e.g. UTC+02:00)
                edf_info['start_localized'] = Timestamp(start_naive).tz_localize(tz_offset)

        edfs.append(edf_info)

    edfs = DataFrame(edfs)

    tz_info = PatientTimezone.from_competition(is_competition)
    if is_competition:
        # ultra2: already localized
        # 1. Sort by filename sequence
        edfs['seq'] = edfs['path'].apply(get_competition_sequence_number)
        edfs = edfs.sort_values('seq').reset_index(drop=True)

        # Batch localize using 'infer' for the London DST folds
        # Subtract a tiny duration from ends to avoid the exact 02:00:00 transition point and avoid
        #  AmbiguousTimeError from pandas
        ends_distorted = edfs['start_naive'] + edfs['duration'] - Timedelta(microseconds=1)

        # Interleave: [start1, end1, start2, end2, ...]
        combined = Series(index=range(len(edfs) * 2), dtype='datetime64[ns]')
        combined.iloc[0::2] = edfs['start_naive']
        combined.iloc[1::2] = ends_distorted

        # noinspection PyUnresolvedReferences
        localized = combined.dt.tz_localize(tz_info.location, ambiguous='infer', nonexistent='shift_forward')
        start_localized = localized.iloc[0::2].reset_index(drop=True)
        edfs['start_localized'] = start_localized

    edfs.sort_values('start_localized', inplace=True)

    # Normalize all to the patient's main timezone and strip TZ info
    # We use row-by-row conversion, because for ultra2, the dtype of start_localized is object, because
    #  localized times of the format UTC+01, are used, rather than datetime[ns, 'Europe/Berlin']
    edfs['start_mtz'] = (edfs['start_localized'].apply(lambda t: t.tz_convert(tz_info.main_timezone).tz_localize(None)))

    if not edfs['start_mtz'].is_unique:
        duplicates = edfs[edfs.duplicated('start_mtz', keep=False)]
        logging.error(f"Non-unique start times in {pdir.name}:\n{duplicates[['old_file_name', 'start_mtz']]}")

    # Add additional values
    edfs['end_mtz'] = edfs['start_mtz'] + edfs['duration']
    edfs.reset_index(drop=True, inplace=True)
    edfs['file_name'] = edfs.apply(lambda row: name_file(pdir.name, row.name, len(edf_paths), row.start_mtz), axis=1)

    # Remove unnecessary columns
    edfs = edfs[['old_file_name', 'file_name', 'start_localized', 'start_mtz', 'end_mtz', 'duration']]
    return edfs


@timeit
def rename_edfs(pdir: PatientDir, edfs: DataFrame):
    if edfs.empty:
        logging.warning(f"No edfs to rename in {pdir.name}")
        return

    # Rename
    edf_paths = DataFrame()
    edf_paths['old'] = pdir.edf_dir / edfs['old_file_name']
    edf_paths['new'] = pdir.edf_dir / edfs['file_name']
    edf_paths.apply(lambda path: path.old.rename(path.new), axis=1)


def list_and_rename_ptnt_edfs(pdir: PatientDir):
    logging.info(f"--- {pdir.name} ---")

    list_already_exists = pickle_path(pdir.edf_files_sheet).exists()
    if list_already_exists:
        raise ValueError(f"EDF list already exists for {pdir.name}. Aborting to preserve old file names.")

    edfs = list_edfs(pdir)

    # Save EDFs
    if not edfs.empty:
        edfs.to_pickle(pickle_path(pdir.edf_files_sheet))
        # Make durations better readable for csv
        edfs_copy = edfs.copy()
        edfs_copy['duration'] = edfs_copy['duration'].apply(lambda x: str(x.to_pytimedelta()))
        edfs_copy.to_csv(pdir.edf_files_sheet.with_suffix('.csv'), index=False)

    rename_edfs(pdir, edfs)
    logging.info(f"--- Finished {pdir.name} ---")
    return edfs


def list_and_rename_edfs(pdirs: list[PatientDir]):
    for pdir in pdirs:
        list_and_rename_ptnt_edfs(pdir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    pdirs = PATHS.patient_dirs()
    list_and_rename_edfs(pdirs)
