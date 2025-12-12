from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import multiprocessing
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
import pyedflib
from pandas import Timestamp, Timedelta, DataFrame
from pyedflib import EdfReader

from config.paths import PATHS
from config.paths import PatientDir
from data_cleaning.file_correction import clean_mac_files
from utils.io import pickle_path
from utils.timezone import PatientTimezone
from utils.utils import timeit

VISIT_FOLDER_PATTERN = re.compile(r"^[vV]\d")


def name_file(patient_id: str, start: Timestamp):
    datetime = start.strftime('%Y-%m-%d_%H-%M-%S')
    return f"{patient_id}_{datetime}.edf"


def _handle_problematic_edf_file(patient: str, visit: str, edf_path: Path, problematic_edfs_folder: Path = None):
    # move it to the problematic edfs folder, if specified, else delete it
    if problematic_edfs_folder:
        problematic_edfs_folder.mkdir(exist_ok=True, parents=True)
        patient_dir = problematic_edfs_folder / patient
        patient_dir.mkdir(exist_ok=True)
        visit_dir = patient_dir / visit
        visit_dir.mkdir(exist_ok=True)
        edf_path.rename(visit_dir / edf_path.name)
    else:
        edf_path.unlink()


def _get_original_edf_dirs(patient_dir: PatientDir, is_competition_ptnt: bool):
    if is_competition_ptnt:
        # The competition data is already moved into the single folder for original edf data. It's added if it still exists.
        ori_edf_dirs = []
        if patient_dir.original_edf_dir.exists():
            ori_edf_dirs.append(patient_dir.original_edf_dir)
    else:
        # Get all visit folders - these contain the edf files
        ori_edf_dirs = [f for f in patient_dir.iterdir() if f.is_dir() and VISIT_FOLDER_PATTERN.match(f.name)]
    return ori_edf_dirs


def timezone_from_edf_header(header: dict) -> str:
    # This returns the format "UTC+02h" for 2 hour offset
    tz = header['annotations'][0][2].removeprefix('LOCAL TIME = ')
    # Modifications to make it work with pandas Timestamp init
    tz = tz.removesuffix('h') + ':00'
    return tz


def read_edf_time_info(edf_path: Path, is_competition_ptnt: bool) -> Tuple[Timestamp, Timestamp, Timedelta]:
    """
    :raises: AmbiguousTimeError, if the edf start can't be unambiguously localized
    :raises OSError, if the edf can't be read
    :param edf_path: The str path to the edf file
    :param is_competition_ptnt: whether this patient is in the competition dataset
    :return: start, end, duration
    """
    tz_info = PatientTimezone.from_competition(is_competition_ptnt)
    if is_competition_ptnt:
        # Patients from the competition were based in this timezone. Their edf annotations don't include a timezone.
        tz_location = tz_info.location
    else:
        header = pyedflib.highlevel.read_edf_header(str(edf_path))
        tz_location = timezone_from_edf_header(header)

    edf = EdfReader(str(edf_path))
    start = edf.getStartdatetime()
    duration = Timedelta(seconds=edf.getFileDuration())
    edf.close()

    # Make the start timezone-aware
    start = Timestamp(start, tz=tz_location)
    # convert to patient's main timezone and remove explicit timezone info
    start = start.tz_convert(tz_info.main_timezone).tz_localize(None)

    end = start + duration

    # noinspection PyTypeChecker
    return start, end, duration


def _filter_duplicate_edfs_and_sort(edfs: list, problematic_edfs: list, patient: str) -> Tuple[DataFrame, DataFrame]:
    """
    Moves EDFs with a duplicated start to problematic_edfs and sort the lists. Returns them as DataFrames.
    """
    edfs = DataFrame(edfs)
    if not edfs.empty:
        edfs.sort_values("start", inplace=True, ascending=True)

        dup_mask = edfs.duplicated(subset=['start'], keep='first')
        if dup_mask.any():
            dup_rows = edfs.loc[dup_mask]
            for _, row in dup_rows.iterrows():
                problematic_edfs.append({'problem': 'duplicate start', 'patient': patient, 'visit': row['visit'],
                                         'old_file_name': row['old_file_name'], 'old_file_path': row['old_file_path']})
            # remove duplicate rows from edfs
            edfs = edfs[~dup_mask]

        edfs.reset_index(drop=True, inplace=True)
    problematic_edfs = DataFrame(problematic_edfs)
    return edfs, problematic_edfs


@timeit
def list_edf_files(ptnt_dir: PatientDir, is_competition_ptnt: bool) -> Tuple[DataFrame, DataFrame]:
    """
    Make a list of all edf files in a patient dir
    :return: edfs, problematic_edfs
    """
    edfs = []
    problematic_edfs = []

    original_edf_dirs = _get_original_edf_dirs(ptnt_dir, is_competition_ptnt)

    # Collect (edf_path, visit) tuples first
    for ori_edf_dir in original_edf_dirs:
        # remove the leading "v" or "V" from the visit folder name
        visit = '' if is_competition_ptnt else ori_edf_dir.name[1:]

        for edf_path in ori_edf_dir.iterdir():

            logging.debug(f"Processing {edf_path}")
            # read edf info
            try:
                start, end, duration = read_edf_time_info(edf_path, is_competition_ptnt)
            except OSError as e:
                logging.info(f"Error reading EDF : {e}")
                problematic_edfs.append(
                    {'problem': 'OSError reading file', 'patient': ptnt_dir.name, 'visit': visit,
                     'old_file_name': edf_path.name, 'old_file_path': edf_path})
                continue

            edfs.append({
                "old_file_name": edf_path.name, "start": start, "end": end, "duration": duration,
                "visit": visit,
                # This should get dropped before saving:
                "old_file_path": edf_path
            })

    # Sort and remove file with the same start
    edfs, problematic_edfs = _filter_duplicate_edfs_and_sort(edfs, problematic_edfs, ptnt_dir.name)
    return edfs, problematic_edfs


@timeit
# noinspection PyUnresolvedReferences
def move_edf_files(ptnt_dir: PatientDir, is_competition_ptnt: bool, edfs: DataFrame, problematic_edfs: DataFrame):
    """
    Move a patient's EDF files to the new dir
    """
    # Make the new EDF dir
    ptnt_dir.edf_dir.mkdir(parents=True, exist_ok=True)

    if not edfs.empty:
        edfs['file_name'] = edfs['start'].apply(lambda s: name_file(ptnt_dir.name, s))

    edfs['new_file_path'] = ptnt_dir.edf_dir / edfs['file_name']
    edfs.apply(lambda edf: edf.old_file_path.rename(edf.new_file_path), axis=1)
    edfs.drop(columns=['new_file_path'], inplace=True)

    # Handle Problematic EDFs
    # Get the part of the path containing dataset, patient, etc.
    if not problematic_edfs.empty:
        pe = problematic_edfs[['old_file_path']].copy()
        pe['relative_path'] = pe['old_file_path'].apply(lambda p: p.relative_to(PATHS.base_dir))
        pe['new_file_path'] = PATHS.problematic_edfs_dir / pe['relative_path']
        # Make parent folders
        pe['new_file_path'].apply(lambda p: p.parent.mkdir(parents=True, exist_ok=True))
        pe.apply(lambda edf: edf.old_file_path.rename(edf.new_file_path), axis=1)

    for orig_edf_dir in _get_original_edf_dirs(ptnt_dir, is_competition_ptnt):
        try:
            orig_edf_dir.rmdir()
        except OSError as e:
            logging.error(f"Error removing {orig_edf_dir}: {e}")


@timeit
def list_rename_move_edf_data_for_ptnt(ptnt_dir: PatientDir):
    """
    Make the EDF list and move files for a patient.
    :return: A list of the problematic edf files
    """
    logging.info(f"--- {ptnt_dir.name} ---")
    is_competition_ptnt = ptnt_dir.is_relative_to(PATHS.competition_dir)
    edfs, problematic_edfs = list_edf_files(ptnt_dir, is_competition_ptnt)
    move_edf_files(ptnt_dir, is_competition_ptnt, edfs, problematic_edfs)
    # If this was executed before, edfs will be empty and don't want to overwrite its file.
    if not edfs.empty:
        # change column order and drop old_file_path
        edfs = edfs[['old_file_name', 'file_name', 'start', 'end', 'duration', 'visit']]
        # noinspection PyCallingNonCallable
        edfs.to_pickle(pickle_path(ptnt_dir.edf_files_sheet))
        # Make durations better readable for csv
        edfs['duration'] = edfs['duration'].apply(lambda x: str(x.to_pytimedelta()))
        edfs.to_csv(ptnt_dir.edf_files_sheet.with_suffix('.csv'), index=False)
    return problematic_edfs


def list_rename_move_edf_data(ptnt_dirs: list[PatientDir]):
    """
    Make an EDF file list and move files for all specified patients. Timestamps get localized.
    :return: A list of all problematic edf files
    """
    # Sequential Processing:
    # problem_edfs_per_ptnt = []
    # for ptnt in ptnt_dirs:
    #     problematic_edfs = list_rename_move_edf_data_for_ptnt(ptnt)
    #     problem_edfs_per_ptnt.append(problematic_edfs)

    # Parallel Processing
    with multiprocessing.Pool() as pool:
        problem_edfs_per_ptnt = pool.map(list_rename_move_edf_data_for_ptnt, ptnt_dirs)

    # Return Problematic EDFs
    all_problematic_edfs = pd.concat(problem_edfs_per_ptnt, ignore_index=True)
    all_problematic_edfs.drop(columns=['old_file_path'], inplace=True, errors='ignore')
    return all_problematic_edfs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    clean_mac_files(PATHS.base_dir)
    # -------------------------------
    ptnt_dirs = PATHS.patient_dirs()
    all_problematic_edfs = list_rename_move_edf_data(ptnt_dirs)

    # Load already existing Problematic EDFs and join them
    # if PATHS.problematic_edfs_file.exists():
    #     prev_problem_edfs = pd.read_csv(PATHS.problematic_edfs_file)
    #     all_problematic_edfs = pd.concat([prev_problem_edfs, all_problematic_edfs], ignore_index=True)

    all_problematic_edfs.to_csv(PATHS.problematic_edfs_file, index=False)
