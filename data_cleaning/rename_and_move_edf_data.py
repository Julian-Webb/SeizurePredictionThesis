import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyedflib

from config import Paths
from data_cleaning.file_correction import clean_mac_files

VISIT_FOLDER_PATTERN = re.compile(r"^[vV]\d")


def name_file(patient_id: str, start_time: datetime):
    # return f"{patient_id}_{start_time.strftime('%Y-%m-%d_%H-%M')}_{end_time.strftime('%Y-%m-%d_%H-%M')}.edf"
    return f"{patient_id}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}.edf"


def _handle_problematic_edf_file(patient: str, visit: str, edf_path: Path, problematic_edfs_folder: Path = None):
    # move it to the problematic edfs folder, if specified, else delete it
    if problematic_edfs_folder:
        problematic_edfs_folder.mkdir(exist_ok=True)
        patient_dir = problematic_edfs_folder / patient
        patient_dir.mkdir(exist_ok=True)
        visit_dir = patient_dir / visit
        visit_dir.mkdir(exist_ok=True)
        edf_path.rename(visit_dir / edf_path.name)
    else:
        edf_path.unlink()


def move_and_list_edf_files(patient_dir: Path, is_competition_patient: bool, problematic_edfs_folder: Path = None):
    """Make a list of all edf files in a patient folder, rename the files, and move them to a single folder.
    :problematic_edfs_dir: if specified, move problematic edf files to this folder.
    :is_competition_patient: whether the patient is from dataset 20250501_SUBQ_SeizurePredictionCompetition_2025final
    :return: a dataframe with all edf files and their metadata, and a dataframe with all problematic edf files."""
    edfs = []
    problematic_edfs = []

    if is_competition_patient:
        # The competition data is already moved into the single folder for original edf data. It's added if it still exists.
        old_edf_dirs = []
        original_edf_dir = Paths.original_edf_dir(patient_dir)
        if original_edf_dir.exists():
            old_edf_dirs.append(original_edf_dir)
    else:
        # Get all visit folders - these contain the edf files
        old_edf_dirs = [f for f in patient_dir.iterdir() if f.is_dir() and VISIT_FOLDER_PATTERN.match(f.name)]

    # make a new folder for all edf files
    new_edf_dir = Paths.edf_dir(patient_dir)
    new_edf_dir.mkdir(exist_ok=True)
    patient = patient_dir.name

    # loop through all old edf folders/visit folders and handle edf files
    for old_edf_dir in old_edf_dirs:
        if is_competition_patient:
            visit = ''
        else:
            # remove the leading "v" or "V" from the visit folder name
            visit = old_edf_dir.name[1:]

        for edf_path in old_edf_dir.iterdir():
            logging.debug(f"Processing {edf_path}")
            # read edf info
            try:
                edf = pyedflib.EdfReader(str(edf_path))
                start_datetime = edf.getStartdatetime()
                duration_sec = timedelta(seconds=edf.getFileDuration())
                end_datetime = start_datetime + duration_sec
                edf.close()
            # Handle corrupted files
            except OSError as e:
                logging.error(f"Error reading {edf_path}:\n{e}")
                problematic_edfs.append(
                    {'problem': 'OSError reading file', 'patient': patient, 'visit': visit,
                     'old_file_name': edf_path.name})
                _handle_problematic_edf_file(patient, visit, edf_path, problematic_edfs_folder)
                continue

            new_edf_path = new_edf_dir / name_file(patient, start_datetime)

            # move the file to the new folder
            if not new_edf_path.exists():
                edf_path.rename(new_edf_path)
                edfs.append({
                    "old_file_name": edf_path.name,
                    "file_name": new_edf_path.name,
                    "start": start_datetime,
                    "end": end_datetime,
                    "duration_hours": str(duration_sec),
                    "visit": visit,
                })
            else:
                logging.error(f"File {new_edf_path} already exists - skipping.\nOriginal file: {edf_path.name}")
                problematic_edfs.append(
                    {'problem': 'File already exists', 'patient': patient, 'visit': visit,
                     'old_file_name': edf_path.name})
                _handle_problematic_edf_file(patient, visit, edf_path, problematic_edfs_folder)

        # delete visit folder or old edf directory
        try:
            old_edf_dir.rmdir()
        except OSError as e:
            logging.error(f"Error removing {old_edf_dir}: {e}")

    edfs_df = pd.DataFrame(edfs)
    if not edfs_df.empty:
        edfs_df.sort_values("start", inplace=True, ascending=True)
    edfs_df.reset_index(drop=True, inplace=True)

    problematic_edfs_df = pd.DataFrame(problematic_edfs)
    return edfs_df, problematic_edfs_df


def rename_and_move_edf_data(problematic_edfs_folder: Path = None) -> pd.DataFrame:
    """Rename and move all edf files in the patient folders.
    :return: a dataframe with all problematic edf files."""
    problematic_edfs_df = pd.DataFrame()

    # iterate through patient folders
    for patient_dir in Paths.patient_dirs():
        logging.info(f"Processing {patient_dir.parent}/{patient_dir.name}")
        is_competition_patient = patient_dir.is_relative_to(Paths.COMPETITION_DIR)
        edfs_df, problematic_edfs = move_and_list_edf_files(patient_dir, is_competition_patient,
                                                            problematic_edfs_folder)
        if not edfs_df.empty:
            edfs_df.to_csv(Paths.edf_files_sheet(patient_dir), index=False)
        problematic_edfs_df = pd.concat([problematic_edfs_df, problematic_edfs])

    return problematic_edfs_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clean_mac_files(Paths.BASE_DIR)
    # problem_edfs = rename_and_move_edf_data([FOR_MAYO_DIR, UNEEG_EXTENDED_DIR], PROBLEMATIC_EDFS_FOLDER)
    problem_edfs = rename_and_move_edf_data(Paths.PROBLEMATIC_EDFS_DIR)
    problem_edfs.to_csv(Paths.PROBLEMATIC_EDFS_FILE, index=False)
