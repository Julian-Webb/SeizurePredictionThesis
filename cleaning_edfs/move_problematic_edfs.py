import re
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas import DataFrame
from pyedflib import EdfReader

from config import PatientDir, PATHS, Dataset
from cleaning_edfs.list_and_rename_edfs import list_edfs
from utils.io import pickle_path

# EDFs with a start before this year are considered corrupted
BOUNDARY_YEAR = 2010


def _check_unknown_implant(string: str):
    # Pattern explanation:
    # - matches "unknown"
    # - [ _-]* matches any number of spaces, underscores, or dashes
    # - impl[an]{1,2}t matches "implant" even with common typos like "implat" or "implaant"
    unknown_implant_pattern = r"unknown[ _-]*impl[an]{1,2}t"
    return re.search(unknown_implant_pattern, string, re.IGNORECASE)


def check_edf(path: Path, correct_patientcode: str, duplicate_groups: list[list[str]]) -> Optional[str]:
    """Check EDF for the following possible issues:
    * Unknown Implant (based on file name)
    * OSError reading file
    * False starting year
    * False patient code (implant ID)
    :returns: None if everything is ok, else the problem
    """
    # Check for Unknown_Implant
    if _check_unknown_implant(path.name):
        return 'Unknown_Implant'
    try:
        with EdfReader(str(path)) as edf:
            header = edf.getHeader()
            if _check_unknown_implant(header['patientcode']):
                return 'Unknown_Implant'
            # Check the patientcode, but only if there is a patientcode (not the case for competition patients)
            if correct_patientcode and header['patientcode'] != correct_patientcode:
                is_duplicate = any(str(path) in group for group in duplicate_groups)
                if is_duplicate:
                    return 'Wrong_patientcode_and_duplicate'
                else:
                    raise NotImplementedError(f"File has wrong patientcode, but it's not a duplicate: {str(path)}")
            # Check the start year
            start = edf.getStartdatetime()
            if start.year < BOUNDARY_YEAR:
                return 'False_start_year'
    except OSError as e:
        print(f"Error reading EDF : {e}")
        return 'OSError_reading_file'

    # If everything is fine
    return None


def move_edf(edf_path: Path, problem: str):
    """Move the edf to the dir associated with the problem"""
    rel_path = edf_path.relative_to(PATHS.datasets_dir)
    new_path = PATHS.problematic_edfs_dir / problem / rel_path
    new_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Moving:\n   {str(edf_path)}\n ->{str(new_path)}')
    edf_path.rename(new_path)


def move_problematic_ptnt_edfs(pdir: PatientDir, correct_patientcode: str, duplicate_groups: list[list[str]]):
    print(f'Processing {pdir.name}')
    edf_files = list(pdir.edf_dir.iterdir())
    total_files = len(edf_files)

    for i, edf_path in enumerate(edf_files):
        print(f"   Progress: {i}/{total_files} files checked.", end="\r")
        # print(f'Processing {str(edf_path)}')
        problem = check_edf(edf_path, correct_patientcode, duplicate_groups)
        if problem:
            print()
            move_edf(edf_path, problem)
            print()
    print(f'    Finished {pdir.name}: {total_files} files checked.')


def move_duplicates_within_patients(duplicate_groups: list[list[str]], dataset_dir: Path):
    """
    Move EDFs that are duplicated in a patient's directory
    """
    # Convert duplicate groups to Path
    duplicate_groups = [[Path(path) for path in group] for group in duplicate_groups]

    # Iterate through groups and check if the files are from the same patient dir
    for group in duplicate_groups:
        patient_ids = set()
        for path in group:
            # This yields a path of this form:
            # /dataset/patientID/edf_data/filename.edf
            rel_path = path.relative_to(dataset_dir)
            patient_ids.add(rel_path.parts[1])
        same_ptnt = len(patient_ids) == 1

        # Decide which of the files to keep
        if same_ptnt:
            patient_id = list(patient_ids)[0]
            # Scoring system to find the best file to keep
            # Higher score = better candidate
            best_idx = 0
            best_score = float('-inf')

            for i, path in enumerate(group):
                score = 0
                # Check if the file starts with the patient's name
                if path.name.startswith(patient_id):
                    score += 2
                # Check if it contains a suffix like _1 or _2, which is typical of copying a file (ultra2 dataset).
                # For competition patients, most files are like this, so we choose the one with the lower index
                match = re.search(r"_\d+$", path.stem)
                if match:
                    suffix_value = int(match.group(0).removeprefix('_'))
                    score -= suffix_value
                else:
                    score += 1

                if score > best_score:
                    best_score = score
                    best_idx = i

            to_move = group[:best_idx] + group[best_idx + 1:]
            for path in to_move:
                move_edf(path, 'Duplicate_within_patient')


def find_duplicate_files(root_dir: Path) -> list[list[str]]:
    """
    :param root_dir: The dir that contains all files to check (they can be in subdirs)
    :return: A large str with all duplicate groups
    """
    res = subprocess.run(
        ["fdupes", "-rn", root_dir],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"fdupes failed: {res.stderr}")

    # duplicates groups are separated by a blank line
    groups = res.stdout.split("\n\n")
    groups = [group.strip().split("\n") for group in groups if group.strip()]
    return groups


def move_edfs_with_duplicate_start(pdir: PatientDir, edfs: DataFrame):
    """
    For EDFs within a patient that have a duplicate start, keep just one of them and move the rest.
    :param edfs: DataFrame containing the EDF files for the patient from function list_edfs
    """
    dups = edfs[edfs.duplicated('start', keep=False)]
    if dups.empty:
        print(f'No EDFs with duplicate start found for {pdir.name}')
        return

    to_move = []
    # Group by start time and handle each set of duplicates separately
    for start, group in dups.groupby('start'):
        # Prefer files that start with the patient ID. It appears they are from a new software version
        # Create a score tuple: (duration, starts_with_id)
        # This ensures we pick the longest file first. If durations are equal,
        # we pick the one starting with the patient ID.
        scores = group.apply(
            lambda row: (row['duration'], 1 if row['old_file_name'].startswith(pdir.name) else 0),
            axis=1
        )
        keep_idx = scores.idxmax()

        # Add all other indices in this group to the move list
        group_to_move = group.index.difference([keep_idx])
        to_move.extend(group.loc[group_to_move, 'old_file_name'].tolist())

    # Move the identified files
    for edf_name in to_move:
        move_edf(pdir.edf_dir / edf_name, 'Duplicate_start_within_patient')
    return to_move


def move_problematic_edfs(pdirs: list[PatientDir]):
    patient_info = pd.read_excel(PATHS.basic_patient_info, index_col='ID', dtype={'patientcode': str})
    patient_info['patientcode'] = patient_info['patientcode'].fillna('')

    # Get a list of with all groups of duplicate files
    print("Searching for duplicates...", end='\r')
    duplicate_groups = find_duplicate_files(PATHS.datasets_dir)
    # Iterate over patients and EDFs
    for pdir in pdirs:
        correct_patientcode = patient_info.loc[pdir.name, 'patientcode']
        move_problematic_ptnt_edfs(pdir, correct_patientcode, duplicate_groups)

    move_duplicates_within_patients(duplicate_groups, PATHS.datasets_dir)

    for pdir in pdirs:
        edfs = list_edfs(pdir)
        move_edfs_with_duplicate_start(pdir, edfs)


if __name__ == '__main__':
    move_problematic_edfs(PATHS.patient_dirs())
    # for pdir in PATHS.patient_dirs([Dataset.ultra2]):
    #     move_edfs_with_duplicate_start(pdir,
    #                                    edfs=pd.read_pickle(pickle_path(pdir.edf_files_table)))
