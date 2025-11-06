import itertools
import subprocess
from datetime import datetime
from pathlib import Path
import logging
import shutil
import os

import pyedflib

from config.paths import PATHS, PatientDir, Dataset

MAC_PATTERNS = [
    '._.DS_Store',  # Resource fork for .DS_Store
    '.DS_Store',  # Finder metadata
    '._*',  # All resource fork files
    '.AppleDouble',  # Apple Double format directory
    '.LSOverride'  # Finder custom attributes
]


def remove_png():
    # This folder contains a random .png -> Delete it
    png_path = PATHS.uneeg_extended_dir / 'P4Hk23M7L' / 'P4Hk23M7L_WrongRecordingTime.png'
    try:
        png_path.unlink()
    except FileNotFoundError:
        logging.warning(f'The png {png_path} not found')


def clean_mac_files(directory: Path):
    """Removes macOS system files, such as .DS_Store and ._ files"""
    # go through all the files in the directory and delete them
    # logging.info('===== Removing macOS system files =====')
    removed_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Check exact matches
            path = Path(root, filename)
            if filename in MAC_PATTERNS:
                try:
                    path.unlink()
                    removed_files.append(path)
                    # logging.info(f"Removed: {Path(root, filename)}")
                except OSError as e:
                    logging.warning(f"Error removing {path}: {e}")

            # Check pattern matches (for ._*)
            elif filename.startswith('._'):
                try:
                    path.unlink()
                    removed_files.append(path)
                    # logging.info(f"Removed: {path}")
                except OSError as e:
                    logging.warning(f"Error removing {path}: {e}")

    logging.info(f"Removed {len(removed_files)} mac system files")
    # logging.info('===== Finished removing macOS system files =====')


def move_annotation_files():
    """
    Seizure annotations are moved into the appropriate folder for all patients
    """
    # make the seizure_annotations dir for patients in for_mayo and uneeg_extended
    for patient_dir in PATHS.patient_dirs(Dataset.for_mayo, Dataset.uneeg_extended):
        patient_dir.szr_anns_dir.mkdir(exist_ok=True)

    old_anns_dirs = [
        Path('A4RW34Z5B/annotations'),
        Path('E15T65H3Z/annotations'),
        Path('F5TW95P3X/Annotation text files'),
        Path('K53T36N7F/Annotation text files'),
        Path('L3GS57K2T/annotations'),
        Path('P4Hk23M7L/Annotation text files'),
    ]
    # prepend the base path
    old_anns_dirs = [PATHS.uneeg_extended_dir / folder for folder in old_anns_dirs]

    # rename the folder for uneeg_extended
    for folder in old_anns_dirs:
        patient_dir = PatientDir(folder.parent)
        new_name = patient_dir.szr_anns_original_dir
        if folder != new_name:
            try:
                folder.rename(new_name)
                logging.info(f"Renamed {folder} to {new_name}")
            except FileNotFoundError:
                logging.warning(f"Could not find {folder} to rename")

    # For patient D63Q51K2N in uneeg_extended, there's no annotations folder, just like with the for_mayo patients
    # For for_mayo, create an original annotations folder and move the file into it.
    for patient_dir in [*PATHS.for_mayo_dir.iterdir(), PATHS.uneeg_extended_dir / 'D63Q51K2N']:
        patient_dir = PatientDir(patient_dir)
        patient_dir.szr_anns_original_dir.mkdir(exist_ok=True)
        for annotation in patient_dir.glob('*.txt'):
            try:
                annotation.rename(patient_dir.szr_anns_original_dir / annotation.name)
            except FileNotFoundError:
                logging.warning(f"Could not find {annotation} to move.")


def handle_competition_data():
    """In 20250501_SUBQ_SeizurePredictionCompetition_2025final, a folder is created for each participant and their
    edf data.
    """
    competition_dir = PATHS.competition_dir
    # Delete the TrainingData folder (it's the only folder in the base directory)
    training_dir = competition_dir / 'TrainingData'
    if training_dir.exists():
        for item in training_dir.iterdir():
            shutil.move(item, competition_dir / item.name)
        training_dir.rmdir()
        logging.info(f'Competition data was moved out of the TrainingData folder and the folder was deleted.')
    else:
        logging.warning(f"{training_dir} does not exist. It's contents can't be moved")

    # make folders for the patients
    for i in (1, 2, 3):
        patient = f'P{i}'
        patient_dir = PatientDir(competition_dir / patient)
        patient_dir.mkdir(exist_ok=True)
        # move their edf data into the patient folder
        old = competition_dir / f'TrainingP{i}'
        if old.exists():
            new = patient_dir.original_edf_dir
            old.rename(new)


def fix_filename_typos():
    """Fix filenames containing 'Seiuzre' to 'Seizure' and remove trailing spaces."""
    for patient_dir in PATHS.patient_dirs(Dataset.uneeg_extended):
        if not patient_dir.szr_anns_original_dir.is_dir():
            continue

        # Look for files with the typo
        for file_path in patient_dir.szr_anns_original_dir.iterdir():
            new_name = file_path.name.replace("Seiuzre", "Seizure").lstrip()
            if new_name != file_path.name:
                # Create new filename with correct spelling
                new_path = file_path.parent / new_name

                try:
                    file_path.rename(new_path)
                    logging.info(f"Renamed {file_path.name} to {new_path.name}")
                except FileNotFoundError:
                    logging.warning(f"Could not find file {file_path}")
                except FileExistsError:
                    logging.warning(f"Cannot rename {file_path} - destination {new_path} already exists")


def line_correction(annotation_path: Path, false_line_start: str, correct_line: str):
    """Corrects individual typos throughout the files."""

    # First read the file line by line
    try:
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.warning(f"Could not find {annotation_path} to correct")
        return

    # Find and fix the specific line
    fixed = False
    for i, line in enumerate(lines):
        if line.startswith(false_line_start):
            lines[i] = correct_line
            fixed = True
            break

    # Only write back if we actually found and fixed the line
    if fixed:
        with open(annotation_path, 'w') as f:
            f.writelines(lines)
            logging.info(f"Replaced line '{false_line_start}' with '{correct_line}' in {annotation_path.name}")
    else:
        logging.warning(f"Did not find the line needing correction in {annotation_path.name} - no changes made")


def line_corrections():
    uneeg_extended = PATHS.uneeg_extended_dir
    # 'A4RW34Z5B_OUTPT_SUBQ_all automatic detections.txt' is missing as S somewhere
    line_correction(
        PatientDir(
            uneeg_extended / 'A4RW34Z5B').szr_anns_original_dir / 'A4RW34Z5B_OUTPT_SUBQ_all automatic detections.txt',
        false_line_start='eizure-rhythmic\t2025-04-15 20:09:03.064\t2025-04-15 20:09:03.064\tStart V5f',
        correct_line='Seizure-rhythmic\t2025-04-15 20:09:03.064\t2025-04-15 20:09:03.064\tStart V5f\n')

    # 'L3GS57K2T_OUTPT_SUBQ_all automatic detections.txt' has a double space in an inconvenient place
    line_correction(
        PatientDir(
            uneeg_extended / 'L3GS57K2T').szr_anns_original_dir / 'L3GS57K2T_OUTPT_SUBQ_all automatic detections.txt',
        false_line_start='Seizure-rhythmic\t2025-01-09 09:39:55.927\t2025-01-09 09:39:55.927\tEnd  V5a',
        correct_line='Seizure-rhythmic\t2025-01-09 09:39:55.927\t2025-01-09 09:39:55.927\tEnd V5a\t\n')

    # 'Seizure-rhythmic	2023-11-30 04:15:42.658	2023-11-30 04:15:42.658	end 5a' is actually the end of a seizure
    # This mistake may have been produced because it's the end of a seizure, as well as the end of a visit
    # This may have corrupted the annotation
    E15T65H3Z_anns_original_dir = PatientDir(uneeg_extended / 'E15T65H3Z').szr_anns_original_dir
    line_correction(
        E15T65H3Z_anns_original_dir / 'E15T65H3Z_OUTPT_SUBQ_SeizureStartEnd.txt',
        false_line_start='Seizure-rhythmic\t2023-11-30 04:15:42.658\t2023-11-30 04:15:42.658\tend 5a',
        correct_line='User seizure marker\t2023-11-30 04:15:42.658\t2023-11-30 04:15:42.658\tSeizure End, end 5a\t\n')

    # Remove that same line from E15T65H3Z_OUTPT_SUBQ_CONSENSUS.txt
    line_correction(
        E15T65H3Z_anns_original_dir / 'E15T65H3Z_OUTPT_SUBQ_CONSENSUS.txt',
        false_line_start='Seizure-rhythmic\t2023-11-30 04:15:42.658\t2023-11-30 04:15:42.658\tend 5a',
        correct_line='')

    # E15T65H3Z_EMU_SUBQ_CONSENSUS.txt should contain no seizures.
    # (Seizures from D63Q51K2N have been duplicated into it)
    # First read the file line by line
    path = E15T65H3Z_anns_original_dir / 'E15T65H3Z_EMU_SUBQ_CONSENSUS.txt'
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        correct_lines = [lines[0], 'No Seizures']
        with open(path, 'w') as f:
            f.writelines(correct_lines)
            logging.info(f"Removed all seizures from {path.name}")
    except FileNotFoundError:
        logging.warning(f"Could not find {path} to remove seizures from")


def run_fdupes(base_path: Path):
    """Runs fdupes on the base path to find duplicated files."""
    result = subprocess.run(
        ["fdupes", "-r", base_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"fdupes failed: {result.stderr}")

    # duplicates groups are separated by a blank line
    duplicate_groups = result.stdout.split("\n\n")

    return duplicate_groups


def remove_duplicates():
    """Removes duplicated edf files used fdupes. This must be installed.
    Duplicate patterns:
    1. Many files appear to have simply been duplicated within the same folder -> these are removed
    2. Regarding patient M39K4B3C, visit 5d appears to have been copied into visit 5e -> the files are removed from V5e
    3. Patient P4Hk23M7L and A4RW34Z5B share certain files -> A4RW34Z5B's files are removed
    """
    duplicate_groups = run_fdupes(PATHS.base_dir)

    # with open(base_path / 'duplicates.txt', "w") as f:
    #     f.writelines(duplicate_groups)

    # loop through the groups and delete files
    unprocessed_duplicate_groups = []
    removed_files = []

    def remove_file(path: str):
        Path(path).unlink()
        removed_files.append(path)
        # logging.debug(f"Removed {path}")

    while len(duplicate_groups) > 0:
        duplicate_group = duplicate_groups.pop(0)
        files = duplicate_group.split("\n")
        # we record the properties of the different files
        visits = []
        patients = []
        for f in files:
            path = Path(f)
            visits.append(path.parent.name)
            patients.append(path.parent.parent.name)

        # If the patient and visit are the same, the duplicates are removed
        if len(set(patients)) == 1 and len(set(visits)) == 1:
            for i in range(1, len(files)):
                remove_file(files[i])
        # Delete duplicated files between visits for patient M39K4B3C
        elif len(set(patients)) == 1 and patients[0] == 'M39K4B3C' and 'V5d' in visits and 'V5e' in visits:
            for i, f in enumerate(files):
                if visits[i] == 'V5e':
                    remove_file(f)
        # Handle patient P4Hk23M7L and A4RW34Z5B
        elif 'P4Hk23M7L' in patients and 'A4RW34Z5B' in patients:
            for i, f in enumerate(files):
                if patients[i] == 'A4RW34Z5B':
                    remove_file(f)
        else:
            unprocessed_duplicate_groups.append(duplicate_group)

    if unprocessed_duplicate_groups:
        logging.warning(f"=== Unprocessed duplicate groups: ===")
        for group in unprocessed_duplicate_groups:
            for line in group.split('\n'):
                logging.warning(line)
            logging.warning("")
        logging.warning(f"=== End unprocessed duplicate groups ===")

    return removed_files


def remove_additional_duplicates() -> list[Path]:
    """Remove duplicates that were not caught by fdupes. For patient M39K4B3C, these are files that from Visit V5b that are also copied into V5d or V5e
    :return: a list of removed duplicate files."""
    patient_dir = PATHS.for_mayo_dir / 'M39K4B3C'
    removed_files = []

    # The bounds of visit 5b:
    time_format = "%Y-%m-%d %H:%M:%S.%f"
    v5a_end = datetime.strptime('2021-06-15 11:00:00.015300', time_format)
    v5c_start = datetime.strptime('2021-07-15 10:39:44.0', time_format)

    v5b_file_names = [edf_path.name for edf_path in (patient_dir / 'V5b').iterdir()]
    v5d_files = patient_dir.glob('V5d/*.edf')
    v5e_files = patient_dir.glob('V5e/*.edf')
    potential_duplicates = itertools.chain(v5d_files, v5e_files)

    # loop through all potential duplicates and see if they are duplicates
    for edf_path in potential_duplicates:
        # Check if it's contained in V5b
        if edf_path.name in v5b_file_names:
            # Assert that the datetime corresponds to V5b
            try:
                edf = pyedflib.EdfReader(str(edf_path))
            except OSError:
                logging.warning(f"Could not open {edf_path} to read the start datetime")
                continue

            start_dt = edf.getStartdatetime()
            if v5a_end < start_dt < v5c_start:
                # logging.debug(f"Removing {edf_path}")
                removed_files.append(edf_path)
                edf_path.unlink()
            else:
                raise ValueError(
                    f"File {edf_path}'s start datetime is outside of V5b bounds. Thus it should be removed somewhere else.")

    return removed_files


def file_correction() -> list[Path]:
    """Perform different various unique actions to fix issues in the dataset.
    These could be performed manually, but this script acts as documentation.
    :return: A list of removed files"""

    # remove a random png which should not be there
    remove_png()

    clean_mac_files(PATHS.base_dir)

    # move annotation files to the patient's annotation folder
    move_annotation_files()

    handle_competition_data()

    fix_filename_typos()

    # Take care of typos and lines with errors
    line_corrections()

    logging.info('Removing duplicates...')
    removed_duplicates = remove_duplicates()
    removed_duplicates += remove_additional_duplicates()

    logging.info(f"Removed {len(removed_duplicates)} duplicate files")
    return removed_duplicates


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG')
    file_correction()
