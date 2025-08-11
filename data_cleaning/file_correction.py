from pathlib import Path
import logging
import shutil
import os

from config import get_data_folders


def remove_png(uneeg_extended: Path):
    # This folder contains a random .png -> Delete it
    png_path = uneeg_extended / 'P4Hk23M7L' / 'P4Hk23M7L_WrongRecordingTime.png'
    try:
        png_path.unlink()
    except FileNotFoundError:
        logging.warning(f'The png {png_path} not found')


def clean_mac_files(directory: Path):
    """Removes macOS system files, such as .DS_Store and ._ files"""
    mac_patterns = [
        '._.DS_Store',  # Resource fork for .DS_Store
        '.DS_Store',  # Finder metadata
        '._*',  # All resource fork files
        '.AppleDouble',  # Apple Double format directory
        '.LSOverride'  # Finder custom attributes
    ]

    # go through all the files in the directory and delete them
    logging.info('===== Removing macOS system files =====')

    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Check exact matches
            if filename in mac_patterns:
                try:
                    Path(root, filename).unlink()
                    logging.info(f"Removed: {Path(root, filename)}")
                except OSError as e:
                    logging.warning(f"Error removing {Path(root, filename)}: {e}")

            # Check pattern matches (for ._*)
            elif filename.startswith('._'):
                try:
                    Path(root, filename).unlink()
                    logging.info(f"Removed: {Path(root, filename)}")
                except OSError as e:
                    logging.warning(f"Error removing {Path(root, filename)}: {e}")

    logging.info('===== Finished removing macOS system files =====')


def move_annotation_files(uneeg_extended: Path):
    # Certain patients in unneg_extended have their annotation files in a separate subfolder, rather than the patient folder
    # -> These are moved into the patient's base folder
    annotations_folders = [
        Path('A4RW34Z5B/annotations'),
        Path('E15T65H3Z/annotations'),
        Path('F5TW95P3X/Annotation text files'),
        Path('K53T36N7F/Annotation text files'),
        Path('L3GS57K2T/annotations'),
        Path('P4Hk23M7L/Annotation text files'),
    ]
    # prepend the base path
    annotations_folders = [uneeg_extended / folder for folder in annotations_folders]

    for annotation_folder in annotations_folders:
        if annotation_folder.exists():
            for annotation in annotation_folder.iterdir():
                new_path = annotation_folder.parent / annotation.name
                try:
                    shutil.move(annotation, new_path)
                    logging.info(f'Moved {annotation} from {annotation_folder} to {new_path}')
                except FileNotFoundError as e:
                    logging.warning(f'Could not move {annotation} from {annotation_folder} to {new_path}')
                    logging.warning(e)

            # delete folder after moving files
            try:
                annotation_folder.rmdir()
            except (FileNotFoundError, OSError) as e:
                logging.warning(f'Could not remove directory {annotation_folder}')
                logging.warning(e)
        else:
            logging.warning(f'The folder {annotation_folder} could not be found')


def handle_competition_data(competition_dir: Path):
    # In 20250501_SUBQ_SeizurePredictionCompetition_2025final, there's only a single folder TrainingData.
    # We move its contents to the parent folder
    training_dir = competition_dir / 'TrainingData'
    if training_dir.exists():
        for item in training_dir.iterdir():
            shutil.move(item, competition_dir / item.name)
        training_dir.rmdir()
        logging.info(f'Competition data was moved out of the TrainingData folder and the folder was deleted.')
    else:
        logging.warning(f"{training_dir} does not exist. It's contents can't be moved")

    # We rename the folders called "TrainingP1", "TrainingP2", etc. to just "P1", "P2", etc.
    for i in (1, 2, 3):
        old = competition_dir / f'TrainingP{i}'
        new = competition_dir / f'P{i}'
        try:
            old.rename(new)
            logging.info(f"{old} renamed to {new}")
        except FileNotFoundError:
            logging.warning(f"{old} could not be renamed to {new} because it was not found.")


def fix_filename_typos(uneeg_extended: Path):
    """Fix filenames containing 'Seiuzre' to 'Seizure' and remove trailing spaces."""
    for patient_folder in uneeg_extended.iterdir():
        if not patient_folder.is_dir():
            continue

        # Look for files with the typo
        for file_path in patient_folder.iterdir():
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
    with open(annotation_path, 'r') as f:
        lines = f.readlines()

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


def line_corrections(uneeg_extended: Path):
    # 'A4RW34Z5B_OUTPT_SUBQ_all automatic detections.txt' is missing as S somewhere
    line_correction(uneeg_extended / 'A4RW34Z5B' / 'A4RW34Z5B_OUTPT_SUBQ_all automatic detections.txt',
                    false_line_start='eizure-rhythmic\t2025-04-15 20:09:03.064\t2025-04-15 20:09:03.064\tStart V5f\n',
                    correct_line='Seizure-rhythmic\t2025-04-15 20:09:03.064\t2025-04-15 20:09:03.064\tStart V5f\n')

    # 'L3GS57K2T_OUTPT_SUBQ_all automatic detections.txt' has a double space in an inconvenient place
    line_correction(uneeg_extended / 'L3GS57K2T' / 'L3GS57K2T_OUTPT_SUBQ_all automatic detections.txt',
                    false_line_start='Seizure-rhythmic\t2025-01-09 09:39:55.927\t2025-01-09 09:39:55.927\tEnd  V5a\t\n',
                    correct_line='Seizure-rhythmic\t2025-01-09 09:39:55.927\t2025-01-09 09:39:55.927\tEnd V5a\t\n')

    # 'Seizure-rhythmic	2023-11-30 04:15:42.658	2023-11-30 04:15:42.658	end 5a' is actually the end of a seizure
    # This mistake may have been produced because it's the end of a seizure, as well as the end of a visit
    # This may have corrupted the annotation
    line_correction(uneeg_extended / 'E15T65H3Z' / 'E15T65H3Z_OUTPT_SUBQ_SeizureStartEnd.txt',
                    false_line_start='Seizure-rhythmic\t2023-11-30 04:15:42.658\t2023-11-30 04:15:42.658\tend 5a',
                    correct_line='User seizure marker\t2023-11-30 04:15:42.658\t2023-11-30 04:15:42.658\tSeizure End, end 5a\t\n')

    # Remove that same line from E15T65H3Z_OUTPT_SUBQ_CONSENSUS.txt
    line_correction(uneeg_extended / 'E15T65H3Z' / 'E15T65H3Z_OUTPT_SUBQ_CONSENSUS.txt',
                    false_line_start='Seizure-rhythmic\t2023-11-30 04:15:42.658\t2023-11-30 04:15:42.658\tend 5a',
                    correct_line='')


def file_correction(base_path: Path):
    """Perform different various unique actions to fix issues in the dataset.
    These could be performed manually, but this script acts as documentation."""
    for_mayo, uneeg_extended, competition_dir = get_data_folders(base_path)

    # remove a random png which should not be there
    remove_png(uneeg_extended)

    # remove macOS system files
    clean_mac_files(base_path)

    # move annotation files to the patient's folder
    move_annotation_files(uneeg_extended)

    handle_competition_data(competition_dir)

    # Rename files that end with SeiuzreStartEnd
    fix_filename_typos(uneeg_extended)

    # Take care of typos and lines with errors
    line_corrections(uneeg_extended)


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG')
    file_correction(Path('/data/home/webb/UNEEG_data'))
