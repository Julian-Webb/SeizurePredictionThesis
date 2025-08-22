import logging
import time
from pathlib import Path

from config import Paths
from data_cleaning.annotations_to_csv import annotations_to_csv
from data_cleaning.combine_annotations import combine_annotations
from data_cleaning.file_correction import file_correction, run_fdupes
from data_cleaning.rename_and_move_edf_data import rename_and_move_edf_data


def save_paths_as_txt(paths: list, save_path: Path):
    with open(save_path, 'a') as f:
        for item in paths:
            f.write(f'{item}\n\n')


def data_cleaning():
    """Apply all data cleaning steps in order"""
    input(f"Cleaning for {Paths.BASE_DIR}. Press enter to continue.")
    start_time = time.time()

    Paths.DATA_CLEANING_LOG_DIR.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    logging.info('========== file_correction ==========')
    removed_duplicates = file_correction(Paths.BASE_DIR)
    save_paths_as_txt(removed_duplicates, Paths.DATA_CLEANING_LOG_DIR / 'removed_duplicates.txt')

    logging.info('========== annotations_to_csv ==========')
    annotations_to_csv(Paths.FOR_MAYO_DIR, Paths.UNEEG_EXTENDED_DIR, Paths.COMPETITION_DIR)

    logging.info('========== combine_annotations ==========')
    combine_annotations(Paths.UNEEG_EXTENDED_DIR.iterdir())

    logging.info('========== rename_and_move_edf_data ==========')
    problematic_edfs = rename_and_move_edf_data(Paths.PROBLEMATIC_EDFS_DIR)
    if not problematic_edfs.empty:
        Paths.PROBLEMATIC_EDFS_DIR.mkdir(exist_ok=True)
        problematic_edfs.to_csv(Paths.PROBLEMATIC_EDFS_FILE, index=False)

    duplicate_groups = run_fdupes(Paths.BASE_DIR)
    if duplicate_groups:
        logging.warning(f'{len(duplicate_groups)} duplicate group remaining. See {Paths.REMAINING_DUPLICATES_FILE.name}')
        save_paths_as_txt(duplicate_groups, Paths.REMAINING_DUPLICATES_FILE)

    logging.info(f'Data cleaning completed in {time.time() - start_time:.2f} seconds')


if __name__ == '__main__':
    data_cleaning()
