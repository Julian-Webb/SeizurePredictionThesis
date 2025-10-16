import logging
import time
from pathlib import Path

from config import PATHS
from data_cleaning.annotations_to_csv import annotations_to_csv
from data_cleaning.file_correction import file_correction, run_fdupes
from data_cleaning.rename_and_move_edf_data import rename_and_move_edf_data


def save_paths_as_txt(paths: list, save_path: Path):
    with open(save_path, 'a') as f:
        for item in paths:
            f.write(f'{item}\n\n')


def data_cleaning():
    """Apply all data cleaning steps in order"""
    input(f"Cleaning for {PATHS.base_dir}. Press enter to continue.")
    start_time = time.time()

    PATHS.data_cleaning_logs_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    logging.info('========== file_correction ==========')
    removed_duplicates = file_correction()
    save_paths_as_txt(removed_duplicates, PATHS.data_cleaning_logs_dir / 'removed_duplicates.txt')

    logging.info('========== annotations_to_csv ==========')
    annotations_to_csv()

    logging.info('========== rename_and_move_edf_data ==========')
    problematic_edfs = rename_and_move_edf_data(PATHS.problematic_edfs_dir)
    if not problematic_edfs.empty:
        PATHS.problematic_edfs_dir.mkdir(exist_ok=True)
        problematic_edfs.to_csv(PATHS.problematic_edfs_file, index=False)

    duplicate_groups = run_fdupes(PATHS.base_dir)
    if duplicate_groups:
        logging.warning(f'{len(duplicate_groups)} duplicate group remaining. See {PATHS.remaining_duplicates_file.name}')
        save_paths_as_txt(duplicate_groups, PATHS.remaining_duplicates_file)

    logging.info(f'Data cleaning completed in {time.time() - start_time:.2f} seconds')


if __name__ == '__main__':
    data_cleaning()
