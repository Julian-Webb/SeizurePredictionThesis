import logging
from pathlib import Path

from config.paths import PATHS, Dataset, PatientDir
from data_cleaning.convert_txt_annotations import convert_txt_annotations
from data_cleaning.combine_annotations import combine_annotations
from data_cleaning.file_correction import file_correction, run_fdupes
from data_cleaning.list_rename_move_edf_data import list_rename_move_edf_data
from scripts.remove_P4Hk23M7L_files_from_2000 import remove_P4Hk23M7L_files_from_2000
from utils.utils import FunctionTimer


def save_paths_as_txt(paths: list, save_path: Path):
    with open(save_path, 'a') as f:
        for item in paths:
            f.write(f'{item}\n\n')


def data_cleaning(ask_confirm: bool = True):
    """Apply all data cleaning steps in order"""
    if ask_confirm:
        input(f"Cleaning for {PATHS.base_dir}. Press enter to continue.")

    PATHS.data_cleaning_logs_dir.mkdir(exist_ok=True)
    logging.info("Starting data cleaning")

    logging.info('========== file_correction ==========')
    with FunctionTimer("file_correction"):
        removed_duplicates = file_correction()
    save_paths_as_txt(removed_duplicates, PATHS.data_cleaning_logs_dir / 'removed_duplicates.txt')

    logging.info('========== annotations_to_csv ==========')
    with FunctionTimer("annotations_to_csv"):
        convert_txt_annotations()

    logging.info('========== combine_annotations =============')
    with FunctionTimer("combine_annotations"):
        combine_annotations(PATHS.patient_dirs([Dataset.uneeg_extended]))

    logging.info('========== list_rename_move_edf_data ==========')
    with FunctionTimer("list_rename_move_edf_data"):
        problematic_edfs = list_rename_move_edf_data(PATHS.patient_dirs())
    if not problematic_edfs.empty:
        PATHS.problematic_edfs_dir.mkdir(exist_ok=True, parents=True)
        problematic_edfs.to_csv(PATHS.problematic_edfs_file.with_suffix('.csv'), index=False)

    # todo remove this later
    with FunctionTimer("======== remove_P4Hk23M7L_files_from_2000 ======="):
        remove_P4Hk23M7L_files_from_2000(PatientDir(PATHS.uneeg_extended_dir / 'P4Hk23M7L'))

    with FunctionTimer("run_fdupes"):
        duplicate_groups = run_fdupes(PATHS.base_dir)
    if duplicate_groups:
        logging.warning(
            f'{len(duplicate_groups)} duplicate group remaining. See {PATHS.remaining_duplicates_file.name}')
        save_paths_as_txt(duplicate_groups, PATHS.remaining_duplicates_file)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    data_cleaning()
