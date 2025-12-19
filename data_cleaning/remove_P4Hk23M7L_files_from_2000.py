import logging

import pandas as pd

from config.paths import PatientDir, PATHS
from utils.io import pickle_path


# noinspection PyPep8Naming
def remove_P4Hk23M7L_files_from_2000(ptnt_dir: PatientDir):
    """
    This is meant to be called after data_cleaning
    He doesn't have annotations in the 2000s, so only the edf files must be removed from the list and the directory
    """
    if not ptnt_dir.exists():
        logging.error(f"PatientDir doesn't exist: {ptnt_dir}")
        return

    edfs = pd.read_pickle(pickle_path(ptnt_dir.edf_files_sheet))
    # noinspection PyTypeChecker
    mask = edfs['start'] < pd.Timestamp('2001-01-01')
    bad_edfs = edfs[mask].copy()

    # Move files
    rel_path = ptnt_dir.edf_dir.relative_to(PATHS.base_dir)
    bad_edfs_dir = PATHS.problematic_edfs_dir / rel_path
    bad_edfs_dir.mkdir(parents=True, exist_ok=True)
    bad_edfs['old_file_path'] = ptnt_dir.edf_dir / bad_edfs['file_name']
    bad_edfs['new_file_path'] = bad_edfs_dir / bad_edfs['file_name']
    bad_edfs.apply(lambda row: row.old_file_path.rename(row.new_file_path), axis=1)

    # Update EDF lists
    bad_edfs.to_csv(PATHS.problematic_edfs_dir / 'P4Hk23M7L_files_from_2000.csv', index=False)

    edfs = edfs[~mask]
    edfs.to_pickle(pickle_path(ptnt_dir.edf_files_sheet))
    # Make durations better readable for csv
    edfs['duration'] = edfs['duration'].apply(lambda x: str(x.to_pytimedelta()))
    edfs.to_csv(ptnt_dir.edf_files_sheet.with_suffix('.csv'), index=False)


if __name__ == '__main__':
    remove_P4Hk23M7L_files_from_2000(PatientDir(PATHS.uneeg_extended_dir / 'P4Hk23M7L'))
