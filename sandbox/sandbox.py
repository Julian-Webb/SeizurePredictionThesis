from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from config.paths import PATHS
from utils.io import pickle_path


def rename_edfs_start_mtz():
    for pdir in PATHS.patient_dirs():
        edfs = pd.read_pickle(pickle_path(pdir.edf_files_sheet))


        if not edfs.empty:
            edfs.rename(columns={'start': 'start_mtz'}, inplace=True)

            edfs.to_pickle(pickle_path(pdir.edf_files_sheet))
            # Make durations better readable for csv
            edfs_copy = edfs.copy()
            edfs_copy['duration'] = edfs_copy['duration'].apply(lambda x: str(x.to_pytimedelta()))
            edfs_copy.to_csv(pdir.edf_files_sheet.with_suffix('.csv'), index=False)


def check_files_contained_in_both():
    src = Path('/data/home/webb/seizure_annotations/STEP1_original_anns/toadd')
    dst = Path('/data/home/webb/seizure_annotations/STEP1_original_anns/')

    for src_pdir in sorted(list(src.iterdir())):
        dst_pdir = dst / src_pdir.name
        dst_files = [f.name for f in dst_pdir.iterdir()]
        for file in src_pdir.iterdir():
            if file.name in dst_files:
                # print(f'✓ File contained: {file.parent.name}/{file.name}')
                pass
            else:
                print(f'x File not in dst: {file.parent.name}/{file.name}')


def plot_something():
    plt.ion()
    plt.plot([1, 2, 3])
    plt.show()

if __name__ == '__main__':
    # check_files_contained_in_both()
    plot_something()