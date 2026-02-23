import shutil
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from config.paths import PATHS, PatientDir
from utils.io import pickle_path, save_dataframe_multiformat


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


def rename_horizon_to_intervention():
    for pdir in PATHS.patient_dirs():
        segs = pd.read_pickle(pickle_path(pdir.segments_table))
        segs['type'] = segs['type'].replace('horizon', 'intervention')
        save_dataframe_multiformat(segs, pdir.segments_table)


def combine_segment_probabilities_tables():
    pdirs = [
        PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-01-MINIFAKE'),
        PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-1'),
        PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-2'),
        PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-3'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-01'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-01-FAKE'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-03'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-04'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-05'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-07'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-12'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-15'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-16'),
        PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-17')
    ]

    for pdir in pdirs:
        print(f'Processing {pdir.name}...')
        # cnn = pd.read_pickle(pickle_path(pdir.predictions_dir / 'CNN' / 'segment_probabilities'))
        # ensemble = pd.read_pickle(pickle_path(pdir.predictions_dir / 'ensemble' / 'segment_probabilities'))
        #
        # cnn.rename(columns={'probabilities': 'CNN'}, inplace=True)
        # cnn['ensemble'] = ensemble['probabilities']
        #
        # save_dataframe_multiformat(cnn, pdir.segment_probabilities_table, csv_index=False)

        shutil.rmtree(pdir.predictions_dir / 'CNN')
        shutil.rmtree(pdir.predictions_dir / 'ensemble')


if __name__ == '__main__':
    pass