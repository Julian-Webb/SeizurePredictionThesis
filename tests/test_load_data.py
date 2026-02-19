import time
import unittest

import pandas as pd

from config.paths import PATHS
from models.load_data import load_data
from utils.io import pickle_path


class TestLoadData(unittest.TestCase):
    def test_load_data(self):
        pdir = PATHS.patient_dirs()[7]
        print(f"\nPatient: {pdir.name}\n")
        count = 0
        for type_ in ['features', 'eeg']:
            for train, test in [[True, False], [False, True], [True, True]]:
                for subsample_shuffle_and_subselect_types in [False, True]:
                    print(f'--- Test {count} ---')
                    print(f'{type_=},')
                    print(f'{train=},')
                    print(f'{test=},')
                    print(f'{subsample_shuffle_and_subselect_types=},')

                    start = time.perf_counter()
                    # Mostly, I just want to see if the function executes
                    res = load_data(
                        segs=pd.read_pickle(pickle_path(pdir.segments_table)),
                        type_=type_,
                        subsample_shuffle_and_subselect_types=subsample_shuffle_and_subselect_types,
                        train=train,
                        test=test,
                        split_idx=pd.read_pickle(pickle_path(pdir.train_test_split)).segment_index,
                        edf_dir=pdir.edf_dir,
                    )

                    # Assert all have the same length
                    self.assertEqual(len(res['x']), len(res['y']))
                    self.assertEqual(len(res['x']), len(res['index_and_start']))

                    self.assertGreater(len(res['x']), 0)

                    print(f'✓ passed in {time.perf_counter() - start:.2f} seconds', end='\n\n')
                    count += 1
