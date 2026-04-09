import time
import unittest

import pandas as pd

from config import PATHS
from models.load_data import load_data


class TestLoadData(unittest.TestCase):
    def test_load_data(self):
        pdir = PATHS.patient_dirs()[0]
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
                        segs=pd.read_pickle(pdir.segments_table.pickle),
                        type_=type_,
                        subsample_shuffle_and_subselect_types=subsample_shuffle_and_subselect_types,
                        train=train,
                        test=test,
                        test_start_mtz=pd.read_pickle(pdir.dataset_partition.pickle).loc['test', 'start_mtz'],
                        edf_dir=pdir.edf_dir,
                    )

                    # Assert all have the same length
                    self.assertEqual(len(res['x']), len(res['y']))
                    self.assertEqual(len(res['x']), len(res['index_and_start']))

                    self.assertGreater(len(res['x']), 0)

                    print(f'✓ passed in {time.perf_counter() - start:.2f} seconds', end='\n\n')
                    count += 1
