import time
import unittest

from config.paths import PATHS
from models.load_data import load_data


class TestLoadData(unittest.TestCase):
    def test_load_data(self):
        pdir = PATHS.patient_dirs()[7]
        print(f"\nPatient: {pdir.name}\n")
        count = 0
        for type_ in ['eeg', 'features']:
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
                        pdir,
                        type_,
                        subsample_shuffle_and_subselect_types,
                        train,
                        test
                    )

                    self.assertGreater(len(res), 0)

                    print(f'Test {count} passed in {time.perf_counter() - start:.2f} seconds', end='\n\n')
                    count += 1
