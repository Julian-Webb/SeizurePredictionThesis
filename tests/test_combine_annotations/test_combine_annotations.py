import unittest
from pathlib import Path

import pandas as pd

from data_cleaning.combine_annotations import combine_annotation_files


class TestCombineAnnotations(unittest.TestCase):
    def test_combine_annotation_files(self):
        data_dir = Path(__file__).parent / 'data'
        # files to combine
        files = ['marker', 'start_end', 'start_marker_end']
        paths = [data_dir / f"{file}.csv" for file in files]

        res_computed = combine_annotation_files(paths)

        res_correct = pd.read_csv(data_dir / 'combined_annotations_correct.csv')
        res_correct.fillna('', inplace=True)

        pd.testing.assert_frame_equal(res_correct, res_computed)
