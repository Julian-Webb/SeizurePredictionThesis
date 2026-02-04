import unittest

from numpy import array

from preprocessing.filter_signals import merge_intervals_with_gap


class TestFilterSignals(unittest.TestCase):
    def test_merge_intervals_with_gap(self):
        # Test with empty intervals
        ivs0 = array([])
        res0 = merge_intervals_with_gap(ivs0, 10)
        self.assertSequenceEqual(res0.tolist(), ivs0.tolist(), f'merge with empty intervals not equal.')

        # Test with single interval
        ivs1 = array([[0, 20]])
        res1 = merge_intervals_with_gap(ivs1, 10)
        self.assertSequenceEqual(res1.tolist(), ivs1.tolist(), f'merge with single interval not equal')

        # Test with regular intervals
        ivs2 = array([
            [0, 10],
            # no gap (merge)
            [10, 20],
            # small gap (merge)
            [25, 30],
            # large gap (don't merge)
            [50, 60]
        ])
        res2_correct = array([
            [0, 30],
            [50, 60]
        ])
        res2 = merge_intervals_with_gap(ivs2, 10)
        self.assertSequenceEqual(res2.tolist(), res2_correct.tolist(), f'merge with multiple intervals not equal')


