import unittest
import portion as P
from pandas import DataFrame, Timestamp
from model_eval.event_based_metrics import subtract_intervals

class TestTimeInFalseWarning(unittest.TestCase):
    def test_subtract_intervals(self):
        t = [
            Timestamp('2000-01-01 00:00:00'),
            Timestamp('2000-01-01 00:01:00'),
            Timestamp('2000-01-01 00:02:00'),
            Timestamp('2000-01-01 00:03:00'),
            Timestamp('2000-01-01 00:04:00'),
            Timestamp('2000-01-01 00:05:00'),
            Timestamp('2000-01-01 00:06:00'),
            Timestamp('2000-01-01 00:07:00'),
            Timestamp('2000-01-01 00:08:00'),
            Timestamp('2000-01-01 00:09:00'),
            Timestamp('2000-01-01 00:10:00'),
            Timestamp('2000-01-01 00:11:00'),
            Timestamp('2000-01-01 00:12:00'),
            Timestamp('2000-01-01 00:13:00'),
            Timestamp('2000-01-01 00:14:00'),
            Timestamp('2000-01-01 00:15:00'),
            Timestamp('2000-01-01 00:16:00'),
        ]

        base_ivs = [
            P.closed(t[0], t[3]), # Exclude middle portion (split)
            P.closed(t[5], t[8]), # Exclude start
            P.closed(t[10], t[12]), # Exclude end
            P.closed(t[14], t[15]), # Leave untouched

        ]

        subtract_ivs = [
            P.closed(t[1], t[2]),
            P.closed(t[4], t[6]),
            P.closed(t[11], t[13]),
        ]

        expected_ivs = [
            # From iv0:
            P.closed(t[0], t[1]),
            P.closed(t[2], t[3]),
            # From iv1:
            P.closed(t[6], t[8]),
            # From iv2:
            P.closed(t[10], t[11]),
            # From iv3:
            P.closed(t[14], t[15]),
        ]

        def ivs_to_df(ivs):
            rows = []
            for iv in ivs:
                rows.append({'start': iv.lower, 'end': iv.upper})
            return DataFrame(rows)

        base_df = ivs_to_df(base_ivs)
        expected_df = ivs_to_df(expected_ivs)

        subtract_iv = P.empty()
        for iv in subtract_ivs:
            subtract_iv |= iv

        res = subtract_intervals(base_df, subtract_iv)
        self.assertTrue(res.equals(expected_df))