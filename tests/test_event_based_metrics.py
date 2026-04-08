import unittest

import pandas as pd
from pandas import DataFrame, Timedelta, Timestamp

from model_eval.event_based_metrics import (
    event_based_metrics_for_ptnt,
    non_overlapping_interval_duration,
)


# noinspection PyTypeChecker
class TestEventBasedMetrics(unittest.TestCase):
    THRESHOLD = 0.5
    SPH_DURATION = Timedelta(minutes=4)

    @staticmethod
    def _make_clips(preictal_scores, interictal_scores, valid_preictal=None, valid_interictal=None):
        base = Timestamp('2020-01-01 00:00:00')
        rows = []

        if valid_preictal is None:
            valid_preictal = [True] * len(preictal_scores)
        if valid_interictal is None:
            valid_interictal = [True] * len(interictal_scores)

        for i, (score, valid) in enumerate(zip(preictal_scores, valid_preictal)):
            rows.append({
                'end_mtz': base + Timedelta(minutes=10 * i),
                'preictal': True,
                'score': score,
                'valid': valid,
            })

        offset = len(preictal_scores)
        for i, (score, valid) in enumerate(zip(interictal_scores, valid_interictal)):
            rows.append({
                'end_mtz': base + Timedelta(minutes=10 * (offset + i)),
                'preictal': False,
                'score': score,
                'valid': valid,
            })

        return DataFrame(rows)

    @staticmethod
    def _seizures_for_preictal_clips(n_preictal):
        base = Timestamp('2020-01-01 00:00:00')
        # Put each seizure 2 minutes after each preictal clip end, inside a 4-minute SPH.
        return pd.to_datetime([
            base + Timedelta(minutes=10 * i + 2)
            for i in range(n_preictal)
        ]).to_numpy()

    def test_rel_tifw_is_one_when_all_interictal_are_predicted_true(self):
        clips = self._make_clips(preictal_scores=[0.9, 0.9], interictal_scores=[0.9, 0.8, 1.0])
        szr_starts = self._seizures_for_preictal_clips(2)

        metrics, _ = event_based_metrics_for_ptnt(
            clips,
            szr_starts,
            score_col='score',
            thresholds=[self.THRESHOLD],
            intervention_duration=Timedelta(0),
            sph_duration=self.SPH_DURATION,
        )

        self.assertAlmostEqual(metrics.loc[self.THRESHOLD, 'rel_tifw'], 1.0)

    def test_rel_tifw_is_zero_when_all_interictal_are_predicted_false(self):
        clips = self._make_clips(preictal_scores=[0.9, 0.9], interictal_scores=[0.1, 0.2, 0.49])
        szr_starts = self._seizures_for_preictal_clips(2)

        metrics, _ = event_based_metrics_for_ptnt(
            clips,
            szr_starts,
            score_col='score',
            thresholds=[self.THRESHOLD],
            intervention_duration=Timedelta(0),
            sph_duration=self.SPH_DURATION,
        )

        self.assertAlmostEqual(metrics.loc[self.THRESHOLD, 'rel_tifw'], 0.0)

    def test_rel_szrs_pred_is_one_when_all_preictal_are_predicted_true(self):
        clips = self._make_clips(preictal_scores=[0.8, 0.7, 0.9], interictal_scores=[0.2, 0.2])
        szr_starts = self._seizures_for_preictal_clips(3)

        metrics, szrs_pred = event_based_metrics_for_ptnt(
            clips,
            szr_starts,
            score_col='score',
            thresholds=[self.THRESHOLD],
            intervention_duration=Timedelta(0),
            sph_duration=self.SPH_DURATION,
        )

        self.assertAlmostEqual(metrics.loc[self.THRESHOLD, 'rel_szrs_pred'], 1.0)
        self.assertTrue(szrs_pred.loc[self.THRESHOLD].all())

    def test_rel_szrs_pred_is_zero_when_all_preictal_are_predicted_false(self):
        clips = self._make_clips(preictal_scores=[0.1, 0.2, 0.3], interictal_scores=[0.1, 0.1])
        szr_starts = self._seizures_for_preictal_clips(3)

        metrics, szrs_pred = event_based_metrics_for_ptnt(
            clips,
            szr_starts,
            score_col='score',
            thresholds=[self.THRESHOLD],
            intervention_duration=Timedelta(0),
            sph_duration=self.SPH_DURATION,
        )

        self.assertAlmostEqual(metrics.loc[self.THRESHOLD, 'rel_szrs_pred'], 0.0)
        self.assertFalse(szrs_pred.loc[self.THRESHOLD].any())

    def test_non_overlapping_interval_duration_truncates_overlaps(self):
        starts = pd.Series(pd.to_datetime([
            '2020-01-01 00:00:00',
            '2020-01-01 00:03:00',
            '2020-01-01 00:08:00',
        ]))
        ends = pd.Series(pd.to_datetime([
            '2020-01-01 00:05:00',
            '2020-01-01 00:10:00',
            '2020-01-01 00:12:00',
        ]))

        total_duration, ivs = non_overlapping_interval_duration(starts, ends)

        self.assertEqual(total_duration, Timedelta(minutes=12))
        self.assertEqual(ivs.loc[0, 'end'], starts.loc[1])
        self.assertEqual(ivs.loc[1, 'end'], starts.loc[2])
        self.assertEqual(ivs.loc[2, 'end'], ends.loc[2])

    def test_invalid_clips_are_excluded_before_metrics(self):
        clips = self._make_clips(
            preictal_scores=[0.9],
            interictal_scores=[0.1, 0.95],
            valid_interictal=[True, False],
        )
        szr_starts = self._seizures_for_preictal_clips(1)

        metrics, _ = event_based_metrics_for_ptnt(
            clips,
            szr_starts,
            score_col='score',
            thresholds=[self.THRESHOLD],
            intervention_duration=Timedelta(0),
            sph_duration=self.SPH_DURATION,
        )

        # If invalid clips are ignored, the only valid interictal clip is below threshold.
        self.assertAlmostEqual(metrics.loc[self.THRESHOLD, 'rel_tifw'], 0.0)

    def test_szrs_pred_series_has_expected_multiindex(self):
        clips = self._make_clips(preictal_scores=[0.9, 0.4], interictal_scores=[0.1])
        szr_starts = self._seizures_for_preictal_clips(2)
        thresholds = [0.3, 0.8]

        _, szrs_pred = event_based_metrics_for_ptnt(
            clips,
            szr_starts,
            score_col='score',
            thresholds=thresholds,
            intervention_duration=Timedelta(0),
            sph_duration=self.SPH_DURATION,
        )

        self.assertEqual(szrs_pred.index.names, ['threshold', 'seizure_start'])
        self.assertEqual(len(szrs_pred), len(thresholds) * len(szr_starts))


if __name__ == '__main__':
    unittest.main()
