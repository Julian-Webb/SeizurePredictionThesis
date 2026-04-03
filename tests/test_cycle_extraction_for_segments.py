import unittest

import numpy as np
import pandas as pd
from config.intervals import SEGMENT
from cycle_extraction.cycle_extraction_for_segments import split_dataframe_by_nan_gaps, cycle_extraction_for_ptnt, \
    map_events_to_interval_index


class TestCycleExtraction(unittest.TestCase):
    def test_split_by_nan_gaps_correct_splitting(self):
        df = pd.DataFrame({
            "start_mtz": pd.date_range("2026-01-01 00:00:00", periods=6, freq=SEGMENT.exact_dur),
            "feature_1": [1.0, 2.0, np.nan, np.nan, 5.0, 6.0],
            "feature_2": [10, 20, np.nan, np.nan, 50, 60]
        })
        chunks = split_dataframe_by_nan_gaps(df, ["feature_1", "feature_2"])
        self.assertEqual(len(chunks), 2)
        self.assertTrue((chunks[0]["feature_1"] == [1.0, 2.0]).all())
        self.assertTrue((chunks[1]["feature_1"] == [5.0, 6.0]).all())

    def test_event_indices_mapping_to_chunks(self):
        start_mtz = pd.date_range("2026-01-01 00:00:00", periods=5, freq=SEGMENT.exact_dur).to_numpy()
        end_mtz = start_mtz + SEGMENT.exact_dur
        event_timestamps = np.array(["2026-01-01 00:00:10", "2026-01-01 00:01:01"], dtype="datetime64")
        event_indices_correct = np.array([0, 4])

        event_indices_res = map_events_to_interval_index(event_timestamps, start_mtz, end_mtz)
        np.testing.assert_array_equal(event_indices_correct, event_indices_res)

    def test_cycle_extraction_output_metrics_format(self):
        n_days = 10
        n_segs = round(n_days * 24 * 60 * 60 / SEGMENT.exact_dur.total_seconds())
        seg_features = pd.DataFrame({
            "start_mtz": pd.date_range("2026-01-01 00:00:00", periods=n_segs, freq=SEGMENT.exact_dur),
            "feature_1": np.random.rand(n_segs),
            "feature_2": np.random.rand(n_segs),
        })
        n_features = 2

        event_timestamps = {"events_1": np.array(["2026-01-02 12:00:00"], dtype="datetime64")}
        n_event_types = len(event_timestamps)
        metrics, filtered_features, event_phases_per_type_per_feat = \
            cycle_extraction_for_ptnt(seg_features, event_timestamps, feature_names=["feature_1", "feature_2"])

        # Check metrics structure
        self.assertFalse(metrics.empty)
        self.assertIn("events_1", metrics.columns.levels[0])
        self.assertIn("plv", metrics.columns.levels[1])
        self.assertGreaterEqual(metrics.shape[0], 2)

        # Check filtered features structure
        self.assertEqual(len(filtered_features), len(seg_features))
        self.assertIn("feature_1", filtered_features.columns)
        self.assertIn("feature_2", filtered_features.columns)

        # Check Event Phases
        self.assertEqual(len(event_phases_per_type_per_feat), n_event_types)
        for event_type, event_phases_per_feat in event_phases_per_type_per_feat.items():
            self.assertEqual(len(event_phases_per_feat), n_features, f"Wrong number of event phases for {event_type}")
