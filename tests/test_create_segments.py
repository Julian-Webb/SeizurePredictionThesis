import unittest

import numpy as np
import pandas as pd

from config.constants import SAMPLING_FREQUENCY_HZ
from config.intervals import SEGMENT, PREICTAL, INTER_PRE, INTERVENTION, POSTICTAL, INTER_POST, INTERICTAL
# Now import the module under test
from preprocessing import create_segments
from utils.edf_utils import time_to_index


class TestSegmentTables(unittest.TestCase):
    def test_find_existing_segs_marks_exists_and_start_index(self):
        # Setup: one EDF file starting at time 0
        file_start = pd.Timestamp('2020-01-01 00:00:00')
        edfs = pd.DataFrame(
            [{'file_name': 'f1', 'start_mtz': file_start, 'end_mtz': file_start + pd.Timedelta(minutes=10)}])

        # valid interval that covers the first three segments
        valid_edf_intervals = pd.DataFrame(
            [{'file_name': 'f1', 'start_mtz': file_start, 'end_mtz': file_start + 3 * SEGMENT.exact_dur}])

        # Create 8 sequential segments
        starts = [file_start + i * SEGMENT.exact_dur for i in range(8)]
        segs = pd.DataFrame({'start_mtz': starts})
        segs['end_mtz'] = segs['start_mtz'] + SEGMENT.exact_dur
        segs['type'] = np.nan
        segs['lead_szr'] = np.nan
        segs['exists'] = False
        segs['file'] = None
        segs['start_index'] = None

        res = create_segments.find_existing_segs(valid_edf_intervals, edfs, segs.copy())

        # First three segments should exist
        self.assertTrue(res.loc[0, 'exists'])
        self.assertTrue(res.loc[1, 'exists'])
        self.assertTrue(res.loc[2, 'exists'])
        # Later segments should not exist
        self.assertFalse(res.loc[3, 'exists'])

        # start_index should be set for existing segments and match time_to_index rounded
        expected_idx0 = round(time_to_index(file_start, res.loc[0, 'start_mtz'], SAMPLING_FREQUENCY_HZ))
        self.assertEqual(res.loc[0, 'start_index'], expected_idx0)

    def test_find_seg_type_assigns_interval_labels(self):
        # Create a seizure start
        szr_start = pd.Timestamp('2020-01-01 12:00:00')
        szrs = pd.DataFrame([{'start_mtz': szr_start, 'lead': True}])

        # Intervals before seizure considered together (closest to seizure outward): INTERVENTION, PREICTAL, INTER_PRE
        # find_seg_type computes iv_start = szr_start - (INTER_PRE + PREICTAL + INTERVENTION)
        iv_start = szr_start - (INTER_PRE.exact_dur + PREICTAL.exact_dur + INTERVENTION.exact_dur)

        # Buffers that should be INTERICTAL
        buffer_before = pd.Timedelta(minutes=30)
        buffer_after = pd.Timedelta(minutes=30)

        start_time = iv_start - buffer_before
        end_time = szr_start + POSTICTAL.exact_dur + INTER_POST.exact_dur + buffer_after

        # Generate segments from start_time up to end_time
        starts = []
        t = start_time
        while t < end_time:
            starts.append(t)
            t = t + SEGMENT.exact_dur

        segs = pd.DataFrame({'start_mtz': starts})
        segs['end_mtz'] = segs['start_mtz'] + SEGMENT.exact_dur
        segs['type'] = np.nan
        segs['lead_szr'] = np.nan

        res = create_segments.find_seg_type(segs.copy(), szrs)

        # Compute exact interval boundaries used in find_seg_type
        t0 = iv_start
        t1 = t0 + INTER_PRE.exact_dur
        t2 = t1 + PREICTAL.exact_dur
        t3 = t2 + INTERVENTION.exact_dur
        t4 = t3 + POSTICTAL.exact_dur
        t5 = t4 + INTER_POST.exact_dur

        # Build expected labels and lead flags for each generated segment
        expected_labels = []
        expected_lead = []
        for s in starts:
            if t0 <= s < t1:
                expected_labels.append(INTER_PRE.label)
                expected_lead.append(True)
            elif t1 <= s < t2:
                expected_labels.append(PREICTAL.label)
                expected_lead.append(True)
            elif t2 <= s < t3:
                expected_labels.append(INTERVENTION.label)
                expected_lead.append(True)
            elif t3 <= s < t4:
                expected_labels.append(POSTICTAL.label)
                expected_lead.append(True)
            elif t4 <= s < t5:
                expected_labels.append(INTER_POST.label)
                expected_lead.append(True)
            else:
                expected_labels.append(INTERICTAL.label)
                expected_lead.append(np.nan)

        # Assert labels match exactly the expectation
        self.assertEqual(expected_labels, res['type'].tolist())

        # Assert lead flags: True for labeled intervals, NaN for interictal
        for i, exp in enumerate(expected_lead):
            if pd.isna(exp):
                self.assertTrue(pd.isna(res.loc[i, 'lead_szr']))
            else:
                self.assertTrue(res.loc[i, 'lead_szr'])

    def test_make_segs_for_ptnt_produces_expected_columns_and_types(self):
        # Minimal working inputs to call make_segs_table
        first_recording_start = pd.Timestamp('2020-01-01 00:00:00')
        timespan = pd.Timedelta(minutes=2)

        # Make a small edf table and valid intervals that include some segments
        edfs = pd.DataFrame([{'file_name': 'f1', 'start_mtz': first_recording_start,
                              'end_mtz': first_recording_start + pd.Timedelta(minutes=5)}])
        valid_edf_intervals = pd.DataFrame([{'file_name': 'f1', 'start_mtz': first_recording_start,
                                             'end_mtz': first_recording_start + 3 * SEGMENT.exact_dur}])

        # Single seizure somewhere inside
        valid_szrs = pd.DataFrame([{'start_mtz': first_recording_start + pd.Timedelta(seconds=60), 'lead': False}])

        segs = create_segments.make_segs_for_ptnt(first_recording_start, timespan, valid_edf_intervals, edfs, valid_szrs)

        # Basic sanity checks
        self.assertIn('start_mtz', segs.columns)
        self.assertIn('type', segs.columns)
        self.assertIn('exists', segs.columns)
        self.assertGreater(len(segs), 0)

    def test_plot_segs_runs_without_errors_show_false(self):
        # Use a small segs and szrs table and ensure plotting with show=False does not raise
        starts = [pd.Timestamp('2020-01-01 00:00:00') + i * SEGMENT.exact_dur for i in range(5)]
        segs = pd.DataFrame({'start_mtz': starts, 'end_mtz': [s + SEGMENT.exact_dur for s in starts],
                             'type': [INTERICTAL.label] * 5, 'lead_szr': [None] * 5, 'exists': [True] * 5})
        szrs = pd.DataFrame([{'start_mtz': pd.Timestamp('2020-01-01 00:01:00'), 'lead': False}])

        # Should not raise
        create_segments.plot_segs(segs, szrs, edfs=None, title='test', show=False)


if __name__ == '__main__':
    unittest.main()
