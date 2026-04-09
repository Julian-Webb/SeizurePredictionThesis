import unittest
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config.constants import MIN_VALID_SEIZURES_PER_PATIENT
from preprocessing.validate_patients import ptnt_valid_szrs


class TestValidPatients(unittest.TestCase):
    def test_validate_patient(self):
        szrs_corr = pd.DataFrame(
            {
                "start_mtz": pd.to_datetime([
                    "2000-01-01 00:00:00.000",
                    "2000-01-01 01:05:00.000",
                    "2000-01-01 03:00:00.000",
                    "2000-01-01 03:30:00.000",
                    "2000-01-01 04:35:00.000",
                    "2000-01-01 04:36:00.000",
                    "2000-01-01 05:39:00.000",
                ]),
                "should_be_valid": [True, True, True, False, True, False, False],
            },
        )

        valid_szrs_comp, szrs_comp, patient_info = ptnt_valid_szrs(szrs_corr)

        # correct results
        valid_szrs_corr = szrs_corr[szrs_corr['should_be_valid']].reset_index(drop=True)

        pd.testing.assert_series_equal(valid_szrs_comp['start_mtz'], valid_szrs_corr['start_mtz'], )

        total_szrs = len(szrs_corr)
        valid_szrs = len(valid_szrs_corr)
        enough_valid_szrs = valid_szrs >= MIN_VALID_SEIZURES_PER_PATIENT
        self.assertEqual(patient_info['total_seizures'], total_szrs, 'The total number of seizures is incorrect.')
        self.assertEqual(patient_info['valid_seizures'], valid_szrs, 'The number of valid seizures is incorrect.')
        self.assertEqual(patient_info['enough_valid_seizures'], enough_valid_szrs,
                         'The validity of the patient is incorrect.')
