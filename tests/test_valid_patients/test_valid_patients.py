import unittest
import os
from pathlib import Path

import pandas as pd

from config.constants import MIN_VALID_SEIZURES_PER_PATIENT
from preprocessing.validate_patients import validate_patient


class TestValidPatients(unittest.TestCase):
    def test_validate_patient(self):
        szr_starts_path = Path(__file__).parent / 'data' / 'seizure_starts.csv'
        szrs_corr = pd.read_csv(szr_starts_path, parse_dates=['start'], index_col=0)

        valid_szrs_comp, szrs_comp, patient_info = validate_patient(szrs_corr)

        # load the expected results
        valid_szrs_corr = szrs_corr[szrs_corr['should_be_valid']]

        comparison = valid_szrs_comp['start'].values == valid_szrs_corr['start'].values
        self.assertTrue(comparison.all(),
                        'The computed valid seizure starts do not match the expected ones.\n'
                        f'valid_szrs_comp: {valid_szrs_comp["start"].values}\n'
                        f'valid_szrs_corr: {valid_szrs_corr["start"].values}')

        total_szrs = len(szrs_corr)
        valid_szrs = len(valid_szrs_corr)
        valid_ptnt = valid_szrs >= MIN_VALID_SEIZURES_PER_PATIENT
        self.assertEqual(patient_info['total_seizures'], total_szrs, 'The total number of seizures is incorrect.')
        self.assertEqual(patient_info['valid_seizures'], valid_szrs, 'The number of valid seizures is incorrect.')
        self.assertEqual(patient_info['valid'], valid_ptnt, 'The validity of the patient is incorrect.')
