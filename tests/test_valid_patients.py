import unittest
import os
from pathlib import Path

import pandas as pd

from config.constants import Constants
from preprocessing.validate_patients import _validate_patient
from config.paths import PatientDir


class TestValidPatients(unittest.TestCase):
    def test_validate_patient(self):
        patient_dir = PatientDir(Path(os.path.dirname(__file__)) / 'test_files' / 'valid_patients' / 'example_patient')
        valid_szrs_computed, _, patient_info = _validate_patient(patient_dir)

        # load the expected results
        szrs = pd.read_csv(patient_dir.all_szr_starts_file, parse_dates=['start'], index_col=0)
        valid_szrs_expected = szrs[szrs['should_be_valid']]

        comparison = valid_szrs_computed['start'].values == valid_szrs_expected['start'].values
        self.assertTrue(comparison.all(),
                         'The computed valid seizure starts do not match the expected ones.\n'
                         f'valid_szrs_computed: {valid_szrs_computed["start"].values}\n'
                         f'valid_szrs_expected: {valid_szrs_expected["start"].values}')

        total_szrs = len(szrs)
        valid_szrs = len(valid_szrs_expected)
        valid_ptnt = valid_szrs >= Constants.MIN_VALID_SEIZURES_PER_PATIENT
        self.assertEqual(patient_info['total_seizures'], total_szrs, 'The total number of seizures is incorrect.')
        self.assertEqual(patient_info['valid_seizures'], valid_szrs, 'The number of valid seizures is incorrect.')
        self.assertEqual(patient_info['valid'], valid_ptnt, 'The validity of the patient is incorrect.')
