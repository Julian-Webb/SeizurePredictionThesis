import unittest
from pathlib import Path
from pprint import pprint
from typing import List, Tuple

from config.paths import Paths, Dataset, PatientDir


class TestPaths(unittest.TestCase):
    def setUp(self):
        base_dir = Path(__file__).parent / "data"
        self.PATHS = Paths(base_dir)


    def _adjust_path_lists(self, correct: List[str], computed: List[Path]) -> Tuple[List[Path], List[Path]]:
        # prepend the base path
        correct = [PatientDir(self.PATHS.base_dir / path) for path in correct]
        correct.sort()
        computed.sort()
        return correct, computed

    @staticmethod
    def _print_results(correct, computed):
        print('correct:')
        pprint(correct)
        print('computed:')
        pprint(computed)
        return correct, computed


    def test_patient_dirs_all(self):
        correct = [
            '20240201_UNEEG_ForMayo/mayo_p1',
            '20240201_UNEEG_ForMayo/mayo_p2',
            '20250217_UNEEG_Extended/ext_p1',
            '20250501_SUBQ_SeizurePredictionCompetition_2025final/comp_p1',
            'invalid_patients/20250217_UNEEG_Extended/ext_p2'
        ]
        computed = self.PATHS.patient_dirs(include_invalid_ptnts=True)
        correct, computed = self._adjust_path_lists(correct, computed)

        # self._print_results(correct, computed)
        self.assertListEqual(correct, computed,
                         f"When requesting all patient dirs, the result isn't correct.\n{correct=}\n{computed=}")

    def test_patient_dirs_valid(self):
        correct = [
            '20240201_UNEEG_ForMayo/mayo_p1',
            '20240201_UNEEG_ForMayo/mayo_p2',
            '20250217_UNEEG_Extended/ext_p1',
            '20250501_SUBQ_SeizurePredictionCompetition_2025final/comp_p1',
        ]
        computed = self.PATHS.patient_dirs(include_invalid_ptnts=False)
        correct, computed = self._adjust_path_lists(correct, computed)

        # self._print_results(correct, computed)
        self.assertListEqual(correct, computed,
                         f"When requesting invalid patient dirs, the result isn't correct.\n{correct=}\n{ computed=}")


    def test_patient_dirs_single_dataset(self):
        correct = [
            '20250217_UNEEG_Extended/ext_p1',
            'invalid_patients/20250217_UNEEG_Extended/ext_p2'
        ]
        computed = self.PATHS.patient_dirs([Dataset.uneeg_extended], include_invalid_ptnts=True)
        correct, computed = self._adjust_path_lists(correct, computed)

        # self._print_results(correct, computed)
        self.assertListEqual(correct, computed,
                         f"When patient dirs for just uneeg_extended, the result isn't correct.\n{correct=}\n{ computed=}")



    def test_patient_dirs_single_dataset_valid(self):
        correct = [
            '20250217_UNEEG_Extended/ext_p1',
        ]
        computed = self.PATHS.patient_dirs([Dataset.uneeg_extended], include_invalid_ptnts=False)
        correct, computed = self._adjust_path_lists(correct, computed)

        # self._print_results(correct, computed)
        self.assertListEqual(correct, computed,
                         f"When just valid patient dirs for just uneeg_extended, the result isn't correct.\n{correct=}\n{ computed=}")


