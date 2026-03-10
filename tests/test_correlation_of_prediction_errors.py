import unittest
import numpy as np

from model_comparison.correlation_of_prediction_errors import correlation_of_prediction_errors


class TestCorrelationOfPredictionErrors(unittest.TestCase):
    def test_identical_models_zero_correlation_when_errors_have_variance(self):
        # Two identical predictors should have perfect correlation of their errors (1.0)
        L = np.array([0, 1, 0, 1, 0, 1], dtype=float)
        pi = np.array([0.1, 0.9, 0.2, 0.7, 0.3, 0.6], dtype=float)
        pj = pi.copy()
        corr = correlation_of_prediction_errors(pi, pj, L)
        self.assertAlmostEqual(corr, 1.0, places=6)

    def test_perfectly_anti_correlated_errors(self):
        # Construct predictions so that |pi - L| is linearly related to (1 - |pj - L|)
        # Example: let L=0 for all, ei = x, ej = 1 - x -> centered errors perfectly anticorrelated
        L = np.zeros(6, dtype=float)
        ei = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], dtype=float)
        ej = 1.0 - ei
        pi = ei  # since L=0, |pi-L| = pi
        pj = ej
        corr = correlation_of_prediction_errors(pi, pj, L)
        self.assertAlmostEqual(corr, -1.0, places=6)

    def test_weighting_emphasizes_larger_errors(self):
        # Design data where large-error samples align across models; correlation should be strongly positive
        L = np.array([0, 0, 0, 0, 0], dtype=float)
        pi = np.array([0.0, 0.9, 0.0, 0.8, 0.0], dtype=float)  # big errors at idx 1,3
        pj = np.array([0.0, 0.85, 0.0, 0.75, 0.0], dtype=float)  # similar big errors at idx 1,3
        corr = correlation_of_prediction_errors(pi, pj, L)
        # Expect high correlation close to 1 due to aligned big-error points and weighting by max(ei, ej)
        self.assertGreater(corr, 0.95)

    def test_invariant_to_label_encoding_symmetry_between_models(self):
        # Symmetry: swapping i and j should yield identical correlation
        L = np.array([0, 1, 1, 0, 1], dtype=float)
        pi = np.array([0.2, 0.7, 0.6, 0.3, 0.8], dtype=float)
        pj = np.array([0.1, 0.6, 0.55, 0.25, 0.75], dtype=float)
        c1 = correlation_of_prediction_errors(pi, pj, L)
        c2 = correlation_of_prediction_errors(pj, pi, L)
        self.assertAlmostEqual(c1, c2, places=12)

    def test_handles_zero_variance_errors(self):
        # If one model has zero-variance errors, correlation is defined as 0.0 by implementation
        L = np.zeros(5, dtype=float)
        pi = np.zeros(5, dtype=float)  # perfect predictions -> ei all zeros -> std 0
        pj = np.array([0.0, 0.2, 0.4, 0.2, 0.0], dtype=float)
        corr = correlation_of_prediction_errors(pi, pj, L)
        self.assertEqual(corr, 0.0)


if __name__ == '__main__':
    unittest.main()
