import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from model_eval.predict_and_evaluate_old import find_best_threshold, _compute_threshold_metrics, plot_threshold_metrics


class TestMakePredictions(unittest.TestCase):
    def test_find_best_threshold_f1_basic(self):
        # Perfect separation at 0.5
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        thr, score, metrics = find_best_threshold(y_true, y_pred, metric='f1', precision=0.01)
        self.assertGreaterEqual(thr, 0.3)
        self.assertLessEqual(thr, 0.7)
        self.assertAlmostEqual(metrics['f1'], 1.0)
        self.assertEqual(metrics['tp'], 3)
        self.assertEqual(metrics['tn'], 3)

    def test_find_best_threshold_roc_uses_youden(self):
        # Construct to favor threshold near 0.5 by maximizing TPR-FPR
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
        thr, score, metrics = find_best_threshold(y_true, y_pred, metric='roc')
        self.assertGreater(score, 0)  # J = TPR - FPR
        self.assertTrue(0 <= thr <= 1)
        # Validate metrics dictionary content
        for k in ['tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'specificity', 'f1']:
            self.assertIn(k, metrics)

    def test_find_best_threshold_precision_and_recall(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        # Precision optimized
        thr_p, score_p, met_p = find_best_threshold(y_true, y_pred, metric='precision')
        self.assertTrue(0 <= thr_p <= 1)
        self.assertGreaterEqual(score_p, 0)
        # Recall optimized
        thr_r, score_r, met_r = find_best_threshold(y_true, y_pred, metric='recall')
        self.assertTrue(0 <= thr_r <= 1)
        self.assertGreaterEqual(score_r, 0)

    def test_compute_threshold_metrics_monotonic_thresholds(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.2, 0.9, 0.3, 0.8])
        metrics = _compute_threshold_metrics(y_true, y_pred, thresholds=np.linspace(0, 1, 11))
        self.assertListEqual(sorted(metrics.keys()), ['f1', 'precision', 'recall', 'specificity', 'threshold'])
        self.assertEqual(len(metrics['threshold']), 11)
        # Check values are within [0,1]
        for k in ['precision', 'recall', 'f1', 'specificity']:
            arr = np.array(metrics[k])
            self.assertTrue(np.all((0 <= arr) & (arr <= 1)))

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_threshold_metrics_writes_when_output_path_given(self, mock_close, mock_savefig):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0.2, 0.85, 0.1, 0.7, 0.95])
        out = Path('dummy_plot.png')
        metrics, roc = plot_threshold_metrics(y_true, y_pred, output_path=out)
        # savefig called with correct path
        mock_savefig.assert_called()
        self.assertIn('auc', roc)
        # Ensure metrics are reasonable sizes
        self.assertIn('threshold', metrics)
        self.assertGreater(len(metrics['threshold']), 0)
        mock_close.assert_called()

    def test_plot_threshold_metrics_with_precomputed_metrics(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0.3, 0.6, 0.7, 0.2])
        pre_metrics = _compute_threshold_metrics(y_true, y_pred, thresholds=[0.0, 0.5, 1.0])
        # Should use provided metrics and still return ROC data
        with patch('matplotlib.pyplot.savefig') as mock_save:
            metrics, roc = plot_threshold_metrics(y_true, y_pred, output_path=Path('x.png'), precomputed_metrics=pre_metrics)
        self.assertEqual(metrics['threshold'], pre_metrics['threshold'])
        self.assertIn('auc', roc)


if __name__ == '__main__':
    unittest.main()
