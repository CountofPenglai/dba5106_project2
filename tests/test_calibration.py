import unittest

import numpy as np
import pandas as pd

from carvana.calibration import calibration_metrics, probabilities, select_method


class CalibrationTests(unittest.TestCase):
    def test_reliability_retains_ties_and_weights_bins_by_vehicle_count(self):
        y = pd.Series([0, 0, 1, 1])
        p = pd.Series([0.2, 0.2, 0.2, 0.8])
        metrics, bins = calibration_metrics(y, p)
        self.assertEqual(bins.n.tolist(), [3, 1])
        self.assertAlmostEqual(metrics["ece"], 0.15)
        self.assertAlmostEqual(metrics["brier"], 0.19)
        self.assertAlmostEqual(metrics["observed_rate"], 0.5)

    def test_identity_and_logistic_recalibration_use_the_same_probability_scale(self):
        raw = pd.Series([-0.01, 0.2, 0.5, 1.01], index=[8, 3, 1, 2])
        p = probabilities(raw)
        self.assertTrue(p.index.equals(raw.index))
        self.assertTrue((p > 0).all() and (p < 1).all())
        adjusted = probabilities(raw, np.log(2), 1)
        self.assertAlmostEqual(adjusted.iloc[1], 1/3)
        self.assertAlmostEqual(adjusted.iloc[2], 2/3)
        self.assertTrue(adjusted.is_monotonic_increasing)

    def test_keep_raw_if_log_loss_worsens_or_brier_ties(self):
        raw = {"brier": .10, "log_loss": .35}
        self.assertEqual(select_method(raw, {"brier": .099, "log_loss": .36}, 1), "raw")
        self.assertEqual(select_method(raw, raw, 1), "raw")
        self.assertEqual(select_method(raw, {"brier": .099, "log_loss": .34}, 1), "logistic")
