import unittest

import pandas as pd

from business_loss import threshold_sensitivity


class BusinessLossTests(unittest.TestCase):
    def setUp(self):
        self.ids = pd.Index([18, 5, 20, 17], name="RefId")
        self.y = pd.Series([1, 0, 1, 0], index=self.ids)
        self.scores = pd.Series([0.8, 0.6, 0.6, 0.1], index=self.ids)
        self.prices = pd.DataFrame({
            "MMRAcquisitionAuctionAveragePrice": [3000, 4000, 1000, 5000],
            "MMRAcquisitionRetailAveragePrice": [4500, 5050, 1700, 5700],
        }, index=self.ids)

    def test_minimum_dollar_loss_moves_threshold_across_tied_scores(self):
        report = threshold_sensitivity(self.y, self.scores, self.prices)
        self.assertEqual(report["threshold"].tolist(), [0.8, 0.6, 0.6])
        self.assertEqual(report["bought_bad_FN"].tolist(), [1, 0, 0])
        self.assertEqual(report["skipped_good_FP"].tolist(), [0, 1, 1])
        self.assertEqual(report["loss_usd"].tolist(), [100, 150, 150])
        self.assertEqual(report["loss_per_car_usd"].tolist(), [25, 37.5, 37.5])
        self.assertEqual(report["buy_all_loss_usd"].tolist(), [400, 800, 1200])
        self.assertEqual(report["skip_all_loss_usd"].tolist(), [150, 150, 150])

    def test_all_buy_and_all_skip_are_available(self):
        scores = pd.Series([0.1, 0.4, 0.3, 0.9], index=self.ids)
        prices = self.prices.copy()
        prices.loc[17, "MMRAcquisitionRetailAveragePrice"] = 6400
        report = threshold_sensitivity(self.y, scores, prices)
        self.assertEqual(report["threshold"].tolist(), [float("inf"), 0.1, 0.1])

    def test_refuses_misaligned_quotes(self):
        with self.assertRaises(ValueError):
            threshold_sensitivity(self.y, self.scores, self.prices.iloc[::-1])


if __name__ == "__main__":
    unittest.main()
