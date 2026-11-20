import unittest

import pandas as pd

from high_risk_features import add_features, ranking_metrics


class HighRiskFeatureTests(unittest.TestCase):
    def test_peer_medians_use_only_training_cars_with_age_then_global_fallback(self):
        cols = ["VehicleAge", "Size", "VehOdo", "MMRAcquisitionAuctionAveragePrice",
                "MMRAcquisitionAuctionCleanPrice"]
        past = pd.DataFrame([[3, "A", 100, 8000, 9200], [3, "B", 300, 6000, 6600],
                             [3, "A", 140, 5000, 5000], [6, "B", 260, 5000, 4500]], columns=cols)
        later = pd.DataFrame([[3, "A", 240, 8000, 9200], [3, "C", 280, 8000, 8000],
                              [8, "C", 400, 8000, 7200]], columns=cols, index=[43, 61, 55])
        train, val = add_features(past, later)
        self.assertEqual(val.index.tolist(), [43, 61, 55])
        self.assertEqual(val["RelativeMileage"].tolist(), [2., 2., 2.])
        self.assertEqual(val["AuctionConditionSpread"].tolist(), [.15, 0., -.1])
        self.assertAlmostEqual(train.loc[0, "RelativeMileage"], 100 / 120)
        self.assertNotIn("RelativeMileage", past)

    def test_exact_top_lists_round_up_and_break_ties_by_vehicle_id(self):
        ids = pd.Index(range(103, 0, -1), name="RefId")
        scores = pd.Series(0.1, index=ids)
        scores.loc[[4, 2, 3, 1]] = 0.9
        labels = pd.Series(0, index=ids)
        labels.loc[[1, 3, 5, 103]] = 1
        result = ranking_metrics(labels, scores)
        self.assertEqual((result['top1_n'], result['top1_bad']), (2, 1))
        self.assertEqual((result['top5_n'], result['top5_bad']), (6, 3))
        self.assertEqual(result['top5_precision'], .5)
