import unittest

import pandas as pd

from preprocessing import ACQUISITION_PRICES, FEATURES, NUMERIC, TRIM_FIELDS, prepare_feature_pair
from tune_trees import choose_finalist, shortlist


class TuningTests(unittest.TestCase):
    def test_fold_statistics_come_only_from_earlier_cars(self):
        history = pd.DataFrame("X", columns=FEATURES + TRIM_FIELDS, index=range(20))
        history["Model"] = "COMMON"
        history[NUMERIC] = 5
        history[ACQUISITION_PRICES] = 100
        later = history.iloc[:2].copy()
        later["Model"] = "NEW"
        later.iloc[0, later.columns.get_loc(ACQUISITION_PRICES[0])] = 0
        later.iloc[1, later.columns.get_loc(ACQUISITION_PRICES[0])] = 90000
        later.iloc[0, later.columns.get_loc("VehicleAge")] = float("nan")
        a, b = prepare_feature_pair(history, later)
        self.assertEqual(b["Model"].tolist(), ["__OTHER__", "__OTHER__"])
        self.assertEqual(b[ACQUISITION_PRICES[0]].tolist(), [100, 90000])
        self.assertEqual(b.iloc[0]["VehicleAge"], 5)
        self.assertEqual(a.iloc[0]["Model"], "COMMON")

    def test_two_screening_leaders_plus_baseline(self):
        rows = pd.DataFrame({"depth": [8, 14, 20, 8], "regularization": [12, 12, 12, 30],
                             "cv_auc": [.68, .60, .62, .67], "cv_sd": [.01] * 4})
        got = shortlist(rows, "random_forest")
        self.assertEqual(set(zip(got.depth, got.regularization)), {(8, 12), (8, 30), (14, 12)})

    def test_near_ties_prefer_simpler_configuration(self):
        rows = pd.DataFrame({"depth": [8, 14, 8], "regularization": [12, 12, 30],
                             "validation_auc": [.67000, .67008, .6690], "cv_sd": [.01] * 3})
        self.assertEqual(choose_finalist(rows).depth, 8)
        self.assertEqual(choose_finalist(rows).regularization, 12)


if __name__ == "__main__":
    unittest.main()
