import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from preprocessing import ACQUISITION_PRICES, FEATURES, TRIM_FIELDS


def cars(quotes, makes, start):
    df = pd.DataFrame("X", columns=FEATURES + TRIM_FIELDS, index=range(start, start + len(makes)))
    df["Make"] = makes
    df["VehicleAge"] = 3
    df["VehOdo"] = 50000
    df[ACQUISITION_PRICES] = pd.DataFrame(quotes, index=df.index, dtype=float).to_numpy()
    df["IsBadBuy"] = 0
    return df.rename_axis("RefId")


class QuoteTreatmentTests(unittest.TestCase):
    def test_default_filters_before_fitting_statistics_and_keeps_scoring_ids(self):
        from preprocessing import prepare_feature_pair

        history = cars([[40]*4, [80]*4, [0, 1, 0, 0], [0, 1, 0, 50]], ["A"]*4, 20)
        later = cars([[0]*4, [0, 1, 0, 0]], ["A"]*2, 30)
        x, v = prepare_feature_pair(history, later)
        self.assertEqual(x.index.tolist(), [20, 21, 23])
        self.assertEqual(v.index.tolist(), [30, 31])
        self.assertEqual(v.loc[30, ACQUISITION_PRICES[1]], 40)

    def test_notebook_loader_aligns_labels_after_training_row_exclusion(self):
        from preprocessing import load_development

        with TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows = cars([[0, 1, 0, 0], [25]*4, [0, 1, 0, 0]], ["A"]*3, 40)
            rows["PurchDate"] = ["01/01/2010", "01/02/2010", "06/10/2010"]
            rows["IsBadBuy"] = [0, 1, 0]
            rows.to_csv(p / "training.csv")
            for name, ids in [("train", [40, 41]), ("validation", [42])]:
                pd.DataFrame({"RefId": ids}).to_csv(p / f"{name}.csv", index=False)
            x, y, v, observed = load_development(p, raw=p)
            self.assertEqual(x.index.tolist(), [41])
            self.assertTrue(y.index.equals(x.index))
            self.assertEqual(y.tolist(), [1])
            self.assertEqual(v.index.tolist(), [42])
            self.assertTrue(observed.index.equals(v.index))
            x11, y11, v11, _ = load_development(p, raw=p, include_prices=False)
            self.assertTrue(y11.equals(y))
            self.assertTrue(x11.index.equals(x.index))
            self.assertEqual(len(x11.columns), 11)
            self.assertTrue(v11.index.equals(v.index))

    def test_same_cars_scored_with_training_only_quote_replacements(self):
        from quote_treatments import prepare_fold

        sentinel = [0, 1, 0, 0]
        train = cars([[100]*4, [200]*4, [600]*4, [700]*4, [1000]*4,
                      [1, 10000, 0, 10], sentinel], ["A","A","B","C","D","A","A"], 0)
        later = cars([sentinel, sentinel, [99999]*4, [0, 1, 0, 50]], ["A","B","A","A"], 10)
        original, untouched = prepare_fold(train, later, "original")
        x, v = prepare_fold(train, later, "peer_median", min_peers=2)
        self.assertEqual(x.loc[6, ACQUISITION_PRICES].tolist(), [150]*4)
        self.assertEqual(v.loc[10, ACQUISITION_PRICES].tolist(), [150]*4)
        self.assertEqual(v.loc[11, ACQUISITION_PRICES].tolist(), [600]*4)
        pd.testing.assert_frame_equal(x.drop(6), original.drop(6))
        pd.testing.assert_frame_equal(v.loc[[12,13]], untouched.loc[[12,13]])
        dropped, scored = prepare_fold(train, later, "drop_training_rows")
        self.assertEqual(dropped.index.tolist(), list(range(6)))
        self.assertTrue(scored.index.equals(later.index))
        self.assertEqual(original.loc[6, ACQUISITION_PRICES[1]], 1)

    def test_train_only_loader_skips_other_outcomes_and_dates(self):
        from preprocessing import read_training_rows

        with TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows = cars([[15]*4, [20]*4], ["A","B"], 10)
            rows["PurchDate"] = ["01/01/2010", "UNREADABLE"]
            rows.to_csv(p / "training.csv")
            pd.DataFrame({"RefId":[10]}).to_csv(p / "train.csv", index=False)
            self.assertEqual(read_training_rows(p, raw=p).index.tolist(), [10])
