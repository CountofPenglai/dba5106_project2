import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from carvana.preprocessing import ACQUISITION_PRICES, FEATURES


def cars(quotes, makes, start):
    df = pd.DataFrame("X", columns=FEATURES, index=range(start, start + len(makes)))
    df["Make"] = makes
    df["VehicleAge"] = 3
    df["VehOdo"] = 50000
    df[ACQUISITION_PRICES] = pd.DataFrame(quotes, index=df.index, dtype=float).to_numpy()
    df["IsBadBuy"] = 0
    return df.rename_axis("RefId")


class QuoteTreatmentTests(unittest.TestCase):
    def test_default_filters_before_fitting_statistics_and_keeps_scoring_ids(self):
        from carvana.preprocessing import prepare_feature_pair

        history = cars([[40]*4, [80]*4, [0, 1, 0, 0], [0, 1, 0, 50]], ["A"]*4, 20)
        later = cars([[0]*4, [0, 1, 0, 0]], ["A"]*2, 30)
        x, v = prepare_feature_pair(history, later)
        self.assertEqual(x.index.tolist(), [20, 21, 23])
        self.assertEqual(v.index.tolist(), [30, 31])
        self.assertEqual(v.loc[30, ACQUISITION_PRICES[1]], 40)

    def test_validation_loader_leaves_holdout_outcomes_unread(self):
        from carvana.preprocessing import TARGET, prepare_feature_pair, read_development_rows, read_holdout_rows

        with TemporaryDirectory() as tmp:
            raw = Path(tmp)
            rows = cars([[10]*4, [0, 1, 0, 0], [25]*4, [0]*4], ["A"]*4, 40)
            rows["PurchDate"] = ["09/30/2010", "06/07/2010", "06/08/2010", "09/29/2010"]
            rows[TARGET] = ["private", 0, 1, 0]
            rows.to_csv(raw / "training.csv")
            train, val = read_development_rows(raw)
            x, v = prepare_feature_pair(train, val)
            self.assertEqual(x.columns.tolist(), FEATURES)
            self.assertEqual(train.loc[x.index, TARGET].tolist(), [1])
            self.assertEqual(x.index.tolist(), [42])
            self.assertEqual(v.index.tolist(), [43])
            self.assertEqual(v.loc[43, ACQUISITION_PRICES[0]], 25)
            with self.assertRaisesRegex(ValueError, "labelled auctions"):
                read_holdout_rows(raw)
