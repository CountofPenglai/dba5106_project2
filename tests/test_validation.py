import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd

from carvana.business_loss import AUCTION, RETAIL, threshold_sensitivity
from carvana.evaluation import read_validation_scores, save_validation, validation_auc
from carvana.glassbox import temporal_inner_split
from carvana.holdout import score_rule
from carvana.registry import MODELS


class ValidationTests(unittest.TestCase):
    def test_all_six_models_compete_on_identical_vehicles(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            y = pd.Series([0, 1, 0, 1], index=pd.Index([7, 10, 2, 12], name="RefId"), name="IsBadBuy")
            for family, names in MODELS.items():
                scores = pd.DataFrame(.4, columns=names, index=y.index)
                if family == "glassbox":
                    scores[names[0]] = y * .9
                save_validation(y, scores, out, family)
            ranking = validation_auc(y, read_validation_scores(out, y))
            self.assertEqual(len(ranking), 6)
            self.assertEqual(ranking.index[0], "Glassbox_L2")
            p = out / "catboost_scores.csv"
            pd.read_csv(p).iloc[::-1].to_csv(p, index=False)
            with self.assertRaisesRegex(ValueError, "validation set"):
                read_validation_scores(out, y)

    def test_inner_glassbox_split_keeps_whole_dates(self):
        dates = pd.Series(pd.to_datetime(["2010-01-01"]*4 + ["2010-01-02"]*3))
        history, later, cutoff = temporal_inner_split(dates)
        self.assertTrue(dates.loc[history].le(cutoff).all())
        self.assertTrue(dates.loc[later].gt(cutoff).all())

    def test_holdout_applies_frozen_threshold_including_ties(self):
        rows = pd.DataFrame({"IsBadBuy": [1, 0, 1], "PurchDate": pd.to_datetime(["2010-10-01"]*3)})
        prices = pd.DataFrame({AUCTION: [3000]*3, RETAIL: [4000]*3})
        scores = pd.Series([.7, .7, .3])
        policy = SimpleNamespace(threshold=.7, bad_buy_loss_rate=.2, normal_cost_usd=900)
        report = score_rule(rows, prices, scores, policy)
        self.assertEqual(report["skipped_bids"], 2)
        self.assertEqual(report["fixed_rule_loss_usd"], 700)
        best = threshold_sensitivity(rows.IsBadBuy, scores, prices, loss_rates=(.2,))
        self.assertLess(best.loss_usd.item(), report["fixed_rule_loss_usd"])
