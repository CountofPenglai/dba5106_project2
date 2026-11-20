import unittest

import pandas as pd

from tree_shap import global_importance, select_cases


class TreeShapTests(unittest.TestCase):
    def test_global_impact_uses_magnitude_before_averaging(self):
        shap = pd.DataFrame({"age": [.2, -.2], "miles": [.1, .1]})
        importance = global_importance(shap)
        self.assertAlmostEqual(importance.loc["age", "mean_abs_shap"], .2)
        self.assertAlmostEqual(importance.loc["age", "share_of_total"], 2/3)

    def test_skip_boundary_and_middle_missed_case(self):
        ids = pd.Index([5, 2, 6, 7, 1, 4], name="RefId")
        scores = pd.Series([.50, .50, .2, .4, .3, .31], index=ids)
        y = pd.Series([1, 1, 1, 1, 1, 0], index=ids)
        cases = select_cases(scores, y, .50)
        self.assertEqual(cases, {"marginal_skip": 2, "median_missed_bad": 1})
