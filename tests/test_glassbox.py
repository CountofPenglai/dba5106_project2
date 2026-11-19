import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import roc_auc_score

from evaluation import compare_families
from glassbox import CATEGORICAL_FIELDS, NUMERIC_FIELDS, run_glassbox, temporal_inner_split
from preprocessing import ACQUISITION_PRICES, prepare_data


def _auctions():
    groups = [
        (pd.Timestamp("2010-01-01"), 160),
        (pd.Timestamp("2010-06-09"), 60),
        (pd.Timestamp("2010-10-01"), 40),
    ]
    dates = []
    for start, size in groups:
        dates.extend(start + pd.to_timedelta(np.arange(size) // 4, unit="D"))
    n = len(dates)
    i = np.arange(n)
    frame = pd.DataFrame(
        {
            "RefId": 10_000 + i,
            "PurchDate": pd.DatetimeIndex(dates).strftime("%m/%d/%Y"),
            "VehicleAge": 1 + i % 9,
            "VehOdo": 35_000 + (i * 937) % 90_000,
            "Make": np.where(i % 3, "Ford", "Chevrolet"),
            "Model": np.where(i % 17, "Focus", "Rare model"),
            "Size": np.where(i % 2, "Medium", "Large"),
            "Transmission": np.where(i % 5, "Auto", "Manual"),
            "Auction": np.where(i % 2, "ADESA", "MANHEIM"),
            "VNST": np.where(i % 3, "TX", "CA"),
            "IsOnlineSale": (i % 4 == 0).astype(int),
            "Trim": np.where(i % 4, "SE", "LX"),
            "SubModel": np.where(i % 3, "4D SEDAN", "SUV"),
            "IsBadBuy": ((i % 9 >= 7) | ((i * 7) % 19 < 2)).astype(int),
        }
    )
    for offset, field in enumerate(ACQUISITION_PRICES):
        frame[field] = 3_000 + offset * 500 + (i * (31 + offset)) % 7_000
    test = frame.drop(columns="IsBadBuy").iloc[:32].copy()
    test["RefId"] += 1_000_000
    return frame, test


class GlassboxTests(unittest.TestCase):
    def test_temporal_split_keeps_dates_whole(self):
        dates = pd.Series(
            pd.to_datetime(["2010-01-01"] * 3 + ["2010-01-02"] * 4 + ["2010-01-03"] * 3),
            index=pd.Index(range(10), name="RefId"),
        )
        inner, tuning, cutoff = temporal_inner_split(dates)
        self.assertEqual(cutoff, pd.Timestamp("2010-01-02"))
        self.assertTrue(set(dates.loc[inner]).isdisjoint(set(dates.loc[tuning])))
        self.assertEqual(len(inner), 7)

    def test_end_to_end_outputs_are_aligned_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            processed = root / "processed"
            results = root / "results"
            raw.mkdir()
            training, test = _auctions()
            training.to_csv(raw / "training.csv", index=False)
            test.to_csv(raw / "test.csv", index=False)
            prepare_data(raw=raw, output=processed)

            first = run_glassbox(processed=processed, raw=raw, directory=results)
            second = run_glassbox(processed=processed, raw=raw, directory=root / "repeat")
            assert_frame_equal(first["scores"], second["scores"], check_exact=True)

            labels = pd.read_csv(
                processed / "validation.csv", usecols=["RefId", "IsBadBuy"],
                index_col="RefId",
            )["IsBadBuy"]
            saved = pd.read_csv(results / "glassbox_scores.csv", index_col="RefId")
            self.assertTrue(saved.pop("IsBadBuy").equals(labels))
            self.assertTrue(np.isfinite(saved.to_numpy()).all())
            self.assertEqual(list(saved), ["Glassbox_L2", "Glassbox_Logloss"])
            for model in saved:
                self.assertAlmostEqual(
                    roc_auc_score(labels, saved[model]), first["auc"].loc[model], places=14
                )
            coefficients = pd.read_csv(results / "glassbox_coefficients.csv")
            self.assertTrue(set(NUMERIC_FIELDS).issubset(set(coefficients.original_field)))
            self.assertTrue(set(CATEGORICAL_FIELDS).issubset(set(coefficients.original_field)))
            self.assertTrue(
                coefficients[["Glassbox_L2_nonzero", "Glassbox_Logloss_nonzero"]]
                .to_numpy().any()
            )
            self.assertTrue((results / "glassbox_top_effects.png").stat().st_size > 0)

    def test_six_model_family_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            labels = pd.Series([0, 1, 0, 1], index=pd.Index([1, 2, 3, 4], name="RefId"))
            families = {
                "glassbox": ["Glassbox_L2", "Glassbox_Logloss"],
                "random_forest": ["RF_L2", "RF_Logloss"],
                "catboost": ["CatBoost_L2", "CatBoost_Logloss"],
            }
            for offset, (family, models) in enumerate(families.items()):
                frame = labels.rename("IsBadBuy").to_frame()
                for model_offset, model in enumerate(models):
                    frame[model] = [0.1, 0.8 - offset * 0.01, 0.2, 0.7 - model_offset * 0.01]
                frame.to_csv(directory / f"{family}_scores.csv")
            report = compare_families(directory, labels)
            self.assertEqual(len(report), 6)
            self.assertEqual(set(report.index), {model for models in families.values() for model in models})

    def test_notebook_is_clean_and_code_cells_compile(self):
        notebook = json.loads(Path("glassbox_linear.ipynb").read_text())
        for cell in notebook["cells"]:
            self.assertFalse(cell.get("outputs"))
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "glassbox_linear.ipynb", "exec")


if __name__ == "__main__":
    unittest.main()
