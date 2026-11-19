"""Validation ranking and cost-based decisions for the Carvana project."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def _aligned(y, scores):
    if not y.index.is_unique or not scores.index.equals(y.index):
        raise ValueError("Labels and scores must share the same unique RefIds in order")
    if set(y.unique()) != {0, 1} or not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Expected both binary labels and finite scores")


def validation_auc(y, scores):
    """One row per model. Scores are continuous, with larger values meaning riskier."""
    _aligned(y, scores)
    return pd.Series({name: roc_auc_score(y, scores[name]) for name in scores},
                     name="validation_auc").sort_values(ascending=False)


def threshold_costs(y, scores, *, missed_bad=5.0, skipped_good=1.0):
    """For one selected model, skip if score >= threshold. First row = skip none."""
    _aligned(y, scores)
    if min(missed_bad, skipped_good) <= 0 or not np.isfinite([missed_bad, skipped_good]).all():
        raise ValueError("Decision costs must be finite and positive")
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    fp = np.rint(fpr * (y == 0).sum()).astype(int)
    fn = np.rint((1 - tpr) * (y == 1).sum()).astype(int)
    return pd.DataFrame({"threshold": thresholds, "skipped_good_FP": fp, "missed_bad_FN": fn,
                         "cost_per_vehicle": (skipped_good * fp + missed_bad * fn) / len(y)})


def save_validation(y, scores, directory, family):
    report = validation_auc(y, scores)
    directory.mkdir(parents=True, exist_ok=True)
    pd.concat([y.rename("IsBadBuy"), scores], axis=1).to_csv(directory / f"{family}_scores.csv")
    report.to_csv(directory / f"{family}_auc.csv", index_label="model")
    return report


if __name__ == "__main__":
    from preprocessing import PROCESSED, ROOT

    directory = ROOT / "results" / "validation"
    labels = pd.read_csv(PROCESSED / "validation.csv", usecols=["RefId", "IsBadBuy"],
                         index_col="RefId")["IsBadBuy"]
    frames = []
    for family in ("random_forest", "catboost"):
        frame = pd.read_csv(directory / f"{family}_scores.csv", index_col="RefId")
        if not frame.pop("IsBadBuy").equals(labels):
            raise ValueError(f"{family} results do not match the current validation set")
        frames.append(frame)
    comparison = validation_auc(labels, pd.concat(frames, axis=1))
    comparison.to_csv(directory / "comparison.csv", index_label="model")
    print(comparison.to_string())
