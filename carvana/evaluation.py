"""Validation ranking and cost-based decisions for the Carvana project."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .registry import MODELS


def risk_scores(estimator, features):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    return estimator.predict(features)


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


def save_validation(y, scores, directory, family):
    report = validation_auc(y, scores)
    directory.mkdir(parents=True, exist_ok=True)
    pd.concat([y.rename("IsBadBuy"), scores], axis=1).to_csv(directory / f"{family}_scores.csv")
    report.to_csv(directory / f"{family}_auc.csv", index_label="model")
    return report


def read_validation_scores(directory, labels):
    """Read the unrounded scores and verify each file's validation IDs and outcomes."""
    frames = []
    for family, variants in MODELS.items():
        frame = pd.read_csv(directory / f"{family}_scores.csv", index_col="RefId",
                            float_precision="round_trip")
        if not frame.pop("IsBadBuy").equals(labels):
            raise ValueError(f"{family} results do not match the current validation set")
        if tuple(frame.columns) != variants:
            raise ValueError(f"Expected {variants} scores in {family} results")
        frames.append(frame)
    return pd.concat(frames, axis=1, verify_integrity=True)
