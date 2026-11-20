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


def read_validation_scores(directory, labels):
    """Read the unrounded scores and verify each file's validation IDs and outcomes."""
    frames = []
    for family in ("random_forest", "catboost"):
        frame = pd.read_csv(directory / f"{family}_scores.csv", index_col="RefId",
                            float_precision="round_trip")
        if not frame.pop("IsBadBuy").equals(labels):
            raise ValueError(f"{family} results do not match the current validation set")
        frames.append(frame)
    return pd.concat(frames, axis=1, verify_integrity=True)


def compare_families(directory, labels):
    return validation_auc(labels, read_validation_scores(directory, labels))


if __name__ == "__main__":
    import argparse
    from preprocessing import PRICE_RESULTS, PROCESSED, VALIDATION_RESULTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-prices", action="store_true",
                        help="compare baseline and acquisition-price AUCs on the same validation cars")
    args = parser.parse_args()

    directory = VALIDATION_RESULTS
    labels = pd.read_csv(PROCESSED / "validation.csv", usecols=["RefId", "IsBadBuy"],
                         index_col="RefId")["IsBadBuy"]
    price_dir = PRICE_RESULTS
    comparison = compare_families(price_dir, labels)
    output = price_dir / "auc.csv"
    if args.compare_prices:
        baseline = compare_families(directory, labels)
        comparison = pd.concat([baseline.rename("baseline_auc"), comparison.rename("with_prices_auc")], axis=1)
        output = price_dir / "comparison.csv"
        comparison["delta_auc"] = comparison.with_prices_auc - comparison.baseline_auc
        comparison = comparison.sort_values("with_prices_auc", ascending=False)
    comparison.to_csv(output, index_label="model")
    print(comparison.to_string())
