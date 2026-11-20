"""Check the tuned winner's calibration and select a validation-only bidding cutoff."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from business_loss import threshold_sensitivity
from evaluation import _aligned, read_validation_scores
from preprocessing import TARGET, prepare_feature_pair, read_development_rows
from tree_models import FAMILIES, make_tree, risk_scores
from tune_trees import FOLDS, OUT as TUNED, inputs

EPS = 1e-6  # finite log odds for regression scores at or beyond 0 and 1


def probabilities(scores, intercept=0.0, slope=1.0):
    raw = scores.clip(EPS, 1 - EPS)
    if intercept == 0 and slope == 1:
        return raw
    return pd.Series(expit(intercept + slope * logit(raw)), index=raw.index)


def calibration_metrics(labels, p):
    _aligned(labels, p)
    cut_points = np.unique(np.quantile(p, np.arange(1, 10) / 10))
    groups = np.searchsorted(cut_points, p, side="left")
    values = pd.DataFrame({"prediction": p, "outcome": labels})
    bins = values.groupby(groups).agg(n=("outcome", "size"), bad_buys=("outcome", "sum"),
                                      mean_prediction=("prediction", "mean"),
                                      observed_rate=("outcome", "mean"))
    bins.index = pd.RangeIndex(1, len(bins) + 1, name="risk_group")
    ece = np.average((bins.mean_prediction - bins.observed_rate).abs(), weights=bins.n)
    return {"n": len(p), "mean_prediction": p.mean(), "observed_rate": labels.mean(),
            "brier": brier_score_loss(labels, p), "log_loss": log_loss(labels, p),
            "ece": ece, "auc": roc_auc_score(labels, p)}, bins


def select_method(raw, adjusted, slope):
    if slope > 0 and adjusted["brier"] < raw["brier"] and adjusted["log_loss"] <= raw["log_loss"]:
        return "logistic"
    return "raw"


def history_predictions(train, name, depth, regularization):
    family = next(f for f, names in FAMILIES.items() if name in names)
    blocks = []
    for end, next_end in FOLDS:
        history = train.loc[train.PurchDate <= end]
        later = train.loc[(train.PurchDate > end) & (train.PurchDate <= next_end)]
        x, y, v = inputs(family, history, later)
        estimator = make_tree(name, depth, regularization)
        estimator.fit(x, y)
        blocks.append(later[[TARGET]].assign(score=risk_scores(estimator, v)))
    return pd.concat(blocks, verify_integrity=True)


def reliability_plot(bins, path):
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    fig = Figure(figsize=(6, 4.5), layout="constrained")
    ax = fig.subplots()
    limit = max(bins.mean_prediction.max(), bins.observed_rate.max()) * 1.12
    ax.plot([0, limit], [0, limit], "--", color="0.55", label="Perfect calibration")
    for method, color, marker in [("raw", "#276095", "o"), ("logistic", "#b44e27", "s")]:
        b = bins[bins.method == method]
        ax.plot(b.mean_prediction, b.observed_rate, marker=marker, markersize=4,
                linewidth=1.4, color=color, label=method.capitalize())
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Average predicted risk",
           ylabel="Observed bad-buy rate", title="Bad-buy risk by validation score group")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)


def main():
    directory = TUNED / "calibration"
    comparison = pd.read_csv(TUNED / "comparison.csv", index_col="model")
    name = comparison.tuned_validation_auc.idxmax()
    winner = comparison.loc[name]
    train, validation = read_development_rows()
    oof = history_predictions(train, name, int(winner.depth), int(winner.regularization))
    oof_logit = logit(probabilities(oof.score)).to_numpy().reshape(-1, 1)
    fit = LogisticRegression(C=1e6, tol=1e-10, max_iter=1000).fit(oof_logit, oof[TARGET])
    intercept, slope = float(fit.intercept_[0]), float(fit.coef_[0, 0])

    y = validation[TARGET]
    scores = read_validation_scores(TUNED, y)[name]
    if not np.isclose(roc_auc_score(y, scores), winner.tuned_validation_auc, rtol=0, atol=1e-12):
        raise ValueError("Saved validation scores do not match the tuned winner")
    _, prices = prepare_feature_pair(train, validation)
    results, groups, decisions = {}, [], []
    for method, a, b in [("raw", 0.0, 1.0), ("logistic", intercept, slope)]:
        p = probabilities(scores, a, b)
        metrics, bins = calibration_metrics(y, p)
        results[method] = dict(model=name, intercept=a, slope=b, eps=EPS, **metrics)
        groups.append(bins.reset_index().assign(method=method))
        decisions.append(threshold_sensitivity(y, p, prices).assign(model=name, method=method))
    selected = select_method(results["raw"], results["logistic"], slope)
    summary = pd.DataFrame.from_dict(results, orient="index").rename_axis("method")
    summary["selected"] = summary.index == selected
    summary["calibration_training_cars"] = [0, len(oof)]
    summary["validation_scores_clipped"] = int(((scores < EPS) | (scores > 1 - EPS)).sum())
    bins = pd.concat(groups, ignore_index=True)
    policies = pd.concat(decisions, ignore_index=True)
    policies["selected_method"] = policies.method == selected
    directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(directory / "metrics.csv")
    bins.to_csv(directory / "reliability.csv", index=False)
    policies.to_csv(directory / "thresholds.csv", index=False)
    reliability_plot(bins, directory / "reliability.png")
    print(summary.to_string())
    print(policies[policies.selected_method].to_string(index=False))


if __name__ == "__main__":
    main()
