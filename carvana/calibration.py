"""Check the tuned winner's calibration and select a validation-only bidding cutoff."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .business_loss import threshold_sensitivity
from .evaluation import _aligned, read_validation_scores, validation_auc
from .preprocessing import OUT, TARGET, prepare_feature_pair, read_development_rows
from .training import refit_score
from .tree_training import chronological_folds, save_tree_comparison

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


def history_predictions(train, name):
    blocks = []
    for _, _, history, later in chronological_folds(train):
        _, scores, _ = refit_score(name, history, later)
        blocks.append(later[[TARGET]].assign(score=scores))
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


def validate():
    train, validation = read_development_rows()
    y = validation[TARGET]
    all_scores = read_validation_scores(OUT, y)
    ranking = validation_auc(y, all_scores)
    ranking.to_csv(OUT / "six_model_auc.csv", index_label="model")
    save_tree_comparison(ranking)
    name = ranking.index[0]
    scores = all_scores[name]
    directory = OUT / "calibration"
    oof = history_predictions(train, name)
    oof_logit = logit(probabilities(oof.score)).to_numpy().reshape(-1, 1)
    fit = LogisticRegression(C=1e6, tol=1e-10, max_iter=1000).fit(oof_logit, oof[TARGET])
    intercept, slope = float(fit.intercept_[0]), float(fit.coef_[0, 0])

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
    chosen = policies.loc[policies.selected_method].drop(columns=["method", "selected_method"]).copy()
    chosen.insert(0, "validation_auc", ranking[name])
    chosen.to_csv(OUT / "business_loss.csv", index=False)
    reliability_plot(bins, directory / "reliability.png")
    return ranking


def frozen_policy():
    """Load the selected validation rule at full precision, including threshold ties."""
    directory = OUT / "calibration"
    metrics = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    thresholds = pd.read_csv(directory / "thresholds.csv", float_precision="round_trip")
    calibration, = metrics.loc[metrics.selected].itertuples()
    rule, = thresholds.loc[thresholds.selected_method & thresholds.bad_buy_loss_rate.eq(.2)
                           & thresholds.normal_cost_usd.eq(900)].itertuples()
    if (rule.model, rule.method) != (calibration.model, calibration.method):
        raise ValueError("Threshold and calibration artifacts belong to different policies")
    return calibration, rule
