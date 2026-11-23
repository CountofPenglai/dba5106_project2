"""Global and individual TreeSHAP explanations for the best validation CatBoost."""
import textwrap

import numpy as np
import pandas as pd
from catboost import Pool

from .calibration import frozen_policy, probabilities
from .evaluation import read_validation_scores, validation_auc
from .preprocessing import (ACQUISITION_PRICES, CATEGORICAL, NUMERIC, TRIM_FIELDS,
                            OUT as TUNED, TARGET, read_development_rows)
from .training import refit_score
from .registry import MODELS

OUT = TUNED / "shap"
LABELS = dict(VehicleAge="Vehicle age", VehOdo="Odometer", VNST="Purchase state",
              Auction="Auction provider", IsOnlineSale="Online sale", SubModel="Submodel")
LABELS.update(dict(zip(ACQUISITION_PRICES, ["Auction quote, average", "Auction quote, clean",
                                            "Retail quote, average", "Retail quote, clean"])))
CASES = {"highest_risk": "Highest validation risk score", "marginal_skip": "Lowest score among skipped cars",
         "median_missed_bad": "Median-scored bad buy among bids"}


def global_importance(shap):
    mean_abs = shap.abs().mean().sort_values(ascending=False).rename("mean_abs_shap")
    return mean_abs.to_frame().assign(share_of_total=mean_abs / mean_abs.sum()).rename_axis("feature")


def select_cases(scores, y, threshold):
    scores = scores.sort_index()  # deterministic tie-break by RefId
    skipped = scores[scores >= threshold]
    missed = scores[(scores < threshold) & y.eq(1)]
    cases = {}
    if not skipped.empty:
        cases["marginal_skip"] = skipped.idxmin()
    if not missed.empty:
        cases["median_missed_bad"] = (missed - missed.median()).abs().idxmin()
    return cases


def plot_importance(importance):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 5.3), layout="constrained")
    ax = fig.subplots()
    top = importance.head(10).iloc[::-1]
    bars = ax.barh([LABELS.get(x, x) for x in top.index], top.mean_abs_shap, color="#256b75")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.set(xlim=(0, top.mean_abs_shap.max() * 1.20), xlabel="Mean absolute contribution to raw risk score",
           title="What moved the CatBoost score across validation cars?")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "global_importance.png", dpi=180)


def plot_cases(local):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8.2, 7), layout="constrained")
    axes = fig.subplots(local.case.nunique(), 1, sharex=True, squeeze=False).ravel()
    extent = 0
    for ax, (case, rows) in zip(axes, local.groupby("case", sort=False)):
        r = rows.iloc[0]
        top = rows.iloc[:4]
        effects = [*top.shap, rows.iloc[4:].shap.sum()]
        names = [textwrap.fill(f"{LABELS.get(f, f)} = {v}", width=32) for f, v in zip(top.feature, top.value)]
        names.append("Other 11 fields combined")
        ax.barh(range(len(names)), effects, color=["#ac472f" if e >= 0 else "#256b75" for e in effects])
        ax.set_yticks(range(len(names)), names, fontsize=9)
        ax.invert_yaxis()
        for i, change in enumerate(effects):
            ax.annotate(f"{change:+.4f}", (change, i), xytext=(4 if change >= 0 else -4, 0),
                        textcoords="offset points", ha="left" if change >= 0 else "right",
                        va="center", fontsize=8)
        extent = max(extent, max(abs(v) for v in effects))
        ax.axvline(0, color="0.4", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{CASES[case]}  ·  #{r.RefId}", loc="left", fontsize=12, weight="bold", pad=18)
        ax.text(0, 1.017, f"Base {r.base_value:.4f}   →   raw {r.raw_score:.4f}   |   risk {r.decision_score:.1%}   |   "
                f"{r.decision.upper()}   |   cutoff {r.threshold:.4f}   |   recorded {'bad buy' if r.IsBadBuy else 'good buy'}",
                transform=ax.transAxes, fontsize=9)
    axes[-1].set(xlim=(-1.28 * extent, 1.28 * extent), xlabel="Signed contribution to raw risk score")
    fig.savefig(OUT / "local_explanations.png", dpi=180)


def explain():
    train, val = read_development_rows()
    observed = val[TARGET]
    saved = read_validation_scores(TUNED, observed)
    tree_scores = saved[list(MODELS["catboost"])]
    name = validation_auc(observed, tree_scores).index[0]
    model, raw_scores, v = refit_score(name, train, val)
    np.testing.assert_allclose(raw_scores, saved[name], atol=1e-12, rtol=0)
    calibration, rule = frozen_policy()
    scores = probabilities(raw_scores, calibration.intercept, calibration.slope)
    threshold = rule.threshold
    if rule.model == name:
        if (scores >= threshold).sum() != rule.skipped_bids:
            raise ValueError("Refit failed to reproduce the selected bid rule")
        examples = select_cases(scores, observed, threshold)
    else:
        # Still explain a tree for the assignment if the overall winner is a different family.
        scores = probabilities(raw_scores)
        threshold = np.nan
        examples = {}
    if not examples:
        examples = {"highest_risk": scores.idxmax()}
    pool = Pool(v, cat_features=CATEGORICAL + TRIM_FIELDS)
    values = model.get_feature_importance(pool, type="ShapValues", shap_calc_type="Regular", thread_count=2)
    shap = pd.DataFrame(values[:, :-1], columns=v.columns, index=v.index)
    base = pd.Series(values[:, -1], index=v.index)
    margin = pd.Series(model.predict(v, prediction_type="RawFormulaVal"), index=v.index)
    np.testing.assert_allclose(base + shap.sum(axis=1), margin, rtol=0, atol=1e-9)
    if name.endswith("Logloss"):
        from scipy.special import expit
        np.testing.assert_allclose(expit(margin), raw_scores, rtol=0, atol=1e-12)
    importance = global_importance(shap)
    numeric = []
    for field in NUMERIC + ACQUISITION_PRICES:
        low, high = v[field].quantile([.25, .75])
        lower, upper = v[field] <= low, v[field] >= high
        numeric.append(dict(feature=field, low_cut=low, high_cut=high, low_n=lower.sum(), high_n=upper.sum(),
                            low_mean_shap=shap.loc[lower, field].mean(), high_mean_shap=shap.loc[upper, field].mean()))

    cases = []
    for case, ref in examples.items():
        impacts = shap.loc[ref].sort_values(key=abs, ascending=False)
        cases.append(pd.DataFrame(dict(case=case, RefId=ref, IsBadBuy=observed[ref],
                                       model=name, decision=("unassigned" if pd.isna(threshold) else
                                                             "skip" if scores[ref] >= threshold else "bid"),
                                       threshold=threshold, decision_score=scores[ref], raw_score=margin[ref], base_value=base[ref],
                                       value=v.loc[ref], shap=impacts)).reindex(impacts.index).rename_axis("feature"))
    local = pd.concat(cases).reset_index()
    OUT.mkdir(exist_ok=True, parents=True)
    importance.to_csv(OUT / "global_importance.csv")
    pd.DataFrame(numeric).to_csv(OUT / "numeric_contrasts.csv", index=False)
    local.to_csv(OUT / "local_explanations.csv", index=False)
    plot_importance(importance)
    plot_cases(local)
    return importance
