"""Global and individual TreeSHAP explanations for the selected 15-field CatBoost L2."""
import textwrap

import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.metrics import roc_auc_score

from preprocessing import ACQUISITION_PRICES, CATEGORICAL, NUMERIC, TRIM_FIELDS, load_development
from tree_models import make_tree, risk_scores
from tune_trees import OUT as TUNED

MODEL = "CatBoost_L2"
OUT = TUNED / "shap"
LABELS = dict(VehicleAge="Vehicle age", VehOdo="Odometer", VNST="Purchase state",
              Auction="Auction provider", IsOnlineSale="Online sale", SubModel="Submodel")
LABELS.update(dict(zip(ACQUISITION_PRICES, ["Auction quote, average", "Auction quote, clean",
                                            "Retail quote, average", "Retail quote, clean"])))
CASES = {"marginal_skip": "Lowest score among skipped cars",
         "median_missed_bad": "Median-scored bad buy among bids"}


def global_importance(shap):
    mean_abs = shap.abs().mean().sort_values(ascending=False).rename("mean_abs_shap")
    return mean_abs.to_frame().assign(share_of_total=mean_abs / mean_abs.sum()).rename_axis("feature")


def select_cases(scores, y, threshold):
    scores = scores.sort_index()  # deterministic tie-break by RefId
    skipped = scores[scores >= threshold]
    missed = scores[(scores < threshold) & y.eq(1)]
    return {"marginal_skip": skipped.idxmin(),
            "median_missed_bad": (missed - missed.median()).abs().idxmin()}


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
    axes = fig.subplots(2, 1, sharex=True)
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
        ax.text(0, 1.017, f"Base {r.base_value:.4f}   →   score {r.raw_score:.4f}   |   "
                f"{r.decision.upper()}   |   cutoff {r.threshold:.4f}   |   recorded {'bad buy' if r.IsBadBuy else 'good buy'}",
                transform=ax.transAxes, fontsize=9)
    axes[-1].set(xlim=(-1.28 * extent, 1.28 * extent), xlabel="Signed contribution to raw risk score")
    fig.savefig(OUT / "local_explanations.png", dpi=180)


def main():
    x, y, v, observed = load_development()
    fit = pd.read_csv(TUNED / "comparison.csv", index_col="model").loc[MODEL]
    rules = pd.read_csv(TUNED / "calibration" / "thresholds.csv", dtype={"threshold": str})
    rule = rules.loc[rules.model.eq(MODEL) & rules.selected_method & rules.bad_buy_loss_rate.eq(.2)].squeeze()
    if rule.method != "raw" or rule.normal_cost_usd != 900 or len(v.columns) != 15:
        raise ValueError("Expected the selected 15-field raw-score policy at 20% and $900")
    threshold = float(rule.threshold)
    model = make_tree(MODEL, int(fit.depth), int(fit.regularization))
    model.fit(x, y)
    scores = pd.Series(risk_scores(model, v), index=v.index)
    if not np.isclose(roc_auc_score(observed, scores), fit.tuned_validation_auc, rtol=0, atol=1e-12):
        raise ValueError("Refit failed to reproduce the validation AUC")
    saved_path = TUNED / "catboost_scores.csv"
    if saved_path.exists():
        saved = pd.read_csv(saved_path, index_col="RefId", dtype={MODEL: str})
        if not saved.IsBadBuy.equals(observed):
            raise ValueError("Saved score IDs or labels differ from the validation split")
        np.testing.assert_allclose(scores, saved[MODEL].astype(float), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(scores >= threshold, saved[MODEL].astype(float) >= threshold)
    if (scores >= threshold).sum() != rule.skipped_bids:
        raise ValueError("Refit failed to reproduce the selected bid rule")

    pool = Pool(v, cat_features=CATEGORICAL + TRIM_FIELDS)
    values = model.get_feature_importance(pool, type="ShapValues", shap_calc_type="Regular", thread_count=2)
    shap = pd.DataFrame(values[:, :-1], columns=v.columns, index=v.index)
    base = pd.Series(values[:, -1], index=v.index)
    np.testing.assert_allclose(base + shap.sum(axis=1), scores, rtol=0, atol=1e-9)
    importance = global_importance(shap)
    numeric = []
    for field in NUMERIC + ACQUISITION_PRICES:
        low, high = v[field].quantile([.25, .75])
        lower, upper = v[field] <= low, v[field] >= high
        numeric.append(dict(feature=field, low_cut=low, high_cut=high, low_n=lower.sum(), high_n=upper.sum(),
                            low_mean_shap=shap.loc[lower, field].mean(), high_mean_shap=shap.loc[upper, field].mean()))

    cases = []
    for case, ref in select_cases(scores, observed, threshold).items():
        impacts = shap.loc[ref].sort_values(key=abs, ascending=False)
        cases.append(pd.DataFrame(dict(case=case, RefId=ref, IsBadBuy=observed[ref],
                                       decision="skip" if scores[ref] >= threshold else "bid",
                                       threshold=threshold, raw_score=scores[ref], base_value=base[ref],
                                       value=v.loc[ref], shap=impacts)).reindex(impacts.index).rename_axis("feature"))
    local = pd.concat(cases).reset_index()
    OUT.mkdir(exist_ok=True, parents=True)
    importance.to_csv(OUT / "global_importance.csv")
    pd.DataFrame(numeric).to_csv(OUT / "numeric_contrasts.csv", index=False)
    local.to_csv(OUT / "local_explanations.csv", index=False)
    plot_importance(importance)
    plot_cases(local)
    print(importance.head(8).to_string())
    print(local.groupby("case", sort=False).head(4).to_string(index=False))
    print(f"Reconstructed all {len(scores):,} scores; largest absolute error: "
          f"{(base + shap.sum(axis=1) - scores).abs().max():.2g}")


if __name__ == "__main__":
    main()
