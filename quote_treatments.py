"""Compare three quote treatments on the existing training-period time folds."""
import pandas as pd
from sklearn.metrics import roc_auc_score

from preprocessing import ACQUISITION_PRICES, ROOT, TARGET, missing_quote_bundle, prepare_feature_pair, read_training_rows
from tree_models import make_tree, risk_scores
from tune_trees import FOLDS

METHODS = ("original", "peer_median", "drop_training_rows")
OUT = ROOT / "results" / "training_cv" / "quote_treatments"


def peer_quotes(history, targets, min_peers):
    prices = ACQUISITION_PRICES
    complete = history[prices].gt(0).fillna(False).all(axis=1)
    donors = history.loc[complete]
    if donors.empty and not targets.empty:
        raise ValueError("Peer imputation requires training cars with complete quotes")
    replacements = pd.DataFrame(index=targets.index, columns=prices, dtype=float)
    for keys in (["Make", "VehicleAge", "Size"], ["VehicleAge", "Size"]):
        groups = donors.groupby(keys)
        medians = groups[prices].median().loc[groups.size() >= min_peers]
        matched = targets[keys].join(medians, on=keys)
        replacements = replacements.fillna(matched[prices])
    return replacements.fillna(donors[prices].median())


def prepare_fold(history, later, method, *, min_peers=20):
    if method not in METHODS:
        raise ValueError(f"Unknown quote treatment: {method}")
    x, v = prepare_feature_pair(history, later, drop_invalid_quotes=(method == "drop_training_rows"))
    if method == "peer_median":
        keys = ["Make", "VehicleAge", "Size"]
        flagged = pd.concat([rows.loc[missing_quote_bundle(rows), keys] for rows in (history, later)],
                            verify_integrity=True)
        replacements = peer_quotes(history, flagged, min_peers)
        # Keep the original transformations identical on every ordinary row.
        x.update(replacements)
        v.update(replacements)
    return x, v


def main():
    train = read_training_rows()
    results = []
    for fold, (cutoff, end) in enumerate(FOLDS, 1):
        history = train.loc[train.PurchDate <= cutoff]
        later = train.loc[(train.PurchDate > cutoff) & (train.PurchDate <= end)]
        for method in METHODS:
            x, v = prepare_fold(history, later, method)
            model = make_tree("CatBoost_L2", depth=8, regularization=3)
            model.fit(x, history.loc[x.index, TARGET])
            auc = roc_auc_score(later.loc[v.index, TARGET], risk_scores(model, v))
            results.append(dict(fold=fold, method=method, train_end=cutoff, fold_end=end,
                                train_n=len(x), eval_n=len(v), flagged_train=missing_quote_bundle(history).sum(),
                                flagged_eval=missing_quote_bundle(later).sum(), auc=auc))
            print(f"Fold {fold} | {method} | {auc:.6f}", flush=True)
    folds = pd.DataFrame(results)
    table = folds.pivot(index="method", columns="fold", values="auc").reindex(METHODS)
    table.columns = [f"fold_{i}_auc" for i in table.columns]
    table["mean_auc"] = table.mean(axis=1)
    table["change_vs_original"] = table.mean_auc - table.loc["original", "mean_auc"]
    OUT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUT / "folds.csv", index=False)
    table.to_csv(OUT / "comparison.csv")
    print(table.to_string())


if __name__ == "__main__":
    main()
