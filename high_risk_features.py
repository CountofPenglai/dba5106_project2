"""Fixed-parameter CatBoost L2 feature comparison on chronological development data."""
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluation import _aligned
from preprocessing import PRICE_RESULTS, TARGET, prepare_feature_pair, read_development_rows
from tree_models import make_tree, risk_scores
from tune_trees import FOLDS

MILEAGE = "RelativeMileage"
SPREAD = "AuctionConditionSpread"
EXTRAS = {"baseline": [], "relative_mileage": [MILEAGE], "auction_spread": [SPREAD],
          "both": [MILEAGE, SPREAD]}
OUT = PRICE_RESULTS / "high_risk_features"


def add_features(history, later):
    """Append the two proposed features to an already preprocessed pair of splits."""
    keys = ["VehicleAge", "Size"]
    by_peer = history.groupby(keys)["VehOdo"].median()
    by_age = history.groupby("VehicleAge")["VehOdo"].median()
    overall = history["VehOdo"].median()
    if pd.isna(overall) or overall <= 0:
        raise ValueError("A positive training median is required for relative mileage")
    def augment(frame):
        peer = by_peer.reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy()
        typical = pd.Series(peer, index=frame.index)
        typical = typical.where(typical > 0, frame["VehicleAge"].map(by_age))
        typical = typical.where(typical > 0, overall)
        average = frame["MMRAcquisitionAuctionAveragePrice"]
        return frame.assign(RelativeMileage=frame["VehOdo"] / typical,
                            AuctionConditionSpread=(frame["MMRAcquisitionAuctionCleanPrice"] - average) / average)

    return augment(history), augment(later)


def ranking_metrics(labels, scores):
    """Precision in exact-size review lists; break score ties by ascending RefId."""
    _aligned(labels, scores)
    ranked = scores.sort_index().sort_values(ascending=False, kind="stable")
    result = {"auc": roc_auc_score(labels, scores)}
    for percent in (1, 5):
        n = (len(ranked) * percent + 99) // 100
        hits = int(labels.loc[ranked.index[:n]].sum())
        result.update({f"top{percent}_n": n, f"top{percent}_bad": hits,
                       f"top{percent}_precision": hits / n})
    return result


def evaluate_split(history, later, split):
    x, v = prepare_feature_pair(history, later)
    original = x.columns.tolist()
    x, v = add_features(x, v)
    scores = pd.DataFrame(index=later.index)
    rows = []
    for variant, extra in EXTRAS.items():
        features = original + extra
        model = make_tree("CatBoost_L2", depth=8, regularization=3)
        model.fit(x[features], history.loc[x.index, TARGET])
        scores[variant] = risk_scores(model, v[features])
        rows.append(dict(split=split, features=variant, feature_n=len(features),
                         train_n=len(x), eval_n=len(v), train_end=history.PurchDate.max().date(),
                         eval_start=later.PurchDate.min().date(), eval_end=later.PurchDate.max().date(),
                         **ranking_metrics(later[TARGET], scores[variant])))
        print(f"{split}: {variant} complete", flush=True)
    return pd.DataFrame(rows), scores


def main():
    train, val = read_development_rows()
    folds = []
    for i, (end, next_end) in enumerate(FOLDS, 1):
        history = train.loc[train.PurchDate <= end]
        later = train.loc[(train.PurchDate > end) & (train.PurchDate <= next_end)]
        result, _ = evaluate_split(history, later, f"fold_{i}")
        folds.append(result)
    folds = pd.concat(folds, ignore_index=True)
    final, scores = evaluate_split(train, val, "validation")
    means = folds.groupby("features")[["auc", "top1_precision", "top5_precision"]].mean()
    report = final.drop(columns=["split"]).join(means.add_prefix("cv_mean_"), on="features")
    OUT.mkdir(exist_ok=True, parents=True)
    folds.to_csv(OUT / "folds.csv", index=False)
    report.to_csv(OUT / "comparison.csv", index=False)
    pd.concat([val[TARGET], scores], axis=1).to_csv(OUT / "validation_scores.csv")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
