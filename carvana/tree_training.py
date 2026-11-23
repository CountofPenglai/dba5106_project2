"""Screen tree configurations on expanding time folds and refit validation finalists."""
from itertools import product

import pandas as pd
from sklearn.metrics import roc_auc_score

from .evaluation import risk_scores, save_validation
from .preprocessing import OUT, TARGET, TRAIN_END, prepare_feature_pair
from .registry import MODELS
from .tree_models import BASELINE, encode_inputs, make_tree

FOLDS = (("2009-11-30", "2010-01-31"), ("2010-01-31", "2010-03-31"),
         ("2010-03-31", TRAIN_END.strftime("%Y-%m-%d")))
GRIDS = {"random_forest": ((8, 14, 20), (12, 30)), "catboost": ((4, 6, 8), (3, 10))}
KEYS = ["model", "depth", "regularization"]
AUC_TIE = 0.0001


def inputs(family, train, val):
    history, later = prepare_feature_pair(train, val)
    labels = train.loc[history.index, TARGET]
    history, later = encode_inputs(family, history, later)
    return history, labels, later


def shortlist(rows, family):
    ranked = rows.sort_values(["cv_auc", "cv_sd", "depth", "regularization"],
                              ascending=[False, True, True, False])
    depth, reg = BASELINE[family]
    baseline = ranked[(ranked.depth == depth) & (ranked.regularization == reg)]
    return pd.concat([ranked.head(2), baseline]).drop_duplicates(["depth", "regularization"])


def choose_finalist(rows):
    near_best = rows[rows.validation_auc >= rows.validation_auc.max() - AUC_TIE]
    return near_best.sort_values(["depth", "cv_sd", "regularization"],
                                 ascending=[True, True, False]).iloc[0]


def chronological_folds(train):
    for cutoff, end in FOLDS:
        earlier = train.loc[train.PurchDate <= cutoff]
        later = train.loc[(train.PurchDate > cutoff) & (train.PurchDate <= end)]
        yield cutoff, end, earlier, later


def screen(family, train):
    results = []
    for fold, (cutoff, end, earlier, later) in enumerate(chronological_folds(train), 1):
        x, y, v = inputs(family, earlier, later)
        for name in MODELS[family]:
            for depth, reg in product(*GRIDS[family]):
                model = make_tree(name, depth, reg)
                model.fit(x, y)
                auc = roc_auc_score(later[TARGET], risk_scores(model, v))
                results.append(dict(model=name, depth=depth, regularization=reg, fold=fold,
                                    train_end=cutoff, fold_end=end, train_n=len(y),
                                    fold_n=len(later), fold_auc=auc))
            print(f"Screened {name}, fold {fold}/3", flush=True)
    folds = pd.DataFrame(results)
    folds.to_csv(OUT / f"{family}_folds.csv", index=False)
    return folds.groupby(KEYS, as_index=False).agg(cv_auc=("fold_auc", "mean"), cv_sd=("fold_auc", "std"))


def tune_family(family, train, val):
    cv = screen(family, train)
    x, y, v = inputs(family, train, val)
    selected_scores = pd.DataFrame(index=val.index)
    finalists = []
    for name in MODELS[family]:
        candidates = shortlist(cv[cv.model == name], family).copy()
        predictions = {}
        for i, row in candidates.iterrows():
            depth, reg = int(row.depth), int(row.regularization)
            model = make_tree(name, depth, reg)
            model.fit(x, y)
            predictions[i] = risk_scores(model, v)
            candidates.loc[i, "train_auc"] = roc_auc_score(y, risk_scores(model, x))
            candidates.loc[i, "validation_auc"] = roc_auc_score(val[TARGET], predictions[i])
        winner = choose_finalist(candidates)
        selected_scores[name] = predictions[winner.name]
        candidates["selected"] = candidates.index == winner.name
        depth, reg = BASELINE[family]
        candidates["baseline"] = (candidates.depth == depth) & (candidates.regularization == reg)
        finalists.append(candidates)
        print(f"{name} selected depth={winner.depth}, regularization={winner.regularization}", flush=True)
    pd.concat(finalists).to_csv(OUT / f"{family}_finalists.csv", index=False)
    return save_validation(val[TARGET], selected_scores, OUT, family)

def save_tree_comparison(ranking):
    tables = [pd.read_csv(OUT / f"{family}_finalists.csv") for family in GRIDS]
    rows = pd.concat(tables)
    baseline = rows[rows.baseline].set_index("model").validation_auc
    summary = rows[rows.selected].set_index("model").drop(columns=["selected", "baseline"])
    summary = summary.rename(columns={"validation_auc": "tuned_validation_auc"})
    summary.insert(0, "baseline_validation_auc", baseline)
    summary["delta_auc"] = summary.tuned_validation_auc - baseline
    summary = summary.loc[ranking.index.intersection(summary.index, sort=False)]
    if not summary.tuned_validation_auc.sub(ranking.loc[summary.index]).abs().le(1e-12).all():
        raise ValueError("Finalists do not match saved validation scores")
    summary.to_csv(OUT / "comparison.csv", index_label="model")
