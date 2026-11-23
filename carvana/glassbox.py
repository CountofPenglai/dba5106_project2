"""Train and explain the two sparse linear Glassbox models."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .evaluation import risk_scores, save_validation
from .preprocessing import (ACQUISITION_PRICES, CATEGORICAL, NUMERIC, OUT, TARGET,
                            TRIM_FIELDS, missing_quote_bundle, prepare_feature_pair)

NUMERIC_FIELDS = NUMERIC + ACQUISITION_PRICES
CATEGORICAL_FIELDS = [*CATEGORICAL, *TRIM_FIELDS]
ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
C_VALUES = (0.01, 0.1, 1.0, 10.0)
RANDOM_STATE = 42


def temporal_inner_split(dates, train_fraction=0.8):
    """Return a complete-date 80/20 split, chosen closest to the requested row fraction."""
    if not dates.index.is_unique or dates.isna().any() or not 0 < train_fraction < 1:
        raise ValueError("Expected unique RefIds, complete dates and a fraction between zero and one")
    counts = dates.value_counts().sort_index()
    if len(counts) < 2:
        raise ValueError("At least two purchase dates are required for temporal tuning")
    cumulative = counts.cumsum().iloc[:-1]
    target = train_fraction * len(dates)
    cutoff = (cumulative - target).abs().idxmin()
    inner_train = dates.index[dates <= cutoff]
    inner_tuning = dates.index[dates > cutoff]
    return inner_train, inner_tuning, pd.Timestamp(cutoff)


def _preprocessor():
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FIELDS),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=True,
                ),
                CATEGORICAL_FIELDS,
            ),
        ]
    )


def _lasso(alpha):
    return Pipeline(
        [
            ("preprocess", _preprocessor()),
            (
                "model",
                Lasso(alpha=alpha, max_iter=20_000, selection="cyclic", tol=1e-5),
            ),
        ]
    )


def _logistic(c_value):
    return Pipeline(
        [
            ("preprocess", _preprocessor()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    penalty="l1",
                    solver="saga",
                    max_iter=5_000,
                    tol=1e-4,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _tune(X_train, y_train, X_tuning, y_tuning, values, builder, parameter, prefer):
    rows = []
    for value in values:
        pipeline = builder(value)
        pipeline.fit(X_train, y_train)
        scores = risk_scores(pipeline, X_tuning)
        rows.append({"parameter": parameter, "value": value,
                     "inner_tuning_auc": roc_auc_score(y_tuning, scores)})
    table = pd.DataFrame(rows)
    ascending = prefer == "smaller"
    best = table.sort_values(
        ["inner_tuning_auc", "value"], ascending=[False, ascending], kind="stable"
    ).iloc[0]
    table["selected"] = table["value"].eq(best["value"])
    return float(best["value"]), table


def coefficient_table(models):
    """Align coefficients and attach the scale needed for honest interpretation."""
    l2 = models["Glassbox_L2"]
    logloss = models["Glassbox_Logloss"]
    l2_preprocessor = l2.named_steps["preprocess"]
    logloss_preprocessor = logloss.named_steps["preprocess"]
    l2_encoder = l2_preprocessor.named_transformers_["categorical"]
    logloss_encoder = logloss_preprocessor.named_transformers_["categorical"]
    l2_names = [*NUMERIC_FIELDS, *l2_encoder.get_feature_names_out(CATEGORICAL_FIELDS)]
    logloss_names = [*NUMERIC_FIELDS, *logloss_encoder.get_feature_names_out(CATEGORICAL_FIELDS)]
    if l2_names != logloss_names:
        raise ValueError("The two final Glassbox models produced different encoded features")

    references = {}
    for field, categories, dropped in zip(
        CATEGORICAL_FIELDS, l2_encoder.categories_, l2_encoder.drop_idx_
    ):
        references[field] = "" if dropped is None else str(categories[dropped])

    numeric_scale = dict(
        zip(NUMERIC_FIELDS, l2_preprocessor.named_transformers_["numeric"].scale_)
    )
    rows = []
    for name in l2_names:
        if name in NUMERIC_FIELDS:
            field, level, kind = name, "", "numeric"
        else:
            matches = [field for field in CATEGORICAL_FIELDS if name.startswith(f"{field}_")]
            if len(matches) != 1:
                raise ValueError(f"Cannot map encoded feature to a raw field: {name}")
            field = matches[0]
            level = name[len(field) + 1:]
            kind = "categorical"
        rows.append(
            {
                "encoded_feature": name,
                "original_field": field,
                "feature_level": level,
                "feature_kind": kind,
                "reference_level": references.get(field, ""),
                "training_scale": numeric_scale.get(field, np.nan),
            }
        )
    table = pd.DataFrame(rows)
    table["Glassbox_L2_coefficient"] = l2.named_steps["model"].coef_
    table["Glassbox_Logloss_coefficient"] = logloss.named_steps["model"].coef_[0]
    table["Glassbox_Logloss_odds_ratio"] = np.exp(
        np.clip(table["Glassbox_Logloss_coefficient"], -700, 700)
    )
    table["Glassbox_L2_nonzero"] = table["Glassbox_L2_coefficient"].ne(0)
    table["Glassbox_Logloss_nonzero"] = table["Glassbox_Logloss_coefficient"].ne(0)
    table["direction_stable"] = (
        table["Glassbox_L2_nonzero"]
        & table["Glassbox_Logloss_nonzero"]
        & np.sign(table["Glassbox_L2_coefficient"]).eq(
            np.sign(table["Glassbox_Logloss_coefficient"])
        )
    )
    return table


SPECIFICATIONS = {
    "Glassbox_L2": (ALPHAS, _lasso, "alpha", "larger"),
    "Glassbox_Logloss": (C_VALUES, _logistic, "C", "smaller"),
}


def tune_glassbox(train, validation):
    eligible = train.loc[~missing_quote_bundle(train)]
    inner, tuning, end = temporal_inner_split(eligible.PurchDate)
    x, v = prepare_feature_pair(eligible.loc[inner], eligible.loc[tuning])
    full, later = prepare_feature_pair(eligible, validation)
    models, searches = {}, []
    scores = pd.DataFrame(index=later.index)
    for name, (grid, build, parameter, tie) in SPECIFICATIONS.items():
        best, table = _tune(x, eligible.loc[x.index, TARGET], v, eligible.loc[v.index, TARGET],
                            grid, build, parameter, tie)
        models[name] = build(best).fit(full, eligible.loc[full.index, TARGET])
        scores[name] = risk_scores(models[name], later)
        searches.append(table.assign(model=name, inner_end=end, train_n=len(x), tuning_n=len(v)))
    coefficient_table(models).to_csv(OUT / "glassbox_coefficients.csv", index=False)
    pd.concat(searches).to_csv(OUT / "glassbox_tuning.csv", index=False)
    return save_validation(validation[TARGET], scores, OUT, "glassbox")
