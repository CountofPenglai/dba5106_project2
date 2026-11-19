"""Train and explain the two sparse linear Glassbox models."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluation import save_validation
from preprocessing import (
    ACQUISITION_PRICES,
    CATEGORICAL,
    PROCESSED,
    RAW,
    ROOT,
    TRIM_FIELDS,
    load_development,
)

NUMERIC_FIELDS = ["VehicleAge", "VehOdo", *ACQUISITION_PRICES]
CATEGORICAL_FIELDS = [*CATEGORICAL, *TRIM_FIELDS]
ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
C_VALUES = (0.01, 0.1, 1.0, 10.0)
RANDOM_STATE = 42


def load_training_dates(output=PROCESSED):
    """Read the purchase dates that define the already-prepared training split."""
    dates = pd.read_csv(
        Path(output) / "train.csv",
        usecols=["RefId", "PurchDate"],
        index_col="RefId",
        parse_dates=["PurchDate"],
    )["PurchDate"]
    if not dates.index.is_unique or dates.isna().any():
        raise ValueError("Training RefIds and purchase dates must be complete and unique")
    return dates


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


def _validate_inputs(X, y, dates):
    expected = NUMERIC_FIELDS + CATEGORICAL_FIELDS
    missing = [field for field in expected if field not in X]
    if missing:
        raise ValueError(f"Missing Glassbox fields: {missing}")
    if not X.index.equals(y.index) or not X.index.equals(dates.index) or not X.index.is_unique:
        raise ValueError("Features, labels and dates must share the same unique RefIds in order")
    if set(y.unique()) != {0, 1}:
        raise ValueError("Training labels must contain both zero and one")


def _tune(X_train, y_train, X_tuning, y_tuning, values, builder, parameter, prefer):
    rows = []
    for value in values:
        pipeline = builder(value)
        pipeline.fit(X_train, y_train)
        if parameter == "C":
            scores = pipeline.predict_proba(X_tuning)[:, 1]
        else:
            scores = pipeline.predict(X_tuning)
        rows.append({"parameter": parameter, "value": value,
                     "inner_tuning_auc": roc_auc_score(y_tuning, scores)})
    table = pd.DataFrame(rows)
    ascending = prefer == "smaller"
    best = table.sort_values(
        ["inner_tuning_auc", "value"], ascending=[False, ascending], kind="stable"
    ).iloc[0]
    table["selected"] = table["value"].eq(best["value"])
    return float(best["value"]), table


def fit_glassbox_models(
    X_train,
    y_train,
    X_validation,
    dates,
    *,
    alphas=ALPHAS,
    c_values=C_VALUES,
):
    """Tune on the final 20% of training dates, then refit both models on all training rows."""
    _validate_inputs(X_train, y_train, dates)
    expected = NUMERIC_FIELDS + CATEGORICAL_FIELDS
    X_train = X_train.loc[:, expected]
    X_validation = X_validation.loc[:, expected]
    inner_ids, tuning_ids, cutoff = temporal_inner_split(dates)
    y_inner = y_train.loc[inner_ids]
    y_tuning = y_train.loc[tuning_ids]
    if set(y_inner.unique()) != {0, 1} or set(y_tuning.unique()) != {0, 1}:
        raise ValueError("Both inner temporal splits must contain both target classes")

    best_alpha, l2_tuning = _tune(
        X_train.loc[inner_ids], y_inner, X_train.loc[tuning_ids], y_tuning,
        alphas, _lasso, "alpha", "larger",
    )
    best_c, logloss_tuning = _tune(
        X_train.loc[inner_ids], y_inner, X_train.loc[tuning_ids], y_tuning,
        c_values, _logistic, "C", "smaller",
    )
    l2_tuning.insert(0, "model", "Glassbox_L2")
    logloss_tuning.insert(0, "model", "Glassbox_Logloss")

    l2 = _lasso(best_alpha).fit(X_train, y_train)
    logloss = _logistic(best_c).fit(X_train, y_train)
    scores = pd.DataFrame(
        {
            "Glassbox_L2": l2.predict(X_validation),
            "Glassbox_Logloss": logloss.predict_proba(X_validation)[:, 1],
        },
        index=X_validation.index,
    )
    scores.index.name = "RefId"
    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Glassbox validation scores must all be finite")
    return {
        "models": {"Glassbox_L2": l2, "Glassbox_Logloss": logloss},
        "scores": scores,
        "tuning": pd.concat([l2_tuning, logloss_tuning], ignore_index=True),
        "inner_training_end": cutoff,
    }


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


def save_top_effects(coefficients, path, each_direction=6):
    """Plot positive and negative coefficients separately for each loss."""
    specifications = [
        ("Glassbox_L2_coefficient", "Glassbox L2: risk-score change"),
        ("Glassbox_Logloss_coefficient", "Glassbox Logloss: log-odds change"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
    for axis, (column, title) in zip(axes, specifications):
        nonzero = coefficients.loc[coefficients[column].ne(0)].copy()
        selected = pd.concat(
            [nonzero.nsmallest(each_direction, column), nonzero.nlargest(each_direction, column)]
        ).drop_duplicates("encoded_feature").sort_values(column)
        labels = selected["original_field"].where(
            selected["feature_level"].eq(""),
            selected["original_field"] + "=" + selected["feature_level"],
        )
        colors = np.where(selected[column].ge(0), "#C44E52", "#4C72B0")
        axis.barh(labels, selected[column], color=colors)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Coefficient (model-specific scale)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_glassbox(
    *,
    processed=PROCESSED,
    raw=RAW,
    directory=ROOT / "results" / "validation" / "acquisition_prices",
):
    """Run the complete Glassbox experiment and write the four agreed artifacts."""
    X_train, y_train, X_validation, y_validation = load_development(
        output=processed, raw=raw, include_prices=True
    )
    dates = load_training_dates(processed)
    if not dates.index.equals(X_train.index):
        raise ValueError("Prepared training dates do not match load_development")
    fitted = fit_glassbox_models(X_train, y_train, X_validation, dates)
    directory = Path(directory)
    report = save_validation(
        y_validation, fitted["scores"], directory, family="glassbox"
    )
    coefficients = coefficient_table(fitted["models"])
    coefficients.to_csv(directory / "glassbox_coefficients.csv", index=False)
    save_top_effects(coefficients, directory / "glassbox_top_effects.png")
    fitted.update({"auc": report, "coefficients": coefficients})
    return fitted


if __name__ == "__main__":
    result = run_glassbox()
    print(result["tuning"].to_string(index=False))
    print("\nValidation AUC\n", result["auc"].to_string())
