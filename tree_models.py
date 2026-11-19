"""Tree estimators and input encoding shared by the notebooks and tuning experiment."""
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from preprocessing import ACQUISITION_PRICES, CATEGORICAL, NUMERIC, TRIM_FIELDS

FAMILIES = {
    "random_forest": ("RF_L2", "RF_Logloss"),
    "catboost": ("CatBoost_L2", "CatBoost_Logloss"),
}
BASELINE = {"random_forest": (14, 12), "catboost": (6, 3)}


def encode_forest(history, later):
    numeric = NUMERIC + [name for name in ACQUISITION_PRICES if name in history]
    encoder = ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL + TRIM_FIELDS),
        ("number", "passthrough", numeric),
    ], sparse_threshold=1.0)
    return encoder.fit_transform(history), encoder.transform(later)


def make_tree(model, depth, regularization):
    if model in FAMILIES["random_forest"]:
        settings = dict(n_estimators=160, max_depth=depth, min_samples_leaf=regularization,
                        max_features="sqrt", bootstrap=True, random_state=42, n_jobs=2)
        if model == "RF_L2":
            return RandomForestRegressor(criterion="squared_error", **settings)
        return RandomForestClassifier(criterion="log_loss", **settings)
    if model not in FAMILIES["catboost"]:
        raise ValueError(f"Unknown tree variant {model}")
    settings = dict(iterations=250, depth=depth, learning_rate=0.05, l2_leaf_reg=regularization,
                    cat_features=CATEGORICAL + TRIM_FIELDS, bootstrap_type="Bernoulli", subsample=0.8,
                    boosting_type="Plain", one_hot_max_size=2, random_seed=42,
                    thread_count=2, verbose=False, allow_writing_files=False)
    if model == "CatBoost_L2":
        return CatBoostRegressor(loss_function="RMSE", **settings)
    return CatBoostClassifier(loss_function="Logloss", **settings)


def risk_scores(estimator, features):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    return estimator.predict(features)
