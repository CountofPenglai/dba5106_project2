"""Public notebook entry point and refitting of a validation-selected configuration."""
import pandas as pd

from .evaluation import risk_scores
from .glassbox import SPECIFICATIONS, tune_glassbox
from .preprocessing import OUT, TARGET, prepare_feature_pair, read_development_rows
from .registry import MODELS, family_for
from .tree_models import encode_inputs, make_tree
from .tree_training import tune_family


def train_family(family):
    if family not in MODELS:
        raise ValueError(f"Unknown family: {family}")
    train, validation = read_development_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    if family == "glassbox":
        return tune_glassbox(train, validation)
    return tune_family(family, train, validation)


def selected_model(name):
    family = family_for(name)
    if family == "glassbox":
        table = pd.read_csv(OUT / "glassbox_tuning.csv")
        value, = table.loc[table.selected & table.model.eq(name), "value"]
        return SPECIFICATIONS[name][1](value)
    table = pd.read_csv(OUT / f"{family}_finalists.csv")
    row, = table.loc[table.selected & table.model.eq(name)].itertuples()
    return make_tree(name, int(row.depth), int(row.regularization))


def refit_score(name, train, later):
    x, v = prepare_feature_pair(train, later)
    a, b = encode_inputs(family_for(name), x, v)
    model = selected_model(name).fit(a, train.loc[x.index, TARGET])
    return model, pd.Series(risk_scores(model, b), index=v.index), v
