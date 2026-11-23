"""The six candidates in the report, in family and notebook order."""
MODELS = {
    "glassbox": ("Glassbox_L2", "Glassbox_Logloss"),
    "random_forest": ("RF_L2", "RF_Logloss"),
    "catboost": ("CatBoost_L2", "CatBoost_Logloss"),
}


def family_for(model):
    for family, variants in MODELS.items():
        if model in variants:
            return family
    raise ValueError(f"Unknown model: {model}")
