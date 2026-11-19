"""Build the pre-bid Carvana dataset. Run: python preprocessing.py"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "DontGetKicked"
PROCESSED = ROOT / "data" / "processed"
TARGET = "IsBadBuy"
NUMERIC = ["VehicleAge", "VehOdo"]
CATEGORICAL = ["Make", "Model", "Size", "Transmission", "Auction", "VNST", "IsOnlineSale"]
FEATURES = NUMERIC + CATEGORICAL
TRAIN_END = pd.Timestamp("2010-06-08")
VALIDATION_END = pd.Timestamp("2010-09-29")
MIN_MODEL_COUNT = 20
TRIM_FIELDS = ["Trim", "SubModel"]
ACQUISITION_PRICES = [
    "MMRAcquisitionAuctionAveragePrice", "MMRAcquisitionAuctionCleanPrice",
    "MMRAcquisitionRetailAveragePrice", "MMRAcquisitonRetailCleanPrice",
]


def clean_category(values):
    return values.str.strip().str.upper().replace("", pd.NA).fillna("__MISSING__")


def read_auctions(path, *, labelled):
    columns = ["RefId", "PurchDate", *FEATURES] + ([TARGET] if labelled else [])
    frame = pd.read_csv(path, usecols=columns, dtype={c: "string" for c in CATEGORICAL})
    frame = frame.set_index("RefId", verify_integrity=True)
    frame["PurchDate"] = pd.to_datetime(frame["PurchDate"], format="%m/%d/%Y", errors="raise")
    if frame.index.hasnans or frame["PurchDate"].isna().any():
        raise ValueError("Each auction needs a RefId and a purchase date")
    if labelled and not frame[TARGET].isin([0, 1]).all():
        raise ValueError("IsBadBuy must contain only 0 and 1")
    for col in CATEGORICAL:
        frame[col] = clean_category(frame[col])
    return frame.sort_values("PurchDate", kind="stable")


def prepare_data(raw=RAW, output=PROCESSED):
    labelled = read_auctions(raw / "training.csv", labelled=True)
    date = labelled["PurchDate"]
    splits = {
        "train": labelled.loc[date <= TRAIN_END].copy(),
        "validation": labelled.loc[(date > TRAIN_END) & (date <= VALIDATION_END)].copy(),
        "holdout": labelled.loc[date > VALIDATION_END].copy(),
        "kaggle_unlabelled": read_auctions(raw / "test.csv", labelled=False),
    }
    if any(frame.empty for frame in splits.values()):
        raise ValueError("An empty date split: check the input and configured cutoff dates")
    train = splits["train"]
    medians = train[NUMERIC].median()
    counts = train["Model"].value_counts()
    retained_models = set(counts.index[counts >= MIN_MODEL_COUNT]) | {"__MISSING__"}

    output.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame[NUMERIC] = frame[NUMERIC].fillna(medians)
        frame["Model"] = frame["Model"].where(frame["Model"].isin(retained_models), "__OTHER__")
        frame.to_csv(output / f"{name}.csv", index=True)
        print(f"{name:18} {len(frame):6,}  {frame.PurchDate.min():%Y-%m-%d} to {frame.PurchDate.max():%Y-%m-%d}")


def read_development_rows(output=PROCESSED, *, raw=RAW, include_prices=True):
    """Read raw values only for the training and validation IDs, in split order."""
    def split_ids(name):
        return pd.Index(pd.read_csv(output / f"{name}.csv", usecols=["RefId"])["RefId"])

    train_ids, val_ids = split_ids("train"), split_ids("validation")
    selected = train_ids.append(val_ids)
    if not selected.is_unique:
        raise ValueError("Duplicate or overlapping development IDs")
    path = raw / "training.csv"
    ids = pd.read_csv(path, usecols=["RefId"])["RefId"]
    skip = ids.index[~ids.isin(selected)] + 1
    categories = CATEGORICAL + TRIM_FIELDS
    prices = ACQUISITION_PRICES if include_prices else []
    columns = ["RefId", "PurchDate", TARGET, *FEATURES, *TRIM_FIELDS, *prices]
    rows = pd.read_csv(path, usecols=columns, skiprows=skip,
                       dtype={name: "string" for name in categories}).set_index("RefId", verify_integrity=True)
    rows["PurchDate"] = pd.to_datetime(rows["PurchDate"], format="%m/%d/%Y")
    for name in categories:
        rows[name] = clean_category(rows[name])
    train, val = rows.loc[train_ids], rows.loc[val_ids]
    if not (train.PurchDate.le(TRAIN_END).all() and val.PurchDate.gt(TRAIN_END).all()
            and val.PurchDate.le(VALIDATION_END).all()):
        raise ValueError("Development IDs do not match the chronological cutoffs")
    return train, val


def prepare_feature_pair(train, later, *, include_prices=True):
    """Fit median substitution and rare-model grouping on the supplied history."""
    prices = ACQUISITION_PRICES if include_prices else []
    fields = FEATURES + TRIM_FIELDS + prices
    history, future = train[fields].copy(), later[fields].copy()
    numeric = NUMERIC + prices
    for frame in (history, future):
        if prices:
            frame[prices] = frame[prices].where(frame[prices].gt(0))
    medians = history[numeric].median()
    if medians.isna().any():
        raise ValueError("A numeric field has no usable training values")
    counts = history["Model"].value_counts()
    retained = set(counts.index[counts >= MIN_MODEL_COUNT]) | {"__MISSING__"}
    for frame in (history, future):
        frame[numeric] = frame[numeric].fillna(medians)
        frame["Model"] = frame["Model"].where(frame["Model"].isin(retained), "__OTHER__")
    return history, future


def load_development(output=PROCESSED, *, raw=RAW, include_prices=True):
    train, val = read_development_rows(output, raw=raw, include_prices=include_prices)
    X_train, X_val = prepare_feature_pair(train, val, include_prices=include_prices)
    return X_train, train[TARGET], X_val, val[TARGET]


if __name__ == "__main__":
    prepare_data()
