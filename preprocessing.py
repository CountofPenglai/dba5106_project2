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


def load_development(output=PROCESSED, *, raw=RAW, include_prices=True):
    """Load only the two labelled splits used by the modelling notebooks."""
    def load(name):
        frame = pd.read_csv(output / f"{name}.csv", index_col="RefId",
                            dtype={c: str for c in CATEGORICAL}, keep_default_na=False)
        return frame[FEATURES], frame[TARGET]

    X_train, y_train = load("train")
    X_val, y_val = load("validation")
    # Read extra predictors only for the two development splits. The index-only
    # pass lets the CSV parser skip all other rows, including internal holdout.
    path = raw / "training.csv"
    ids = pd.read_csv(path, usecols=["RefId"])["RefId"]
    selected = X_train.index.append(X_val.index)
    skip_rows = ids.index[~ids.isin(selected)] + 1  # row 0 is the CSV header
    fields = TRIM_FIELDS + (ACQUISITION_PRICES if include_prices else [])
    extra = pd.read_csv(path, usecols=["RefId", *fields], skiprows=skip_rows,
                        dtype={c: "string" for c in TRIM_FIELDS}).set_index("RefId")
    if not extra.index.is_unique or len(selected.difference(extra.index)):
        raise ValueError("Cannot match trim details to development vehicles")
    for col in TRIM_FIELDS:
        extra[col] = clean_category(extra[col])
    if include_prices:
        # A valid wholesale or retail quote must be positive. Each median is fit on training cars only.
        quotes = extra[ACQUISITION_PRICES]
        quotes = quotes.where(quotes.gt(0))
        medians = quotes.loc[X_train.index].median()
        if medians.isna().any():
            raise ValueError("A price column has no positive training quotes")
        extra[ACQUISITION_PRICES] = quotes.fillna(medians)
    X_train = X_train.join(extra[fields], validate="one_to_one")
    X_val = X_val.join(extra[fields], validate="one_to_one")
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    prepare_data()
