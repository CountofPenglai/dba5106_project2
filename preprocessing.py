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
        frame[col] = frame[col].str.strip().str.upper().replace("", pd.NA).fillna("__MISSING__")
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


def load_development(output=PROCESSED):
    """Load only the two labelled splits used by the modelling notebooks."""
    def load(name):
        frame = pd.read_csv(output / f"{name}.csv", index_col="RefId",
                            dtype={c: str for c in CATEGORICAL}, keep_default_na=False)
        return frame[FEATURES], frame[TARGET]

    return (*load("train"), *load("validation"))


if __name__ == "__main__":
    prepare_data()
