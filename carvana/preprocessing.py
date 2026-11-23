"""Chronological auction splits and shared, training-only feature preparation."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "DontGetKicked"
OUT = ROOT / "results" / "validation" / "drop_training_rows" / "acquisition_prices" / "tuned"
TARGET = "IsBadBuy"
NUMERIC = ["VehicleAge", "VehOdo"]
CATEGORICAL = ["Make", "Model", "Size", "Transmission", "Auction", "VNST", "IsOnlineSale"]
TRAIN_END = pd.Timestamp("2010-06-08")
VALIDATION_END = pd.Timestamp("2010-09-29")
MIN_MODEL_COUNT = 20
TRIM_FIELDS = ["Trim", "SubModel"]
ACQUISITION_PRICES = [
    "MMRAcquisitionAuctionAveragePrice", "MMRAcquisitionAuctionCleanPrice",
    "MMRAcquisitionRetailAveragePrice", "MMRAcquisitonRetailCleanPrice",
]


FEATURES = NUMERIC + CATEGORICAL + TRIM_FIELDS + ACQUISITION_PRICES


def clean_category(values):
    return values.str.strip().str.upper().replace("", pd.NA).fillna("__MISSING__")


def _read_period(raw, *, holdout):
    path = raw / "training.csv"
    # Inspect only dates and identifiers to select the split before reading labels.
    keys = pd.read_csv(path, usecols=["RefId", "PurchDate"])
    dates = pd.to_datetime(keys.PurchDate, format="%m/%d/%Y")
    if keys.RefId.hasnans or not keys.RefId.is_unique or dates.hasnans:
        raise ValueError("Each auction needs a unique RefId and a purchase date")
    selected = dates.gt(VALIDATION_END) if holdout else dates.le(VALIDATION_END)
    rows = pd.read_csv(path, skiprows=keys.index[~selected] + 1,
                       usecols=["RefId", "PurchDate", TARGET, *FEATURES],
                       dtype={name: "string" for name in CATEGORICAL + TRIM_FIELDS})
    rows = rows.set_index("RefId")
    rows["PurchDate"] = pd.to_datetime(rows.PurchDate, format="%m/%d/%Y")
    if rows.empty or not rows[TARGET].isin([0, 1]).all():
        raise ValueError("Expected labelled auctions in the selected period")
    for name in CATEGORICAL + TRIM_FIELDS:
        rows[name] = clean_category(rows[name])
    return rows.sort_values("PurchDate", kind="stable")


def read_development_rows(raw=RAW):
    rows = _read_period(raw, holdout=False)
    train, validation = rows.loc[rows.PurchDate <= TRAIN_END], rows.loc[rows.PurchDate > TRAIN_END]
    if train.empty or validation.empty:
        raise ValueError("Both training and validation periods need auction records")
    return train, validation


def read_holdout_rows(raw=RAW):
    return _read_period(raw, holdout=True)


def missing_quote_bundle(rows):
    auction, clean, retail, clean_retail = ACQUISITION_PRICES
    others_zero = rows[[auction, retail, clean_retail]].eq(0).fillna(False).all(axis=1)
    return rows[clean].eq(1).fillna(False) & others_zero


def prepare_feature_pair(train, later):
    """Filter fitting rows only. Learn medians and vehicle-model frequencies on those rows."""
    history = train.loc[~missing_quote_bundle(train), FEATURES].copy()
    future = later[FEATURES].copy()
    numeric = NUMERIC + ACQUISITION_PRICES
    for frame in (history, future):
        frame[ACQUISITION_PRICES] = frame[ACQUISITION_PRICES].where(frame[ACQUISITION_PRICES].gt(0))
    medians = history[numeric].median()
    if medians.isna().any():
        raise ValueError("A numeric field has no usable training values")
    counts = history["Model"].value_counts()
    retained = set(counts.index[counts >= MIN_MODEL_COUNT]) | {"__MISSING__"}
    for frame in (history, future):
        frame[numeric] = frame[numeric].fillna(medians)
        frame["Model"] = frame["Model"].where(frame["Model"].isin(retained), "__OTHER__")
    return history, future
