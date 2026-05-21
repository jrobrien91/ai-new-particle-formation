"""Merge daily UF/FINE, SMPS geometric, and visual label files by date."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"

UF_FILE = PROCESSED_DIR / "daily_UF_minus_FINE_stats.csv"
SMPS_FILE = PROCESSED_DIR / "smps_daily_geometric_local_utc_minus6.csv"
LABEL_FILE = PROCESSED_DIR / "SPG_2023_visClass.xlsx"
OUTPUT_FILE = PROCESSED_DIR / "daily_merged_features_labels.csv"


LABEL_CLASS = {
    "n": "nonevent",
    "u": "undefined",
    "b": "Class II",
    "e": "Class I",
    "a": "Class I",
}
EVENT_CLASS = {
    "n": "non_event",
    "u": "non_event",
    "b": "event",
    "e": "event",
    "a": "event",
}


def normalize_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.strftime("%Y-%m-%d")


def load_uf() -> pd.DataFrame:
    uf = pd.read_csv(UF_FILE)
    uf["date"] = normalize_date(uf["date"])
    return uf


def load_smps() -> pd.DataFrame:
    smps = pd.read_csv(SMPS_FILE)
    smps = smps.rename(columns={"local_date": "date"})
    smps["date"] = normalize_date(smps["date"])
    return smps


def load_labels() -> pd.DataFrame:
    labels = pd.read_excel(LABEL_FILE, header=None, names=["date", "label_code"])
    labels["date"] = normalize_date(labels["date"])
    labels["label_code"] = labels["label_code"].astype(str).str.strip().str.lower()
    labels["label_class"] = labels["label_code"].map(LABEL_CLASS)
    labels["event_class"] = labels["label_code"].map(EVENT_CLASS)
    labels["is_event"] = labels["event_class"].eq("event").astype(int)

    unknown = sorted(labels.loc[labels["label_class"].isna(), "label_code"].unique())
    if unknown:
        raise ValueError(f"Unknown label code(s): {unknown}")
    return labels


def main() -> None:
    uf = load_uf()
    smps = load_smps()
    labels = load_labels()

    merged = (
        uf.merge(smps, on="date", how="inner")
        .merge(labels, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"UF rows: {len(uf)}")
    print(f"SMPS rows: {len(smps)}")
    print(f"Label rows: {len(labels)}")
    print(f"Merged rows with all three sources: {len(merged)}")
    print(f"Wrote: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
