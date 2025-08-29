# src/utils/data.py
# Utilities for loading and querying the historical AQ dataset (podaciv2.xlsx)

from __future__ import annotations
import os
import re
from functools import lru_cache
from typing import List

import pandas as pd

# Base /data folder (same pattern you used in io.py)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
DATA_FILE = os.path.join(DATA_DIR, "podaciv2.xlsx")


def _clean_name(s: str) -> str:
    """Rough equivalent of janitor::clean_names -> snake_case, ascii-ish."""
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", "_", s)          # non-word to underscore
    s = re.sub(r"\s+", "_", s)              # spaces to underscore
    s = re.sub(r"_+", "_", s)               # collapse repeats
    return s.strip("_")


@lru_cache(maxsize=1)
def load_viz_data() -> pd.DataFrame:
    """
    Load and clean podaciv2.xlsx once (cached).
    - snake_case column names
    - ensure 'grad' (city) exists and is string
    - ensure 'datum' exists and is datetime64[ns]
    - keep original numeric columns as-is (coerce where needed)
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Expected dataset at {DATA_FILE}. Copy podaciv2.xlsx into the /data folder."
        )

    df = pd.read_excel(DATA_FILE)

    # Clean column names (similar to janitor::clean_names)
    df.columns = [_clean_name(c) for c in df.columns]

    # Try to find date & city columns if names vary a bit
    cols = set(df.columns)

    # City column
    city_col = None
    for cand in ("grad", "city", "mjesto"):
        if cand in cols:
            city_col = cand
            break
    if city_col is None:
        raise ValueError("Could not find a city column (expected one of: grad/city/mjesto).")

    # Date column
    date_col = None
    for cand in ("datum", "date", "dt"):
        if cand in cols:
            date_col = cand
            break
    if date_col is None:
        # Heuristic: first column that parses to many datetimes
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
                if parsed.notna().sum() >= len(df) * 0.6:
                    date_col = c
                    df[c] = parsed
                    break
            except Exception:
                pass
        if date_col is None:
            raise ValueError("Could not find a date column (expected: datum/date/dt).")

    # Standardize required columns
    if date_col != "datum":
        df = df.rename(columns={date_col: "datum"})
    if city_col != "grad":
        df = df.rename(columns={city_col: "grad"})

    # Types
    df["grad"] = df["grad"].astype(str)
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")

    # Drop rows without date
    df = df[df["datum"].notna()].copy()

    # Optional: try to coerce typical numeric pollutant columns to numbers
    # (non-numeric become NaN; commas handled)
    for c in df.columns:
        if c in ("datum", "grad"):
            continue
        if df[c].dtype == object:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)


def all_parameters(df: pd.DataFrame | None = None) -> List[str]:
    """List of parameter columns (exclude 'datum' & 'grad')."""
    if df is None:
        df = load_viz_data()
    return [c for c in df.columns if c not in ("datum", "grad")]


def countries_from_cities_table(cities_df: pd.DataFrame) -> List[str]:
    """Helper if you want to reuse the cities Excel for country dropdowns."""
    return sorted(cities_df["Country"].dropna().unique())


def cities_for_country_in_viz(df: pd.DataFrame, country: str, df_cities: pd.DataFrame) -> List[str]:
    """
    Given viz data and the Country/City mapping table, return
    city names available for a chosen country and also present in viz dataset.
    """
    if not country:
        return []
    from src.utils.io import cities_for_country  # reuse existing helper
    cities = set(cities_for_country(df_cities, country))
    present = set(df["grad"].unique())
    return sorted(cities & present)


def years_available(df: pd.DataFrame | None = None) -> List[int]:
    """Distinct years available in the dataset."""
    if df is None:
        df = load_viz_data()
    return sorted(df["datum"].dt.year.dropna().astype(int).unique().tolist())


def monthly_series(df: pd.DataFrame, city: str, params: List[str], year: int) -> pd.DataFrame:
    """
    Aggregate to monthly mean for a given city, parameters, and year.
    Returns a long-form df: month, parameter, value
    """
    x = df[df["grad"] == str(city)].copy()
    x["year"] = x["datum"].dt.year
    x = x[x["year"] == int(year)]
    x["month"] = x["datum"].values.astype("datetime64[M]")  # floor to month

    keep = ["month"] + [p for p in params if p in x.columns]
    x = x[keep].groupby("month", as_index=False).mean()

    # to long format
    out = x.melt(id_vars="month", var_name="parameter", value_name="value").dropna()
    return out.sort_values(["parameter", "month"])
