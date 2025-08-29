# Pomoćne funkcije za učitavanje i pripremu povijesnog skupa podataka (podaciv2.xlsx)

from __future__ import annotations
import os
import re
from functools import lru_cache
from typing import List

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
DATA_FILE = os.path.join(DATA_DIR, "podaciv2.xlsx")


def _clean_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


@lru_cache(maxsize=1)  # učitaj i keširaj dataset jednom (brže izvođenje)
def load_viz_data() -> pd.DataFrame:
    # Učitavanje i čišćenje podataka iz podaciv2.xlsx
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Očekivani dataset nije pronađen: {DATA_FILE}. Kopiraj podaciv2.xlsx u /data mapu."
        )

    df = pd.read_excel(DATA_FILE)

    # Očisti nazive stupaca
    df.columns = [_clean_name(c) for c in df.columns]

    # Detekcija stupca za grad
    cols = set(df.columns)
    city_col = next((c for c in ("grad", "city", "mjesto") if c in cols), None)
    if city_col is None:
        raise ValueError("Nije pronađen stupac za grad (očekuje se: grad/city/mjesto).")

    # Detekcija stupca za datum
    date_col = next((c for c in ("datum", "date", "dt") if c in cols), None)
    if date_col is None:
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
            raise ValueError("Nije pronađen stupac za datum (očekuje se: datum/date/dt).")

    # Standardizacija imena
    if date_col != "datum":
        df = df.rename(columns={date_col: "datum"})
    if city_col != "grad":
        df = df.rename(columns={city_col: "grad"})

    # Tipovi podataka
    df["grad"] = df["grad"].astype(str)
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")

    # Makni redove bez datuma
    df = df[df["datum"].notna()].copy()

    # Pretvoriti vrijednosti polutanata u numerički oblik
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
    #Lista svih parametara
    if df is None:
        df = load_viz_data()
    return [c for c in df.columns if c not in ("datum", "grad")]


def countries_from_cities_table(cities_df: pd.DataFrame) -> List[str]:
    #Dohvaća listu država
    return sorted(cities_df["Country"].dropna().unique())


def cities_for_country_in_viz(df: pd.DataFrame, country: str, df_cities: pd.DataFrame) -> List[str]:
    #Dohvat gradoca
    if not country:
        return []
    from src.utils.io import cities_for_country
    cities = set(cities_for_country(df_cities, country))
    present = set(df["grad"].unique())
    return sorted(cities & present)


def years_available(df: pd.DataFrame | None = None) -> List[int]:
    #Vrati sve godine koje postoje u datasetu.
    if df is None:
        df = load_viz_data()
    return sorted(df["datum"].dt.year.dropna().astype(int).unique().tolist())


def monthly_series(df: pd.DataFrame, city: str, params: List[str], year: int) -> pd.DataFrame:
    # Mjesecni prosjek
    x = df[df["grad"] == str(city)].copy()
    x["year"] = x["datum"].dt.year
    x = x[x["year"] == int(year)]
    x["month"] = x["datum"].values.astype("datetime64[M]")  # zaokruži na mjesec

    keep = ["month"] + [p for p in params if p in x.columns]
    x = x[keep].groupby("month", as_index=False).mean()

    # long format
    out = x.melt(id_vars="month", var_name="parameter", value_name="value").dropna()
    return out.sort_values(["parameter", "month"])
