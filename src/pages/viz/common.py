# Zajedničke pomoćne varijable i funkcije za sve tabove u "Vizualizacijama"

from __future__ import annotations
from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import List
from src.utils.io import load_cities, cities_for_country
from src.utils.data import load_viz_data, all_parameters, years_available

# Podaci se učitavaju jednom i koriste u svim tabovima
df_cities = load_cities()
viz_df = load_viz_data()
PARAMS = all_parameters(viz_df)
YEARS = years_available(viz_df)
COUNTRIES = sorted(df_cities["Country"].dropna().unique())
DEFAULT_COUNTRY = "Croatia" if "Croatia" in COUNTRIES else (COUNTRIES[0] if COUNTRIES else None)
DEFAULT_YEAR = max(YEARS) if YEARS else None
DEFAULT_PARAM = "pm25" if "pm25" in PARAMS else (PARAMS[0] if PARAMS else None)

# Definicija osnovnog seta polutanata za prikaz prosjeka
CORE_PARAMS = [p for p in ["pm25", "pm10", "no2", "o3", "so2", "co"] if p in PARAMS] or PARAMS[:6]

# Gradovi dostupni i u datasetu i u tablici država/gradova
def cities_for_country_in_viz(country: str) -> List[str]:
    if not country:
        return []
    present = set(viz_df["grad"].astype(str).unique())
    allowed = set(cities_for_country(df_cities, country))
    return sorted(allowed & present)

# Godine dostupne za odabrani grad
def years_for_city(city: str) -> List[int]:
    if not city:
        return []
    s = viz_df[viz_df["grad"].astype(str) == str(city)]["datum"].dt.year
    return sorted(s.dropna().astype(int).unique().tolist())

# Pomoćna funkcija za slaganje Bootstrap redaka
def row(*cols):
    return dbc.Row([dbc.Col(c, md=md) for c, md in cols], className="g-2")
