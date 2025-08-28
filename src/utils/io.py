import os
import pandas as pd

# Učitavanje i obrada podataka o gradovima i državama
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

# Funkcija za učitavanje popisa gradova i država
def load_cities():
    path = os.path.join(DATA_DIR, "gradovi_drzave.xlsx")
    df = pd.read_excel(path)
    needed = {"Country", "City"}
    if not needed.issubset(df.columns):
        raise ValueError("gradovi_drzave.xlsx mora sadržavati stupce: Country, City")
    return df

# Funkcija koja za odabranu državu vraća listu gradova
def cities_for_country(df_cities, country: str):
    if not country: return []
    subset = df_cities[df_cities["Country"] == country]
    return sorted(subset["City"].dropna().unique())
