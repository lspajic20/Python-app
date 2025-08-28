import os
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dash import dash_table

from src.utils.io import load_cities, cities_for_country
from src.utils.aqi import aqi_color, get_city_data
from src.state.cache import cache  # Globalni objekt cache-a

# Učitavanje Excel tablice s gradovima i državama
df_cities = load_cities()
COUNTRIES = sorted(df_cities["Country"].dropna().unique())

# Redoslijed prikaza onečišćivača u tablici
POLLUTANT_ORDER = ["pm25", "pm10", "no2", "o3", "so2", "co", "dew", "p", "t", "w", "wg", "h"]

# Kartica s legendom za AQI boje
legend_card = dbc.Card(
    dbc.CardBody([
        html.Div("0–50: Dobro",   style={"backgroundColor": "#aedcae", "padding": 5}),
        html.Div("51–100: Umjereno", style={"backgroundColor": "#ffea61", "padding": 5}),
        html.Div("101–150: Nezdravo za osjetljive", style={"backgroundColor": "#ff9100", "padding": 5}),
        html.Div("151–200: Nezdravo", style={"backgroundColor": "#ff4500", "padding": 5, "color": "white"}),
        html.Div("201–300: Vrlo nezdravo", style={"backgroundColor": "#9f5ea5", "padding": 5, "color": "white"}),
        html.Div("300+: Opasno", style={"backgroundColor": "#7e0023", "padding": 5, "color": "white"}),
    ]),
    className="mt-2"
)

# Layout stranice Informacije
layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id="country",
                options=[{"label": c, "value": c} for c in COUNTRIES],
                value=("Croatia" if "Croatia" in COUNTRIES else (COUNTRIES[0] if COUNTRIES else None)),
                placeholder="Odaberi državu"
            ),
            md=5
        ),
        dbc.Col(dcc.Dropdown(id="city", placeholder="Odaberi grad"), md=5),
        dbc.Col(dbc.Button("Osvježi", id="refresh_city", color="primary", className="w-100"), md=2), # Refresh gumb
    ], className="g-2"),

    dbc.Row([
        dbc.Col(dbc.Card(id="city_info", className="mt-3"), md=8),  # Kartica s AQI informacijama
        dbc.Col(legend_card, md=4),  # AQI legenda
    ], className="g-2"),

    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5("Pregled parametara", className="mb-3"),
                    dash_table.DataTable(
                        id="pollutants_table",
                        columns=[
                            {"name": "Onečišćivač", "id": "Onečišćivač"},
                            {"name": "Vrijednost", "id": "Vrijednost"},
                        ],
                        data=[],
                        style_table={"overflowY": "auto"},
                        style_cell={"textAlign": "left", "padding": "6px"},
                        page_action="none"
                    )
                ])
            ),
            md=12, className="mt-3"
        )
    ])
], fluid=True)

# Funkcija koja dohvaća podatke uz keširanje
def _cached_city_data(city: str, token: str):
    """
    Dohvaća podatke za grad s WAQI API-ja uz spremanje u cache (5 minuta).
    """
    if cache is None:
        return get_city_data(city, token)

    key = f"waqi:{city}"
    data = cache.get(key)
    if data is not None:
        return data
    data = get_city_data(city, token)
    if data:
        cache.set(key, data, timeout=300)  # 5 minuta
    return data

# Registracija callbackova
def register_callbacks(app):
    # Dinamičko popunjavanje dropdowna gradova kad se promijeni država
    @app.callback(
        Output("city", "options"),
        Output("city", "value"),
        Input("country", "value"),
        prevent_initial_call=False
    )
    def _city_options(country):
        cities = cities_for_country(df_cities, country)
        opts = [{"label": c, "value": c} for c in cities]
        return opts, (cities[0] if cities else None)

    # Gumb Osvježi briše cache za odabrani grad
    @app.callback(
        Output("refresh_city", "n_clicks"),
        Input("refresh_city", "n_clicks"),
        State("city", "value"),
        prevent_initial_call=True
    )
    def _clear_city_cache(n_clicks, city):
        if n_clicks and city and cache is not None:
            cache.delete(f"waqi:{city}")  # Briše cache za taj grad
        return 0

    # Dohvaćanje podataka i popunjavanje AQI kartice i tablice
    @app.callback(
        Output("city_info", "children"),
        Output("pollutants_table", "data"),
        Input("city", "value"),
        Input("refresh_city", "n_clicks"),
        prevent_initial_call=False
    )
    def _aqi_and_table(city, _):
        if not city:
            return dbc.Alert("Odaberite grad.", color="secondary"), []

        token = os.getenv("WAQI_TOKEN", "PASTE_YOUR_TOKEN_HERE")
        if token == "PASTE_YOUR_TOKEN_HERE":
            return dbc.Alert("Postavi WAQI token (env WAQI_TOKEN).", color="warning"), []

        data = _cached_city_data(city, token)
        if not data:
            return dbc.Alert("Nije moguće dohvatiti podatke za grad.", color="danger"), []

        aqi = data.get("aqi")
        dom = data.get("dominentpol", "N/A")
        upd = (data.get("time") or {}).get("s", "Nepoznato")
        aqi_card = dbc.Card(
            dbc.CardBody([
                html.H3(str((data.get("city") or {}).get("name", city))),
                html.H1(f"AQI: {aqi}", style={"fontWeight": 700}),
                html.Div([html.B("Dominantni polutant: "), str(dom)]),
                html.Div([html.B("Zadnje ažuriranje: "), str(upd)]),
            ]),
            style={"backgroundColor": aqi_color(aqi), "color": "white", "border": "none"}
        )

        # Priprema podataka za tablicu onečišćivača
        iaqi = data.get("iaqi") or {}
        rows = []
        for key in POLLUTANT_ORDER:
            v = iaqi.get(key)
            if isinstance(v, dict) and "v" in v:
                rows.append({"Onečišćivač": key, "Vrijednost": v["v"]})

        return aqi_card, rows
