# Karta WAQI postaja po odabranoj državi

import os
import requests
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import dash_leaflet as dl

from src.state.cache import cache  
from src.utils.io import load_cities


WAQI_TOKEN = os.getenv("WAQI_TOKEN", "PASTE_YOUR_TOKEN_HERE")

# Uzet ćemo dostupne države iz tablice (isti izvor kao drugdje)
_df_cities = load_cities()
COUNTRIES = sorted(_df_cities["Country"].dropna().unique())
DEFAULT_COUNTRY = "Croatia" if "Croatia" in COUNTRIES else (COUNTRIES[0] if COUNTRIES else None)

# Boje po AQI kategorijama (iste nijanse kao u “Informacije” tabu)
def aqi_color(aqi_val):
    try:
        v = float(aqi_val)
    except Exception:
        return "#999999"
    if v <= 50:   return "#aedcae"
    if v <= 100:  return "#ffea61"
    if v <= 150:  return "#ff9100"
    if v <= 200:  return "#ff4500"
    if v <= 300:  return "#9f5ea5"
    return "#7e0023"

# Keširani fetch stations by country using WAQI /search
def fetch_stations(country: str):
    if not country or WAQI_TOKEN == "PASTE_YOUR_TOKEN_HERE":
        return []

    key = f"waqi_search:{country}"
    if cache:
        cached = cache.get(key)
        if cached is not None:
            return cached

    url = f"https://api.waqi.info/search/?token={WAQI_TOKEN}&keyword={requests.utils.quote(country)}"
    stations = []
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        js = r.json()
        if js.get("status") != "ok":
            return []
        results = js.get("data") or []
        for item in results:
            st = {}
            st["name"] = (item.get("station") or {}).get("name")
            st["aqi"] = item.get("aqi")
            geo = (item.get("station") or {}).get("geo") or item.get("geo")
            if isinstance(geo, (list, tuple)) and len(geo) == 2:
                st["lat"] = float(geo[0])
                st["lon"] = float(geo[1])
            else:
                continue
            st["time"] = (item.get("time") or {}).get("stime") or item.get("time")
            st["uid"] = item.get("uid")
            stations.append(st)
    except Exception:
        stations = []

    if cache:
        cache.set(key, stations, timeout=300)  # 5 min

    return stations


layout = dbc.Container([
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id="map_country",
            options=[{"label": c, "value": c} for c in COUNTRIES],
            value=DEFAULT_COUNTRY,
            placeholder="Odaberi državu"
        ), md=4),
        dbc.Col(dbc.Button("Osvježi", id="map_refresh", color="primary", className="w-100"), md=2),
    ], className="g-2"),

    html.Div(id="map_warning", className="mt-2"),

    # Leaflet mapa
    dl.Map(
        id="aq_map",
        center=(45.5, 16.0),
        zoom=6,
        bounds=None,
        children=[
            dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"),
            dl.LayerGroup(id="aq_markers"),
        ],
        style={"width": "100%", "height": "70vh", "marginTop": "8px"}
    ),


    # Legenda (jednostavna blok legenda u kutu)
    html.Div([
        html.Div("Legenda AQI", style={"fontWeight": 700, "marginBottom": "4px"}),
        html.Div("0–50: Dobro",   style={"backgroundColor": "#aedcae", "padding": "2px 6px"}),
        html.Div("51–100: Umjereno", style={"backgroundColor": "#ffea61", "padding": "2px 6px"}),
        html.Div("101–150: Nezdravo osjetlj.", style={"backgroundColor": "#ff9100", "padding": "2px 6px"}),
        html.Div("151–200: Nezdravo", style={"backgroundColor": "#ff4500", "color": "white", "padding": "2px 6px"}),
        html.Div("201–300: Vrlo nezdravo", style={"backgroundColor": "#9f5ea5", "color": "white", "padding": "2px 6px"}),
        html.Div("300+: Opasno", style={"backgroundColor": "#7e0023", "color": "white", "padding": "2px 6px"}),
    ], style={
        "position": "absolute", "bottom": "100px", "right": "30px",
        "background": "rgba(255,255,255,0.9)", "padding": "5px 5px",
        "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,.2)", "zIndex": 999
    })

], fluid=True)


def register_callbacks(app):
    # Glavni callback: dohvat postaja + izgradnja markera i fit bounds
    @app.callback(
        Output("aq_markers", "children"),
        Output("aq_map", "bounds"),
        Output("map_warning", "children"),
        Input("map_country", "value"),
        Input("map_refresh", "n_clicks"),
        prevent_initial_call=False
    )
    def _update_map(country, _):
        if WAQI_TOKEN == "PASTE_YOUR_TOKEN_HERE":
            return [], None, dbc.Alert("Postavi WAQI_TOKEN (env).", color="warning")

        stations = fetch_stations(country)
        if not stations:
            return [], None, dbc.Alert("Nema dostupnih postaja za odabranu državu ili nema podataka.", color="secondary")

        markers, lats, lons = [], [], []
        for st in stations:
            lat, lon = float(st["lat"]), float(st["lon"])
            lats.append(lat); lons.append(lon)
            popup = html.Div([
                html.B(st.get("name") or "(nepoznato)"),
                html.Div(f"AQI: {st.get('aqi')}"),
                html.Div(f"Vrijeme: {st.get('time') or ''}")
            ], style={"minWidth": "160px"})
            markers.append(
                dl.CircleMarker(
                    center=(lat, lon),
                    radius=7,
                    color="#333",
                    weight=1,
                    fillColor=aqi_color(st.get("aqi")),
                    fillOpacity=0.9,
                    children=dl.Popup(popup)
                )
            )

        bounds = [[min(lats), min(lons)], [max(lats), max(lons)]] if lats and lons else None

        if bounds:
            bounds = [[float(bounds[0][0]), float(bounds[0][1])],
                    [float(bounds[1][0]), float(bounds[1][1])]]

        return markers, bounds, None
