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

# Opisi polutanata (ključ je standardiziran, vidi NAME_ALIASES ispod)
POLLUTANT_DESCRIPTIONS = {
    "PM2.5": "PM2.5 – Fine lebdeće čestice promjera manjeg od 2.5 μm. Vrlo opasne jer prodiru duboko u pluća i krvotok. Najčešće potječu iz prometa, grijanja na kruta goriva, industrije. Glavni uzrok respiratornih i kardiovaskularnih bolesti.",
    "PM10": "PM10 su grublje čestice promjera manjeg od 10 mikrometara. Iako su nešto manje štetne od PM2.5, također negativno utječu na dišni sustav. Dolaze iz prometa, građevinskih aktivnosti, poljoprivrede te iz vjetrom nošene prašine.",
    "NO2": "Dušikov dioksid (NO2) nastaje izgaranjem goriva, najviše u urbanim područjima s gustim prometom. Njegova prisutnost u zraku iritira dišne puteve te pridonosi stvaranju prizemnog ozona i smoga.",
    "SO2": "Sumporov dioksid (SO2) nastaje pri sagorijevanju ugljena, nafte i u industrijskim procesima. Izaziva iritaciju očiju, nosa i grla, pogoršava simptome astme te se može pretvoriti u sulfatne čestice koje pridonose nastanku kiselih kiša.",
    "O3":  "Ozon (O3) na prizemnoj razini sekundarni je onečišćivač koji nastaje kemijskim reakcijama dušikovih oksida i hlapljivih organskih spojeva na suncu. Glavna je komponenta fotokemijskog smoga i uzrokuje iritaciju dišnih puteva, glavobolje te smanjuje radnu sposobnost pluća.",
    "CO":  "Ugljični monoksid (CO) je bezbojan i otrovan plin koji nastaje nepotpunim izgaranjem goriva, primjerice u prometu ili ložištima. Opasan je jer se veže za hemoglobin i time smanjuje sposobnost krvi da prenosi kisik, a u visokim koncentracijama može biti smrtonosan.",
    "DEW": "Temperatura rosišta (DEW) je temperatura na kojoj zrak postaje zasićen vlagom i počinje kondenzirati. Važan je pokazatelj vlažnosti zraka i utječe na ljudsku udobnost te širenje zagađivača.",
    "PRESSURE": "Atmosferski tlak (PRESSURE) je sila koju zrak vrši na površinu. Promjene tlaka utječu na vremenske uvjete i ljudsko zdravlje, posebno kod osoba osjetljivih na promjene vremena.",
    "TEMPERATURE": "Temperatura zraka (TEMPERATURE) mjeri toplinu ili hladnoću okoline. Utječe na ljudsku udobnost, zdravlje te kemijske reakcije u atmosferi koje mogu povećati razine zagađivača.",
    "WIND_SPEED": "Brzina vjetra (WIND_SPEED) mjeri koliko brzo zrak putuje. Vjetar može razrijediti i raspršiti zagađivače, smanjujući njihovu koncentraciju u zraku.",
    "WIND_GUST": "Nagli udari vjetra (WIND_GUST) su kratkotrajna povećanja brzine vjetra. Mogu utjecati na širenje zagađivača i stvaranje turbulencija u atmosferi.",
    "HUMIDITY": "Relativna vlažnost (HUMIDITY) je omjer stvarne količine vodene pare u zraku prema maksimalnoj količini koju zrak može zadržati na određenoj temperaturi. Visoka vlažnost može povećati osjećaj topline i utjecati na širenje zagađivača.",
}

# Mapiranje naziva iz tablice -> ključevi u POLLUTANT_DESCRIPTIONS
NAME_ALIASES = {
    "pm25": "PM2.5",
    "pm2.5": "PM2.5",
    "pm_2_5": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "o3": "O3",
    "so2": "SO2",
    "co": "CO",
    "dew": "DEW",
    "p": "PRESSURE",
    "t": "TEMPERATURE",
    "w": "WIND_SPEED",
    "wg": "WIND_GUST",
}

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
        dbc.Col(dbc.Button("Osvježi", id="refresh_city", color="secondary", className="w-100"), md=2),
    ], className="g-2"),

    dbc.Row([
        dbc.Col(dbc.Card(id="city_info", className="mt-3"), md=8),
        dbc.Col(legend_card, md=4),
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
                        tooltip_data=[],
                        tooltip_delay=0,
                        tooltip_duration=None,
                        style_table={"overflowY": "auto"},
                        style_cell={"textAlign": "left", "padding": "6px"},
                        style_data_conditional=[
                            {"if": {"column_id": "Onečišćivač"}, "cursor": "pointer"},
                            {
                                "if": {"state": "active"},   # deaktivira highlight kod klika
                                "backgroundColor": "inherit",
                            },
                            {
                                "if": {"state": "selected"}, # deaktivira highlight kod selekcije
                                "backgroundColor": "inherit",
                            }
                        ],
                        page_action="none",
                    )
                ])
            ),
            md=12, className="mt-3"
        )
    ]),

    # Modal za opis onečišćivača
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Opis onečišćivača")),
            dbc.ModalBody(id="pollutant_description"),
            dbc.ModalFooter(
                dbc.Button("Zatvori", id="close_pollutant_modal", className="ms-auto", n_clicks=0)
            ),
        ],
        id="pollutant_modal",
        is_open=False,
    )
], fluid=True)  

# Keširanje WAQI odgovora
def _cached_city_data(city: str, token: str):
    """Dohvaća podatke za grad s WAQI API-ja uz spremanje u cache (5 minuta)."""
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

    #AQI kartica + tablica
    @app.callback(
        Output("city_info", "children"),
        Output("pollutants_table", "data"),
        Output("pollutants_table", "tooltip_data"),
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

        name_map = {
            "pm25": "PM2.5",
            "pm10": "PM10",
            "no2": "NO2",
            "o3": "O3",
            "so2": "SO2",
            "co": "CO",
            "dew": "DEW",
            "p": "PRESSURE",
            "t": "TEMPERATURE",
            "w": "WIND_SPEED",
            "wg": "WIND_GUST",
            "h": "HUMIDITY",
        }

        rows = []
        tooltips = []
        for key in POLLUTANT_ORDER:
            v = iaqi.get(key)
            if isinstance(v, dict) and "v" in v:
                label = key  # što će pisati u tablici
                rows.append({"Onečišćivač": label, "Vrijednost": v["v"]})

                # tekst tooltipa (ako postoji u opisima)
                descr_key = name_map.get(key, key).upper()
                desc = POLLUTANT_DESCRIPTIONS.get(descr_key) or POLLUTANT_DESCRIPTIONS.get(descr_key.title()) or ""
                tooltips.append({
                    "Onečišćivač": {"value": desc, "type": "text"}
                })

        return aqi_card, rows, tooltips

    # Modal s opisom onečišćivača
    @app.callback(
        Output("pollutant_modal", "is_open"),
        Output("pollutant_description", "children"),
        Input("pollutants_table", "selected_rows"),
        Input("close_pollutant_modal", "n_clicks"),
        State("pollutants_table", "data"),
        State("pollutant_modal", "is_open"),
        prevent_initial_call=True
    )
    def toggle_pollutant_modal(active_cell, close_click, table_data, is_open):
        if active_cell and not is_open:
            row = table_data[active_cell["row"]]
            pollutant = row["Onečišćivač"]
            desc = POLLUTANT_DESCRIPTIONS.get(pollutant.upper(), "Nema opisa za ovaj parametar.")
            return True, desc
        if close_click and is_open:
            return False, ""
        return is_open, ""