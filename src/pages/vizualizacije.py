# src/pages/vizualizacije.py
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from src.utils.io import load_cities, cities_for_country
from src.utils.data import (
    load_viz_data,
    all_parameters,
    years_available,
    monthly_series,
)

# ---------- Data (load once) ----------
df_cities = load_cities()
viz_df = load_viz_data()
PARAMS = all_parameters(viz_df)                  # e.g. ["pm25","pm10","no2",...]
YEARS = years_available(viz_df)                  # e.g. [2021, 2022, ...]
COUNTRIES = sorted(df_cities["Country"].dropna().unique())

DEFAULT_PARAM = "pm25" if "pm25" in PARAMS else (PARAMS[0] if PARAMS else None)
DEFAULT_YEAR = max(YEARS) if YEARS else None
DEFAULT_COUNTRY = "Croatia" if "Croatia" in COUNTRIES else (COUNTRIES[0] if COUNTRIES else None)


def cities_for_country_in_viz(country: str):
    """Only cities that exist in both the country list and viz dataset."""
    if not country:
        return []
    present = set(viz_df["grad"].astype(str).unique())
    allowed = set(cities_for_country(df_cities, country))
    return sorted(allowed & present)


# Helper to build Bootstrap rows
def _row(*cols):
    return dbc.Row([dbc.Col(c, md=md) for c, md in cols], className="g-2")


# ---------- Tab: Usporedba parametara ----------
tab_usporedba = dbc.Container([
    _row(
        (dcc.Dropdown(
            id="viz_country",
            options=[{"label": c, "value": c} for c in COUNTRIES],
            value=DEFAULT_COUNTRY,
            placeholder="Odaberi državu"
        ), 4),
        (dcc.Dropdown(
            id="viz_city",
            placeholder="Odaberi grad"
        ), 4),
        (dcc.Dropdown(
            id="viz_year",
            options=[{"label": str(y), "value": int(y)} for y in YEARS],
            value=DEFAULT_YEAR,
            placeholder="Odaberi godinu"
        ), 4),
    ),
    html.Hr(),
    _row(
        (dcc.Dropdown(
            id="viz_params",
            options=[{"label": p.upper(), "value": p} for p in PARAMS],
            value=[DEFAULT_PARAM] if DEFAULT_PARAM else [],
            placeholder="Odaberi parametre (max 5)",
            multi=True,
            clearable=True,
        ), 12),
    ),
    html.Small(
        "Savjet: dodajte/uklonite iz izbornika. Maksimalno 5 parametara za usporedbu.",
        className="text-muted"
    ),
    dcc.Graph(id="viz_multi_param_plot")
], fluid=True)


# ---------- Other tabs (placeholders for now) ----------
tab_prosjek = dbc.Container([
    _row(
        (dcc.Dropdown(id="avg_country", placeholder="Odaberi državu"), 4),
        (dcc.Dropdown(id="avg_city", placeholder="Odaberi grad"), 4),
        (dcc.Dropdown(id="avg_year", placeholder="Odaberi godinu"), 4),
    ),
    dcc.Graph(id="avg_pollutants_plot")
], fluid=True)

tab_sezonski = dbc.Container([
    _row(
        (dcc.Dropdown(id="season_country", placeholder="Odaberi državu"), 4),
        (dcc.Dropdown(id="season_city", placeholder="Odaberi grad"), 4),
        (dcc.Dropdown(id="season_param", placeholder="Odaberi parametar"), 4),
    ),
    _row((dbc.Checkbox(id="season_polar", label="Polarni prikaz (radar)"), 4)),
    dcc.Graph(id="seasonal_plot")
], fluid=True)

tab_godisnji = dbc.Container([
    _row(
        (dcc.Dropdown(id="year_trend_country", placeholder="Odaberi državu"), 4),
        (dcc.Dropdown(id="year_trend_city", placeholder="Odaberi grad"), 4),
        (dcc.Dropdown(id="year_trend_parameter", placeholder="Odaberi parametar"), 4),
    ),
    dcc.Graph(id="year_trend_plot")
], fluid=True)

tab_korelacija = dbc.Container([
    _row(
        (dcc.Dropdown(id="corr_country", placeholder="Odaberi državu"), 4),
        (dcc.Dropdown(id="corr_city", placeholder="Odaberi grad"), 4),
        (dcc.Dropdown(id="corr_year", placeholder="Odaberi godinu"), 4),
    ),
    _row((
        dbc.RadioItems(
            id="corr_method",
            options=[{"label": "Pearson", "value": "pearson"},
                     {"label": "Spearman", "value": "spearman"}],
            value="pearson",
            inline=True
        ),
        12
    )),
    dcc.Graph(id="corr_heatmap")
], fluid=True)


# ---------- Page layout + tab router ----------
layout = dbc.Container([
    html.H3("Vizualizacije", className="mb-3"),
    dcc.Tabs(id="viz_tabs", value="usporedba", children=[
        dcc.Tab(label="Usporedba parametara", value="usporedba"),
        dcc.Tab(label="Prosjek po mjesecima", value="prosjek"),
        dcc.Tab(label="Sezonski prikaz", value="sezonski"),
        dcc.Tab(label="Godišnji trend", value="godisnji"),
        dcc.Tab(label="Matrica korelacija", value="korelacija"),
    ]),
    html.Div(id="viz_tabs_content", className="mt-3")
], fluid=True)


def register_callbacks(app):
    # Switch visible tab
    @app.callback(Output("viz_tabs_content", "children"), Input("viz_tabs", "value"))
    def _render_tab(tab):
        return {
            "usporedba": tab_usporedba,
            "prosjek": tab_prosjek,
            "sezonski": tab_sezonski,
            "godisnji": tab_godisnji,
            "korelacija": tab_korelacija,
        }.get(tab, html.Div("Nepoznata kartica"))

    # Populate city options when country changes
    @app.callback(
        Output("viz_city", "options"),
        Output("viz_city", "value"),
        Input("viz_country", "value"),
        prevent_initial_call=False
    )
    def _city_opts(country):
        cities = cities_for_country_in_viz(country)
        opts = [{"label": c, "value": c} for c in cities]
        return opts, (cities[0] if cities else None)

    # Plot: multi-parameter comparison (max 5)
    @app.callback(
        Output("viz_multi_param_plot", "figure"),
        Input("viz_city", "value"),
        Input("viz_year", "value"),
        Input("viz_params", "value"),
        prevent_initial_call=False
    )
    def _plot_multi(city, year, params_selected):
        fig = go.Figure()

        # Guard: need city/year and at least one parameter
        if not (city and year and params_selected):
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            return fig

        # Enforce max 5 parameters
        params = list(params_selected)[:5]

        # Aggregate monthly means for all selected params
        df = monthly_series(viz_df, city, params, int(year))
        if df.empty:
            fig.update_layout(title="Nema podataka za odabrane postavke",
                              margin=dict(l=10, r=10, t=40, b=10))
            return fig

        # One line per parameter
        for param in sorted(df["parameter"].unique()):
            dsub = df[df["parameter"] == param]
            fig.add_trace(go.Scatter(
                x=dsub["month"],
                y=dsub["value"],
                mode="lines+markers",
                name=param.upper()
            ))

        title_params = ", ".join([p.upper() for p in params])
        fig.update_layout(
            title=f"Usporedba parametara – {city} – {year}",
            xaxis_title="Mjesec",
            yaxis_title="Vrijednost",
            hovermode="x unified",
            legend_title=title_params if title_params else "Parametri",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig
