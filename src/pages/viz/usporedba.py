# Tab "Usporedba parametara" – linijski graf s više odabranih zagađivača

from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from src.pages.viz.common import (
    viz_df, PARAMS, YEARS, COUNTRIES, DEFAULT_COUNTRY, DEFAULT_YEAR,
    row, cities_for_country_in_viz
)
from src.utils.data import monthly_series

# Layout za tab
layout = dbc.Container([
    row(
        (dcc.Dropdown(id="viz_country",
                      options=[{"label": c, "value": c} for c in COUNTRIES],
                      value=DEFAULT_COUNTRY, placeholder="Odaberi državu"), 4),
        (dcc.Dropdown(id="viz_city", placeholder="Odaberi grad"), 4),
        (dcc.Dropdown(id="viz_year",
                      options=[{"label": str(y), "value": int(y)} for y in YEARS],
                      value=DEFAULT_YEAR, placeholder="Odaberi godinu"), 4),
    ),
    html.Hr(),
    row((dcc.Dropdown(id="viz_params",
                      options=[{"label": p.upper(), "value": p} for p in PARAMS],
                      value=["pm25"] if "pm25" in PARAMS else [],
                      placeholder="Odaberi parametre (max 5)", multi=True, clearable=True), 12)),
    html.Small("Dodajte/uklonite iz izbornika. Maksimalno 5 parametara.", className="text-muted"),
    dcc.Graph(id="viz_multi_param_plot")
], fluid=True)

# Callbackovi
def register_callbacks(app):
    # Popunjavanje gradova ovisno o državi
    @app.callback(
        Output("viz_city", "options"),
        Output("viz_city", "value"),
        Input("viz_country", "value"),
        prevent_initial_call=False
    )
    def _city_opts(country):
        cities = cities_for_country_in_viz(country)
        return [{"label": c, "value": c} for c in cities], (cities[0] if cities else None)

    # Crtanje grafa usporedbe parametara
    @app.callback(
        Output("viz_multi_param_plot", "figure"),
        Input("viz_city", "value"),
        Input("viz_year", "value"),
        Input("viz_params", "value"),
        prevent_initial_call=False
    )
    def _plot(city, year, params_selected):
        fig = go.Figure()
        if not (city and year and params_selected):
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            return fig

        # ograničen broj parametara na max 5
        params = list(params_selected)[:5]
        df = monthly_series(viz_df, city, params, int(year))
        if df.empty:
            fig.update_layout(title="Nema podataka za odabrane postavke",
                              margin=dict(l=10, r=10, t=40, b=10))
            return fig

        for param in sorted(df["parameter"].unique()):
            dsub = df[df["parameter"] == param]
            fig.add_trace(go.Scatter(x=dsub["month"], y=dsub["value"],
                                     mode="lines+markers", name=param.upper()))

        fig.update_layout(
            title=f"Usporedba parametara – {city} – {year}",
            xaxis_title="Mjesec",
            yaxis_title="Vrijednost",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig
