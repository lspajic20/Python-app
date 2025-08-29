# Tab "Prosjek po mjesecima" – horizontalni bar graf s prosječnim vrijednostima zagađivača

from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from src.pages.viz.common import (
    viz_df, COUNTRIES, DEFAULT_COUNTRY, row,
    cities_for_country_in_viz, years_for_city, CORE_PARAMS
)

# Layout
layout = dbc.Container([
    row(
        (dcc.Dropdown(
            id="avg_country",
            options=[{"label": c, "value": c} for c in COUNTRIES],
            value=DEFAULT_COUNTRY,
            placeholder="Odaberi državu"
        ), 4),
        (dcc.Dropdown(id="avg_city", placeholder="Odaberi grad"), 4),
        (dcc.Dropdown(id="avg_year", placeholder="Odaberi godinu"), 4),
    ),
    dcc.Graph(id="avg_pollutants_plot")
], fluid=True)

def register_callbacks(app):
    # Gradovi ovisno o državi
    @app.callback(
        Output("avg_city", "options"),
        Output("avg_city", "value"),
        Input("avg_country", "value"),
        prevent_initial_call=False
    )
    def _avg_city_opts(country):
        cities = cities_for_country_in_viz(country)
        return [{"label": c, "value": c} for c in cities], (cities[0] if cities else None)

    # Godine ovisno o gradu
    @app.callback(
        Output("avg_year", "options"),
        Output("avg_year", "value"),
        Input("avg_city", "value"),
        prevent_initial_call=False
    )
    def _years_opts(city):
        ys = years_for_city(city)
        return [{"label": str(y), "value": int(y)} for y in ys], (max(ys) if ys else None)

    # Crtanje grafa prosjeka
    @app.callback(
        Output("avg_pollutants_plot", "figure"),
        Input("avg_city", "value"),
        Input("avg_year", "value"),
        prevent_initial_call=False
    )
    def _plot(city, year):
        fig = go.Figure()
        if not (city and year):
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            return fig

        sub = viz_df[(viz_df["grad"].astype(str) == str(city)) &
                     (viz_df["datum"].dt.year == int(year))].copy()

        if sub.empty:
            fig.update_layout(title="Nema podataka za odabrane postavke",
                              margin=dict(l=10, r=10, t=40, b=10))
            return fig

        vals = []
        for p in CORE_PARAMS:
            if p in sub.columns:
                v = float(sub[p].mean(skipna=True))
                if v == v:
                    vals.append((p, v))

        if not vals:
            fig.update_layout(title="Nema dostupnih polutanata za prikaz.",
                              margin=dict(l=10, r=10, t=40, b=10))
            return fig

        vals.sort(key=lambda x: x[1], reverse=True)
        labels = [p.upper() for p, _ in vals]
        values = [v for _, v in vals]

        fig.add_trace(go.Bar(x=values, y=labels, orientation="h", name="Prosjek"))
        fig.update_layout(
            title=f"Prosječne vrijednosti onečišćivača – {city} – {year}",
            xaxis_title="Prosječna vrijednost",
            yaxis_title="Parametar",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig
