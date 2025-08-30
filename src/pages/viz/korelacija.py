# Matrica korelacija (Pearson / Spearman) za odabrani grad i godinu.

from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from src.pages.viz.common import (
    viz_df, COUNTRIES, PARAMS, DEFAULT_COUNTRY, DEFAULT_PARAM, row,
    cities_for_country_in_viz, years_for_city
)

layout = dbc.Container([
    row(
        (dcc.Dropdown(
            id="corr_country",
            options=[{"label": c, "value": c} for c in COUNTRIES],
            value=DEFAULT_COUNTRY,
            placeholder="Odaberi državu",
        ), 4),
        (dcc.Dropdown(id="corr_city", placeholder="Odaberi grad"), 4),
        (dcc.Dropdown(id="corr_year", placeholder="Odaberi godinu"), 4),
    ),
    row((
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


def register_callbacks(app):
    # Popuni gradove kada se promijeni država
    @app.callback(
        Output("corr_city", "options"),
        Output("corr_city", "value"),
        Input("corr_country", "value"),
        prevent_initial_call=False
    )
    def _corr_city_opts(country):
        cities = cities_for_country_in_viz(country)
        return [{"label": c, "value": c} for c in cities], (cities[0] if cities else None)


    # Popuni godine kada se promijeni grad
    @app.callback(
        Output("corr_year", "options"),
        Output("corr_year", "value"),
        Input("corr_city", "value"),
        prevent_initial_call=False
    )
    def _corr_year_opts(city):
        ys = years_for_city(city)
        return [{"label": str(y), "value": int(y)} for y in ys], (max(ys) if ys else None)

    # Crtanje matrice korelacije
    @app.callback(
        Output("corr_heatmap", "figure"),
        Input("corr_city", "value"),
        Input("corr_year", "value"),
        Input("corr_method", "value"),
        prevent_initial_call=False
    )
    def _plot(city, year, method):
        fig = go.Figure()
        if not (city and year):
            # Nema još odabira -> prazan graf
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            return fig

        # Filtriraj podatke za grad i godinu
        sub = viz_df[
            (viz_df["grad"].astype(str) == str(city)) &
            (viz_df["datum"].dt.year == int(year))
        ].copy()

        if sub.empty:
            fig.update_layout(title="Nema podataka za odabrane postavke",
                              margin=dict(l=10, r=10, t=40, b=10))
            return fig

        numeric = sub.select_dtypes(include="number").copy()
        for drop_col in ("datum",):
            if drop_col in numeric.columns:
                numeric = numeric.drop(columns=[drop_col])

        if "grad" in numeric.columns:
            numeric = numeric.drop(columns=["grad"])

        # Ako je previše praznina, ostavimo par koji ima >=3 zajedničkih točaka
        if numeric.shape[1] < 2:
            fig.update_layout(title="Nedovoljno podataka za korelaciju",
                              margin=dict(l=10, r=10, t=40, b=10))
            return fig

        # Korelacija parova s malim minimumom preklopa
        corr = numeric.corr(method=method, min_periods=3)

        # Ukloni sve-redove/stupce koji su 100% NaN nakon korelacije
        corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")

        # Ako nakon čišćenja imamo <2 varijable, nema smislenog prikaza
        if corr.shape[0] < 2 or corr.shape[1] < 2:
            fig.update_layout(title="Nedovoljno podataka za korelaciju",
                              margin=dict(l=10, r=10, t=40, b=10))
            return fig

        # Tekstualne oznake: prikaži broj (2 decimale), sakrij NaN
        text = corr.round(2).astype(object)
        text = text.where(text.notna(), "")

        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=[c.upper() for c in corr.columns],
            y=[c.upper() for c in corr.index],
            colorscale="RdBu",
            zmin=-1, zmax=1,
            text=text.values,
            texttemplate="%{text}",
            hovertemplate="x: %{x}<br>y: %{y}<br>r: %{z:.2f}<extra></extra>"
        ))

        fig.update_layout(
            title=f"Korelacija – {city} – {year} ({method.capitalize()})",
            margin=dict(l=10, r=10, t=60, b=10)
        )
        return fig
