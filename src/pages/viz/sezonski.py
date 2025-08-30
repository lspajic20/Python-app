# Tab "Sezonski prikaz" – mjesečni prosjeci za odabrani parametar (linijski ili polarni graf)

from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import calendar
import pandas as pd

from src.pages.viz.common import (
    viz_df, COUNTRIES, PARAMS, DEFAULT_COUNTRY, DEFAULT_PARAM, row,
    cities_for_country_in_viz
)


layout = dbc.Container([
    row(
        (dcc.Dropdown(
            id="season_country",
            options=[{"label": c, "value": c} for c in COUNTRIES],
            value=DEFAULT_COUNTRY,
            placeholder="Odaberi državu"
        ), 4),
        (dcc.Dropdown(id="season_city", placeholder="Odaberi grad"), 4),
        (dcc.Dropdown(
            id="season_param",
            options=[{"label": p.upper(), "value": p} for p in PARAMS],
            value=(DEFAULT_PARAM if DEFAULT_PARAM in PARAMS else (PARAMS[0] if PARAMS else None)),
            placeholder="Odaberi parametar"
        ), 4),
    ),
    row(
        (dbc.Checkbox(id="season_polar", label="Polarni prikaz (radar)"), 3),
        (dbc.Checkbox(id="season_band", label="Prikaži percentilni pojas (10–90%)"), 4),
        (dbc.Checkbox(id="season_anom", label="Anomalije (oduzmi srednju vrijednost)"), 5),
    ),
    row(
        (dcc.Dropdown(
            id="season_city_compare",
            placeholder="Usporedi s gradom (opcionalno)"
        ), 6),
    ),
    dcc.Graph(id="seasonal_plot")
], fluid=True)


def register_callbacks(app):
    # Callback za punjenje gradova kad se promijeni država
    @app.callback(
        Output("season_city", "options"),
        Output("season_city", "value"),
        Input("season_country", "value"),
        prevent_initial_call=False,
    )
    def _city_opts(country):
        cities = cities_for_country_in_viz(country)
        return [{"label": c, "value": c} for c in cities], (cities[0] if cities else None)

    # Isti izbor gradova i za "compare" dropdown
    @app.callback(
        Output("season_city_compare", "options"),
        Input("season_country", "value"),
        prevent_initial_call=False,
    )
    def _city_opts_compare(country):
        cities = cities_for_country_in_viz(country)
        return [{"label": c, "value": c} for c in cities]

    # Callback za crtanje sezonskog grafa
    @app.callback(
        Output("seasonal_plot", "figure"),
        Input("season_city", "value"),
        Input("season_param", "value"),
        Input("season_polar", "value"),
        Input("season_band", "value"),
        Input("season_anom", "value"),
        Input("season_city_compare", "value"),
        prevent_initial_call=False
    )
    def _plot(city, param, polar_checked, show_band, anomaly_mode, city_cmp):
        fig = go.Figure()
        if not (city and param):
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            return fig
        if param not in viz_df.columns:
            fig.update_layout(title="Nema podataka za odabrani parametar.",
                              margin=dict(l=10, r=10, t=40, b=10))
            return fig

        def monthly_stats_for_city(city_name: str):
            sub = viz_df[viz_df["grad"].astype(str) == str(city_name)][["datum", param]].dropna()
            if sub.empty:
                return None
            sub["year"] = sub["datum"].dt.year
            sub["month"] = sub["datum"].dt.month

            # prosjek po (godina, mjesec)
            ym = (sub.groupby(["year", "month"])[param]
                      .mean()
                      .reset_index())

            # agregat preko godina -> mean i percentili po mjesecu
            agg = (ym.groupby("month")[param]
                     .agg(mean="mean",
                          p10=lambda s: s.quantile(0.10),
                          p90=lambda s: s.quantile(0.90),
                          n="count")
                     .reindex(range(1, 13))
                     .reset_index())

            # anomaly mode: oduzmi ukupnu srednju vrijednost (12 mj)
            if anomaly_mode:
                mbar = agg["mean"].mean(skipna=True)
                agg["mean"] = agg["mean"] - mbar
                agg["p10"]  = agg["p10"]  - mbar
                agg["p90"]  = agg["p90"]  - mbar

            return agg

        base = monthly_stats_for_city(city)
        if base is None:
            fig.update_layout(title="Nema podataka za odabrane postavke",
                              margin=dict(l=10, r=10, t=40, b=10))
            return fig

        labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        is_polar = bool(polar_checked)

        # Pojas (10–90%) – samo na kartesijskom prikazu
        if not is_polar and show_band:
            fig.add_trace(go.Scatter(
                x=labels, y=base["p90"], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=labels, y=base["p10"], mode="lines",
                line=dict(width=0), fill="tonexty",
                name="10–90%", opacity=0.25
            ))

        # Mean linija za osnovni grad
        if is_polar:
            r = base["mean"].tolist() + [base["mean"].iloc[0]]
            theta = labels + [labels[0]]
            fig.add_trace(go.Scatterpolar(
                r=r, theta=theta, mode="lines+markers",
                name=param.upper()
            ))
        else:
            fig.add_trace(go.Scatter(
                x=labels, y=base["mean"], mode="lines+markers",
                name=f"{city}"
            ))

        # Usporedni grad (samo mean)
        if city_cmp and city_cmp != city:
            cmp = monthly_stats_for_city(city_cmp)
            if cmp is not None:
                if is_polar:
                    r = cmp["mean"].tolist() + [cmp["mean"].iloc[0]]
                    theta = labels + [labels[0]]
                    fig.add_trace(go.Scatterpolar(
                        r=r, theta=theta, mode="lines+markers",
                        name=f"{city_cmp}"
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=labels, y=cmp["mean"], mode="lines+markers",
                        name=f"{city_cmp}"
                    ))

        base_sub = " (anomalije)" if anomaly_mode else ""
        title = f"Sezonski profil – {city} – {param.upper()}{base_sub}"
        if is_polar:
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title=title, margin=dict(l=10, r=10, t=40, b=10)
            )
        else:
            fig.update_layout(
                title=title,
                xaxis_title="Mjesec",
                yaxis_title=("Anomalija" if anomaly_mode else "Prosječna vrijednost"),
                hovermode="x unified",
                margin=dict(l=10, r=10, t=40, b=10)
            )
        return fig
