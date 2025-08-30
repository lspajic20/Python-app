# Tab "Godišnji trend" – godišnji prosjek odabranog parametra + linearni trend + interativni raspon godina +  95% CI + R²/Pearsonov koeficijent

from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np

from src.pages.viz.common import (
    viz_df, COUNTRIES, PARAMS, DEFAULT_COUNTRY, DEFAULT_PARAM, row,
    cities_for_country_in_viz
)

# Pomoćna funkcija za postavljanje RangeSlidera
def _range_slider_setup(years):
    if not years:
        years = [2020, 2021]
    mn, mx = int(min(years)), int(max(years))
    if mx - mn > 20:
        step = max(1, (mx - mn) // 20)
        marks = {y: str(y) for y in range(mn, mx + 1, step)}
    else:
        marks = {y: str(y) for y in range(mn, mx + 1)}
    return mn, mx, marks, [mn, mx]

_all_years = viz_df["datum"].dropna().dt.year.astype(int)
_init_min, _init_max, _init_marks, _init_val = _range_slider_setup(_all_years.unique().tolist())

layout = dbc.Container([
    row(
        (dcc.Dropdown(
            id="year_trend_country",
            options=[{"label": c, "value": c} for c in COUNTRIES],
            value=DEFAULT_COUNTRY,
            placeholder="Odaberi državu"
        ), 4),
        (dcc.Dropdown(id="year_trend_city", placeholder="Odaberi grad"), 4),
        (dcc.Dropdown(
            id="year_trend_parameter",
            options=[{"label": p.upper(), "value": p} for p in PARAMS],
            value=(DEFAULT_PARAM if DEFAULT_PARAM in PARAMS else (PARAMS[0] if PARAMS else None)),
            placeholder="Odaberi parametar"
        ), 4),
    ),

    # range slider za odabir godina uvijek inicijaliziran na min/max iz cijelog skupa podataka
    dcc.RangeSlider(
        id="year_trend_range",
        min=_init_min, max=_init_max, value=_init_val, marks=_init_marks,
        allowCross=False, tooltip={"placement": "bottom", "always_visible": False},
        className="mt-2"
    ),

    dbc.Checkbox(id="year_trend_show_ci", label="Prikaži 95% CI za trend", value=True, className="mt-2"),
    dcc.Graph(id="year_trend_plot", className="mt-2")
], fluid=True)


def register_callbacks(app):
    # Callback za punjenje gradova kad se promijeni država
    @app.callback(
        Output("year_trend_city", "options"),
        Output("year_trend_city", "value"),
        Input("year_trend_country", "value"),
        prevent_initial_call=False
    )
    def _city_opts(country):
        cities = cities_for_country_in_viz(country)
        return [{"label": c, "value": c} for c in cities], (cities[0] if cities else None)

    # Ažuriranje raspona godina kad se promijeni grad
    @app.callback(
        Output("year_trend_range", "min"),
        Output("year_trend_range", "max"),
        Output("year_trend_range", "marks"),
        Output("year_trend_range", "value"),
        Input("year_trend_city", "value"),
        prevent_initial_call=False
    )
    def _update_range(city):
        if not city:
            return _init_min, _init_max, _init_marks, _init_val
        sub = viz_df[viz_df["grad"].astype(str) == str(city)][["datum"]].dropna()
        if sub.empty:
            return _init_min, _init_max, _init_marks, _init_val
        years = sub["datum"].dt.year.dropna().astype(int).unique().tolist()
        mn, mx, marks, val = _range_slider_setup(years)
        return mn, mx, marks, val

    # Graf sa trendom, postotnom promjenom, R² i Pearsonovim koeficijentom
    @app.callback(
        Output("year_trend_plot", "figure"),
        Input("year_trend_city", "value"),
        Input("year_trend_parameter", "value"),
        Input("year_trend_range", "value"),
        Input("year_trend_show_ci", "value"),
        prevent_initial_call=False
    )
    def _plot(city, param, year_range, show_ci):
        fig = go.Figure()

        if not (city and param):
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            return fig
        if param not in viz_df.columns:
            fig.update_layout(title="Nema podataka za odabrani parametar.",
                              margin=dict(l=10, r=10, t=40, b=10))
            return fig

        sub = viz_df[viz_df["grad"].astype(str) == str(city)][["datum", param]].dropna().copy()
        if sub.empty:
            fig.update_layout(title="Nema podataka za odabrane postavke",
                              margin=dict(l=10, r=10, t=40, b=10))
            return fig

        sub["year"] = sub["datum"].dt.year.astype(int)
        yearly = (sub.groupby("year")[param].mean().reset_index().sort_values("year"))

        # Filtriraj prema odabranom rasponu godina
        if year_range and isinstance(year_range, (list, tuple)) and len(year_range) == 2:
            lo, hi = int(year_range[0]), int(year_range[1])
            yearly = yearly[(yearly["year"] >= lo) & (yearly["year"] <= hi)]

        if len(yearly) < 2:
            fig.update_layout(title="Nedovoljno godina za prikaz trenda.",
                              margin=dict(l=10, r=10, t=40, b=10))
            return fig

        years_int = yearly["year"].astype(int).tolist()
        values = yearly[param].astype(float).tolist()

        x = np.array(years_int, dtype=float)
        y = np.array(values, dtype=float)
        a, b = np.polyfit(x, y, 1)

        xgrid = np.arange(int(x.min()), int(x.max()) + 1)
        yhat = a * xgrid + b

        mean_level = float(np.nanmean(y)) if np.isfinite(y).any() else 0.0
        pct_change = (100.0 * (yhat[-1] - yhat[0]) / mean_level) if mean_level else 0.0

        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denom = np.sqrt((x_centered**2).sum() * (y_centered**2).sum())
        r = float((x_centered * y_centered).sum() / denom) if denom else 0.0
        r2 = r * r
        subtitle = f"Trend: {a:.2f} / god  ({pct_change:.1f}%)  |  R²: {r2:.2f}  (r={r:.2f})"

        fig.add_trace(go.Scatter(x=years_int, y=values, mode="lines+markers", name="Godišnji prosjek"))
        fig.add_trace(go.Scatter(x=xgrid.tolist(), y=yhat.tolist(), mode="lines", name="Trend", line=dict(dash="dash")))

        if show_ci:
            n = len(x)
            if n >= 3:
                yhat_obs = a * x + b
                residuals = y - yhat_obs
                s2 = float((residuals**2).sum() / (n - 2)) if n > 2 else 0.0
                s = np.sqrt(s2)
                xbar = x.mean()
                sxx = ((x - xbar)**2).sum() if n > 1 else 0.0
                if sxx > 0 and s > 0:
                    se_mean = s * np.sqrt(1.0 / n + ((xgrid - xbar) ** 2) / sxx)
                    ci = 1.96 * se_mean
                    upper = (yhat + ci).tolist()
                    lower = (yhat - ci).tolist()
                    fig.add_trace(go.Scatter(x=xgrid.tolist(), y=upper, mode="lines", line=dict(width=0),
                                             hoverinfo="skip", showlegend=False))
                    fig.add_trace(go.Scatter(x=xgrid.tolist(), y=lower, mode="lines", line=dict(width=0),
                                             fill="tonexty", name="95% CI", opacity=0.25))

        fig.update_layout(
            title=f"Godišnji trend – {city} – {param.upper()}<br><sup>{subtitle}</sup>",
            xaxis=dict(
                title="Godina",
                tickmode="array",
                tickvals=years_int,
                ticktext=[str(y) for y in years_int]
            ),
            yaxis_title="Prosječna vrijednost",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=70, b=10)
        )
        return fig
