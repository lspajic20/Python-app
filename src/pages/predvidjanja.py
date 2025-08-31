# Tab "Predviđanja" – mjesečna predikcija (RF / XGBoost / Prophet / Naivni) + tablica

from __future__ import annotations
from dash import dcc, html, Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import numpy as np
import pandas as pd

# --- Optional ML libs ---
try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:
    RandomForestRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from prophet import Prophet
except Exception:
    Prophet = None

# Zajednički utilsi/podaci
from src.pages.viz.common import (
    viz_df, COUNTRIES, PARAMS, DEFAULT_COUNTRY, DEFAULT_PARAM, row,
    cities_for_country_in_viz, years_for_city
)

# Pomoćne funkcije

def _month_cyc_features(s_month: pd.Series) -> pd.DataFrame:
    """Cikličke značajke mjeseca (sin/cos) – pomažu modelima uhvatiti sezonalnost."""
    rad = 2 * np.pi * (s_month.astype(int) - 1) / 12.0
    return pd.DataFrame({"m_sin": np.sin(rad), "m_cos": np.cos(rad)})

def _build_supervised(df: pd.DataFrame, y_col: str, n_lags: int = 3) -> pd.DataFrame:
    """
    Od mjesečnih srednjaka radi 'supervised' skup.
    y = vrijednost parametra
    Značajke: mjesec (sin/cos), trend (brojač), lagovi y(t-1..t-n)
    """
    out = df.copy()
    out["year"] = out["month"].dt.year
    out["m"] = out["month"].dt.month
    cyc = _month_cyc_features(out["m"])
    out = pd.concat([out, cyc], axis=1)
    out["trend"] = np.arange(len(out))  
    
    for k in range(1, n_lags + 1):
        out[f"lag{k}"] = out[y_col].shift(k)

    out = out.dropna().reset_index(drop=True)
    return out

def _monthly_avg_for_city_param(city: str, param: str) -> pd.DataFrame:
    #Mjesecni prosjeci (month, y) za odabrani grad/parametar, mjesec-na-pocetak
    if param not in viz_df.columns:
        return pd.DataFrame(columns=["month", "y"])
    sub = viz_df[viz_df["grad"].astype(str) == str(city)][["datum", param]].dropna()
    if sub.empty:
        return pd.DataFrame(columns=["month", "y"])

    # mjesec poravnan na početak
    sub["month"] = pd.to_datetime(sub["datum"]).dt.to_period("M").dt.to_timestamp(how="start")
    g = (sub.groupby("month")[param].mean().reset_index().sort_values("month"))
    g = g.rename(columns={param: "y"})
    return g[["month", "y"]]

def _baseline_last_year_same_month(history: pd.DataFrame, horizon: int) -> pd.Series:
    # Naivni model uzima vrijednost istog mjeseca prošle godine
    if history.empty:
        return pd.Series([np.nan] * horizon)

    last_m = pd.to_datetime(history["month"].iloc[-1])
    start = (last_m + pd.offsets.MonthBegin(1)).normalize()
    future_months = pd.date_range(start=start, periods=horizon, freq="MS")
    hist = history.set_index("month")["y"].copy()

    preds = []
    for m in future_months:
        prev = m - pd.DateOffset(years=1)
        val = hist.get(prev, hist.iloc[-1] if len(hist) else np.nan)
        preds.append(float(val))
    return pd.Series(preds, index=future_months)

def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    # Izračun MAE / RMSE / MAPE (robustno uz NaN)
    x = pd.concat([y_true, y_pred], axis=1).dropna()
    if x.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
    e = x.iloc[:, 0] - x.iloc[:, 1]
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    eps = 1e-6  # štiti od dijeljenja nulom
    mape = float(np.mean(np.abs(e) / (np.abs(x.iloc[:, 0]) + eps)) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# Random Forest 
def _rf_forecast_monthly(history: pd.DataFrame, horizon: int, n_lags: int = 3,
                         n_estimators: int = 400, random_state: int = 42):
    if history.empty or RandomForestRegressor is None:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    df = history.copy()
    df_sup = _build_supervised(df, "y", n_lags=n_lags)
    if df_sup.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    X_cols = ["m_sin", "m_cos", "trend"] + [f"lag{k}" for k in range(1, n_lags + 1)]
    X = df_sup[X_cols].to_numpy()
    y = df_sup["y"].to_numpy()
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X, y)

    last_m = pd.to_datetime(df["month"].iloc[-1])
    start = (last_m + pd.offsets.MonthBegin(1)).normalize()
    future_months = pd.date_range(start=start, periods=horizon, freq="MS")

    y_hist = df["y"].tolist()
    preds_mean, preds_low, preds_high = [], [], []

    for step, m in enumerate(future_months, start=1):
        m_num = m.month
        cyc = _month_cyc_features(pd.Series([m_num]))
        trend = len(df) + (step - 1)

        last_vals = (y_hist + preds_mean)[-n_lags:] if preds_mean else y_hist[-n_lags:]
        if len(last_vals) < n_lags:
            last_vals = [y_hist[-1]] * (n_lags - len(last_vals)) + last_vals

        x_row = {"m_sin": cyc["m_sin"].iloc[0], "m_cos": cyc["m_cos"].iloc[0], "trend": trend}
        for k in range(1, n_lags + 1):
            x_row[f"lag{k}"] = last_vals[-k]

        x_arr = np.array([[x_row[col] for col in X_cols]])
        tree_preds = np.array([t.predict(x_arr)[0] for t in model.estimators_])

        preds_mean.append(float(tree_preds.mean()))
        preds_low.append(float(np.quantile(tree_preds, 0.10)))
        preds_high.append(float(np.quantile(tree_preds, 0.90)))

    s_mean = pd.Series(preds_mean, index=future_months)
    s_low = pd.Series(preds_low, index=future_months)
    s_high = pd.Series(preds_high, index=future_months)
    return s_mean, s_low, s_high

# XGBoost 
def _xgb_forecast_monthly(history: pd.DataFrame, horizon: int, n_lags: int = 3,
                          n_estimators: int = 600, learning_rate: float = 0.05,
                          max_depth: int = 4, subsample: float = 0.9,
                          colsample_bytree: float = 0.8, random_state: int = 42):
    # Gradient boosting na istim značajkama kao RF
    if history.empty or XGBRegressor is None:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    df = history.copy()
    df_sup = _build_supervised(df, "y", n_lags=n_lags)
    if df_sup.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    X_cols = ["m_sin", "m_cos", "trend"] + [f"lag{k}" for k in range(1, n_lags + 1)]
    X = df_sup[X_cols].to_numpy()
    y = df_sup["y"].to_numpy()

    # Više slabih modela za intervale (bagging-ish)
    seeds = [random_state + i for i in range(10)]
    models = []
    for sd in seeds:
        model = XGBRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            subsample=subsample, colsample_bytree=colsample_bytree, random_state=sd,
            objective="reg:squarederror", n_jobs=-1
        )
        model.fit(X, y)
        models.append(model)

    last_m = pd.to_datetime(df["month"].iloc[-1])
    start = (last_m + pd.offsets.MonthBegin(1)).normalize()
    future_months = pd.date_range(start=start, periods=horizon, freq="MS")

    y_hist = df["y"].tolist()
    preds_mean, preds_low, preds_high = [], [], []

    for step, m in enumerate(future_months, start=1):
        m_num = m.month
        cyc = _month_cyc_features(pd.Series([m_num]))
        trend = len(df) + (step - 1)

        last_vals = (y_hist + preds_mean)[-n_lags:] if preds_mean else y_hist[-n_lags:]
        if len(last_vals) < n_lags:
            last_vals = [y_hist[-1]] * (n_lags - len(last_vals)) + last_vals

        x_row = {"m_sin": cyc["m_sin"].iloc[0], "m_cos": cyc["m_cos"].iloc[0], "trend": trend}
        for k in range(1, n_lags + 1):
            x_row[f"lag{k}"] = last_vals[-k]

        x_arr = np.array([[x_row[col] for col in X_cols]])
        preds = np.array([m_.predict(x_arr)[0] for m_ in models])

        preds_mean.append(float(preds.mean()))
        preds_low.append(float(np.quantile(preds, 0.10)))
        preds_high.append(float(np.quantile(preds, 0.90)))

    s_mean = pd.Series(preds_mean, index=future_months)
    s_low = pd.Series(preds_low, index=future_months)
    s_high = pd.Series(preds_high, index=future_months)
    return s_mean, s_low, s_high

# Prophet 
def _prophet_forecast_monthly(history: pd.DataFrame, horizon: int, interval_width: float = 0.9):
    if history.empty or Prophet is None:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    df = history.copy()
    df_p = df.rename(columns={"month": "ds", "y": "y"}).copy()
    # Prophet će raditi s mjesečnim frekvencijama (MS)
    m = Prophet(
        growth="linear", yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
        seasonality_mode="additive", interval_width=interval_width
    )
    m.fit(df_p)

    last_m = pd.to_datetime(df["month"].iloc[-1])
    start = (last_m + pd.offsets.MonthBegin(1)).normalize()
    future = m.make_future_dataframe(periods=horizon, freq="MS", include_history=False)
    future["ds"] = pd.date_range(start=start, periods=horizon, freq="MS")

    fc = m.predict(future)
    idx = pd.to_datetime(fc["ds"])
    s_mean = pd.Series(fc["yhat"].astype(float).values, index=idx)
    s_low  = pd.Series(fc["yhat_lower"].astype(float).values, index=idx)
    s_high = pd.Series(fc["yhat_upper"].astype(float).values, index=idx)
    return s_mean, s_low, s_high

# Layout

layout = dbc.Container([
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id="fc_country",
            options=[{"label": c, "value": c} for c in COUNTRIES],
            value=DEFAULT_COUNTRY,
            placeholder="Odaberi državu"
        ), md=4),
        dbc.Col(dcc.Dropdown(id="fc_city", placeholder="Odaberi grad"), md=4),
        dbc.Col(dcc.Dropdown(
            id="fc_param",
            options=[{"label": p.upper(), "value": p} for p in PARAMS],
            value=(DEFAULT_PARAM if DEFAULT_PARAM in PARAMS else PARAMS[0]),
            placeholder="Odaberi parametar"
        ), md=4),
    ], className="g-2"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id="fc_model",
            options=[
                {"label": "Random Forest", "value": "rf"},
                {"label": "Naivni (isti mjesec prošle godine)", "value": "naive"},
                {"label": "XGBoost", "value": "xgb"},
                {"label": "Prophet", "value": "prophet"},
            ],
            value="rf",
            clearable=False
        ), md=4),
        dbc.Col(dcc.Slider(
            id="fc_horizon",
            min=3, max=36, step=1, value=12,
            marks={i: str(i) for i in [3, 6, 9, 12, 18, 24, 30, 36]}
        ), md=8),
    ], className="g-3 mt-2"),
    
    html.Div(id="fc_warning", className="mt-2"),
    dcc.Graph(id="fc_plot", style={"height": "68vh"}),

    dash_table.DataTable(
        id="fc_table",
        columns=[
            {"name": "Mjesec", "id": "month"},
            {"name": "Prognoza", "id": "forecast"},
            {"name": "Donja (10%)", "id": "low"},
            {"name": "Gornja (90%)", "id": "high"},
        ],
        data=[],
        style_table={"overflowX": "auto"},
        style_cell={"padding": "6px"},
        style_header={"fontWeight": "600"},
        page_action="none"
    ),

    html.Div(id="fc_metrics", className="mt-3")
], fluid=True)

# Callbacks

def register_callbacks(app):
    # Gradovi po državi
    @app.callback(
        Output("fc_city", "options"),
        Output("fc_city", "value"),
        Input("fc_country", "value"),
        prevent_initial_call=False
    )
    def _city_opts(country):
        cities = cities_for_country_in_viz(country)
        opts = [{"label": c, "value": c} for c in cities]
        return opts, (cities[0] if cities else None)

    # Treniraj + predvidi
    @app.callback(
        Output("fc_plot", "figure"),
        Output("fc_table", "columns"),
        Output("fc_table", "data"),
        Output("fc_metrics", "children"),
        Output("fc_warning", "children"),
        Input("fc_city", "value"),
        Input("fc_param", "value"),
        Input("fc_model", "value"),
        Input("fc_horizon", "value"),
        prevent_initial_call=False
    )
    def _forecast(city, param, model_name, horizon):
        fig = go.Figure()
        empty_cols = [
            {"name": "Mjesec", "id": "month"},
            {"name": "Prognoza", "id": "forecast"},
            {"name": "Donja (10%)", "id": "low"},
            {"name": "Gornja (90%)", "id": "high"},
        ]
        empty_data = []

        # Provjere i upozorenja za nedostajuće biblioteke
        missing = None
        if model_name == "rf" and RandomForestRegressor is None:
            missing = "scikit-learn"
        if model_name == "xgb" and XGBRegressor is None:
            missing = "xgboost"
        if model_name == "prophet" and Prophet is None:
            missing = "prophet"

        if missing is not None:
            warn = dbc.Alert(f"Model '{model_name}' zahtijeva paket: pip install {missing}", color="warning")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            return fig, empty_cols, empty_data, "", warn

        if not (city and param and horizon):
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            return fig, empty_cols, empty_data, "", None

        hist = _monthly_avg_for_city_param(city, param)
        if hist.empty:
            fig.update_layout(title="Nema povijesnih podataka za odabrani grad/parametar.",
                              margin=dict(l=10, r=10, t=40, b=10))
            return fig, empty_cols, empty_data, "", None

        # Split za metrike (zadnjih 12 mjeseci -> test)
        test_len = min(12, len(hist) // 3)
        train_for_metrics = hist.iloc[:-test_len] if test_len > 0 else hist.copy()
        test = hist.iloc[-test_len:] if test_len > 0 else pd.DataFrame(columns=hist.columns)

        # Future forecast (na cijrlu povijesti), plus forecast za metrike (na train)
        def _forecast_by_model(history, model):
            if model == "rf":
                return _rf_forecast_monthly(history, horizon=horizon, n_lags=3)
            elif model == "xgb":
                return _xgb_forecast_monthly(history, horizon=horizon, n_lags=3)
            elif model == "prophet":
                return _prophet_forecast_monthly(history, horizon=horizon, interval_width=0.9)
            else:  
                s = _baseline_last_year_same_month(history, horizon=horizon)
                return s, None, None

        fc_mean, fc_low, fc_high = _forecast_by_model(hist, model_name)

        # Forecast za test metrike
        metrics_rf = {}
        if test_len > 0:
            if model_name in ("rf", "xgb", "prophet"):
                test_pred, _, _ = _forecast_by_model(train_for_metrics, model_name)
                metrics_rf = _metrics(test.set_index("month")["y"], test_pred)
            else:
                naive_test = _baseline_last_year_same_month(train_for_metrics, horizon=test_len)
                metrics_rf = _metrics(test.set_index("month")["y"], naive_test)

        #Tablica prognoze
        if model_name in ("rf", "xgb", "prophet") and fc_low is not None and fc_high is not None:
            low_vals = fc_low.reindex(fc_mean.index).values
            high_vals = fc_high.reindex(fc_mean.index).values
        else:
            low_vals = [np.nan] * len(fc_mean)
            high_vals = [np.nan] * len(fc_mean)

        table_df = pd.DataFrame({
            "month": [m.strftime("%Y-%m") for m in fc_mean.index],
            "forecast": [float(v) if pd.notna(v) else np.nan for v in fc_mean.values],
            "low": [float(v) if pd.notna(v) else np.nan for v in low_vals],
            "high": [float(v) if pd.notna(v) else np.nan for v in high_vals],
        })
        table_cols = [
            {"name": "Mjesec", "id": "month"},
            {"name": "Prognoza", "id": "forecast", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "Donja (10%)", "id": "low", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "Gornja (90%)", "id": "high", "type": "numeric", "format": {"specifier": ".2f"}},
        ]
        table_data = table_df.to_dict("records")

        # Graf
        fig.add_trace(go.Scatter(
            x=hist["month"], y=hist["y"],
            mode="lines+markers", name="Povijest"
        ))

        if model_name in ("rf", "xgb", "prophet") and fc_low is not None and fc_high is not None and len(fc_low) == len(fc_mean):
            fig.add_trace(go.Scatter(
                x=list(fc_mean.index) + list(fc_mean.index[::-1]),
                y=list(fc_high.values) + list(fc_low.values[::-1]),
                fill="toself", fillcolor="rgba(99,110,250,0.15)",
                line=dict(width=0), hoverinfo="skip",
                name="90% interval"
            ))

        name_map = {
            "rf": "RF prognoza",
            "xgb": "XGB prognoza",
            "prophet": "Prophet prognoza",
            "naive": "Naivna prognoza",
        }
        fig.add_trace(go.Scatter(
            x=fc_mean.index, y=fc_mean.values,
            mode="lines+markers", name=name_map.get(model_name, "Prognoza")
        ))

        fig.update_layout(
            title=f"Predviđanje – {city} – {param.upper()}",
            xaxis_title="Mjesec",
            yaxis_title="Vrijednost",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=50, b=10)
        )

        # Metrike
        if metrics_rf:
            mt = html.Div([
                html.H6("Evaluacija na zadnjih 12 mjeseci"),
                html.Div(f"MAE: {metrics_rf['MAE']:.3f}"),
                html.Div(f"RMSE: {metrics_rf['RMSE']:.3f}"),
                html.Div(f"MAPE: {metrics_rf['MAPE']:.2f}%")
            ], className="text-muted")
        else:
            mt = html.Div("", className="text-muted")

        return fig, table_cols, table_data, mt, None
