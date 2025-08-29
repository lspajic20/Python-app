# Glavna stranica za vizualizacije – prikazuje tabove i povezuje pojedine module

from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from src.pages.viz import usporedba, prosjek, sezonski, godisnji, korelacija

# Layout kartice "Vizualizacije" s tabovima
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
    # Prebacivanje sadržaja ovisno o odabranom tabu
    @app.callback(Output("viz_tabs_content", "children"), Input("viz_tabs", "value"))
    def _render_tab(tab):
        from src.pages.viz import usporedba, prosjek, sezonski, godisnji, korelacija
        return {
            "usporedba": usporedba.layout,
            "prosjek": prosjek.layout,
            "sezonski": sezonski.layout,
            "godisnji": godisnji.layout,
            "korelacija": korelacija.layout,
        }.get(tab, html.Div("Nepoznata kartica"))

    # Registracija callbackova iz svakog podmodula
    for mod in (usporedba, prosjek, sezonski, godisnji, korelacija):
        if hasattr(mod, "register_callbacks"):
            mod.register_callbacks(app)