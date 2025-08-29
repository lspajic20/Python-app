from dotenv import load_dotenv
load_dotenv()   # Učitavanje .env datoteke - pristup API tokenu

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

from src.state.cache import init_cache          
from src.pages import informacije, vizualizacije

# Kreiranje glavne Dash aplikacije uz Bootstrap temu
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "AQ Explorer"

# Inicijalizacija server-side cachea 
init_cache(app.server)

# Navigacijska traka (sidebar) 
sidebar = dbc.Nav(
    [
        dbc.NavLink("Informacije", href="/informacije", active="exact"),
        dbc.NavLink("Vizualizacije", href="/vizualizacije", active="exact"),
        dbc.NavLink("Karta", href="/karta", active="exact"),
        dbc.NavLink("Predviđanja", href="/predvidjanja", active="exact"),
        dbc.NavLink("O aplikaciji", href="/oaplikaciji", active="exact"),
    ],
    vertical=True, pills=True, className="bg-light p-2 rounded-3"
)

# Glavni izgled aplikacije: gornji navbar + sidebar s lijeve strane + sadržaj s desne
app.layout = dbc.Container([
    dcc.Location(id="url"),  # Za praćenje URL-a
    dbc.NavbarSimple(brand="AQ Explorer", color="primary", dark=True),
    dbc.Row([dbc.Col(sidebar, md=3), dbc.Col(html.Div(id="page_content"), md=9)], className="mt-3"),
], fluid=True)

# Callback za rutiranje-prikazuje odgovarajući sadržaj ovisno o URL-u
@app.callback(Output("page_content", "children"), Input("url", "pathname"))
def route(path):
    if path in ("/", "/informacije"):
        return informacije.layout
    elif path == "/vizualizacije":
        return vizualizacije.layout
    return html.Div("404")

# Registracija callbackova sa stranice "Informacije"
informacije.register_callbacks(app)
vizualizacije.register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True) 
