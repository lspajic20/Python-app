from dotenv import load_dotenv
load_dotenv()   # Učitavanje .env datoteke - pristup API tokenu

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from src.state.cache import init_cache          
from src.pages import informacije, vizualizacije, karta, predvidjanja, oaplikaciji

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
    id="sidebar",
    vertical=True, pills=True
    )

# Top bar s logom i nazivom aplikacije
topbar = dbc.Navbar(
    dbc.Container(
        [
            html.Img(src="/assets/logoo.png", height="40", className="me-2"),
            dbc.NavbarBrand("AQ Explorer", className="fw-semibold"),
        ],
        fluid=True,
    ),
    color="#acdce1",
    dark=True,
    className="mb-2",
)



# Glavni izgled aplikacije: gornji navbar + sidebar s lijeve strane + sadržaj s desne
app.layout = dbc.Container(
    [
        dcc.Location(id="url"),
        dcc.Store(id="sidebar_open", data=True), 

        topbar,

        dbc.Row(
            [
                dbc.Col(sidebar, id="sidebar_col", md=3, className="pt-2"),
                dbc.Col(html.Div(id="page_content"), id="content_col", md=9, className="pt-2"),
            ],
            className="g-2",
        ),
    ],
    fluid=True,
)

# Callback za rutiranje-prikazuje odgovarajući sadržaj ovisno o URL-u
@app.callback(Output("page_content", "children"), Input("url", "pathname"))
def route(path):
    if path in ("/", "/informacije"):
        return informacije.layout
    if path == "/vizualizacije":
        return vizualizacije.layout
    if path == "/karta":
        return karta.layout
    if path == "/predvidjanja":
        return predvidjanja.layout
    if path == "/oaplikaciji":
        return oaplikaciji.layout
    return html.Div("404")

@app.callback(
    Output("sidebar_open", "data"),
    Input("logo_btn", "n_clicks"),
    State("sidebar_open", "data"),
    prevent_initial_call=True,
)
def _toggle_sidebar(n, is_open):
    return not bool(is_open)

# --- Apply the open/closed state to layout ---
@app.callback(
    Output("sidebar_col", "style"),
    Output("content_col", "className"),
    Input("sidebar_open", "data"),
)

    

# Registracija callbackova sa stranice "Informacije"
informacije.register_callbacks(app)
vizualizacije.register_callbacks(app)
karta.register_callbacks(app)
predvidjanja.register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True) 
