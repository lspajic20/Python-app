# src/pages/o_aplikaciji.py
from dash import html, dcc
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H2("O aplikaciji", className="mb-4"),

    dcc.Markdown("""
Aplikacija je izrađena u sklopu diplomskog rada i omogućuje **pregled, analizu i vizualizaciju** podataka o kvaliteti zraka.
Kombinira **WAQI API** (trenutna mjerenja) i **povijesne Excel podatke** (agregacije i analize).

    """, className="lead"),

    html.Hr(),

    dbc.Accordion([
        dbc.AccordionItem([
            dcc.Markdown("""
**Što prikazuje?**  
- Trenutni **AQI** za odabrani grad, **dominantni polutant** i **vrijeme zadnjeg ažuriranja**.  
- Lijevo gore birate **državu i grad**, desno je **legenda AQI** s bojama kategorija.

**Kako koristiti?**  
1. Odaberite državu i grad.  
2. Ako je WAQI token postavljen, dohvatit će se aktualni podaci (bojanje kartice prema razini AQI).  
3. U tablici niže nalaze se **IAQI parametri** (npr. PM2.5, PM10, NO₂…), ako su dostupni.

**Napomena:** Ako podaci nisu dostupni za grad ili token nije postavljen, prikazuje se poruka o nemogućnosti dohvaćanja.
            """)
        ], title="Informacije"),

        dbc.AccordionItem([
            dcc.Markdown("""
**Sadržaj pod-kartica:**
- **Usporedba parametara:** linijski graf mjesečnih prosjeka za **više parametara** (max 5) u odabranoj godini.
- **Prosjek po mjesecima:** vodoravni bar-graf prosječnih vrijednosti ključnih polutanata (PM2.5, PM10, NO₂, O₃, SO₂, CO).
- **Sezonski prikaz:** godišnji profil (linijski ili **polarni** prikaz), uz **percentilni pojas (10–90%)**, **anomalije** (oduzimanje srednje vrijednosti) i **usporedbu dvaju gradova**.
- **Godišnji trend:** godišnji prosjeci parametra uz **linearnu trend liniju** i sažetak trenda (nagib i % promjene).
- **Matrica korelacija:** korelacijska matrica (Pearson/Spearman) za odabrani grad i godinu, s **dinamičkim opsegom** i **annotacijom vrijednosti**.

**Savjeti za korištenje:**  
- Usporedba parametara: dodajte/uklonite parametre u multiselect izborniku.  
- Sezonski prikaz: uključite **anomalije** za lakšu usporedbu oblika sezonalnosti između gradova.  
- Korelacije: koristite **Spearman** ako sumnjate na nelinearne odnose ili outliere.
            """)
        ], title="Vizualizacije"),

        dbc.AccordionItem([
            dcc.Markdown("""
**Što prikazuje?**  
- Mjerne stanice **WAQI** pretražene po odabranoj državi (boja markera ~ AQI kategorija).  
- Klik na marker otvara **popup** s nazivom postaje, AQI-jem i vremenom mjerenja.

**Kako koristiti?**  
1. Odaberite državu.  
2. Kliknite **Osvježi** za ponovno dohvaćanje.  
3. Zumirajte/pomaknite kartu prema potrebi.

**Napomena:** U ovoj implementaciji je naglasak na jednostavnosti i brzini (bez klasteriranja); prikaz stanica može biti gušći u područjima s puno mjerenja.
            """)
        ], title="Karta"),

        dbc.AccordionItem([
            dcc.Markdown("""
            **Modeli u aplikaciji (mjesečna frekvencija):**
            - **Random Forest (RF):** koristi cikličke značajke mjeseca (sin/cos), trend i vremenske pomake (lagove).  
            - **Naivni model:** vrijednost iz **istog mjeseca prethodne godine**.  
            - **XGBoost:** gradivni boosting model za nelinearne odnose uz iste značajke (sin/cos, trend, lagovi).  
            - **Prophet:** aditivni model sezonalnosti i trenda (Godina/Sezona), robustan na praznine i pomake.

            **Što dobijete?**  
            - Graf s poviješću, predikcijom i (za RF/XGBoost) **intervalom pouzdanosti (10–90%)**.  
            - **Tablica** s prognozama po mjesecima.  
            - **Evaluacijske metrike** (MAE, RMSE, MAPE) na zadnjih 12 mjeseci (ili kraći test set).

            **Savjeti:**  
            - Kod kratkih ili šumovitih serija, **naivni model** može biti vrlo jak baseline.  
            - **Prophet** je praktičan za izraženu sezonalnost; **XGBoost/RF** su fleksibilni za nelinearnosti.  
            - Za dulje horizonte (npr. 24–36 mj.) intervali nesigurnosti rastu.
                        """)
        ], title="Predviđanja"),

        dbc.AccordionItem([
            dcc.Markdown("""
- **Dash + Plotly** za interaktivno sučelje i grafove.  
- **Pandas + NumPy** za pripremu i agregaciju podataka.  
- **scikit-learn** za Random Forest predviđanja.  
- **dash-leaflet** za prikaz karte postaja.  
- **Flask-Caching** za keširanje WAQI poziva.
            """)
        ], title="Tehnologije"),

        dbc.AccordionItem([
            dcc.Markdown("""
- **WAQI API** (trenutni AQI i postaje): https://aqicn.org  
- **Povijesni Excel** (lokalni dataset): *data/podaciv2.xlsx*  
- **Tablica gradova/država**: *data/gradovi_drzave.xlsx*
            """)
        ], title="Izvori podataka"),

        dbc.AccordionItem([
            dcc.Markdown("""
**Autor:** Studentica Fakulteta organizacije i informatike, 2025.  
**Kod i dokumentacija:**  
- GitHub repo: [lspajic20/PY](https://github.com/lspajic20/PY)

            """)
        ], title="Autor i repozitorij"),
    ], start_collapsed=True, always_open=True),

    html.Hr(),
], fluid=True)
