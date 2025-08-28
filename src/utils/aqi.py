import requests

# Funkcija koja na temelju vrijednosti AQI-ja vraća boju za prikaz
def aqi_color(aqi):
    try:
        aqi = float(aqi)
    except (TypeError, ValueError):
        return "#999999" 
    if aqi <= 50:
        return "#aedcae"
    if aqi <= 100:
        return "#ffea61"
    if aqi <= 150:
        return "#ff9100"
    if aqi <= 200:
        return "#ff4500"
    if aqi <= 300:
        return "#9f5ea5"
    return "#7e0023"

# Funkcija koja dohvaća podatke o kvaliteti zraka za odabrani grad putem WAQI API-ja
def get_city_data(city: str, token: str):
    if not city or not token:
        return None
    url = f"https://api.waqi.info/feed/{requests.utils.quote(city)}/?token={token}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        js = r.json()
        if js.get("status") != "ok":
            print("WAQI response:", js) 
            return None
        return js.get("data")
    except Exception as e:
        print("Error fetching:", e)
        return None
