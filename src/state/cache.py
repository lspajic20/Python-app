from flask_caching import Cache

cache: Cache | None = None   # Globalna varijabla za cache

# Funkcija za inicijalizaciju cache sustava
def init_cache(server, config: dict | None = None):
    global cache
    cfg = {
        "CACHE_TYPE": "SimpleCache",
        "CACHE_DEFAULT_TIMEOUT": 300
    }
    if config:
        cfg.update(config)
    cache = Cache(server, config=cfg)
    return cache
