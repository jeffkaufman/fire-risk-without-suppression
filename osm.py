"""Fetch OpenStreetMap building footprints around a point (Overpass API) into data/osm_<name>.json.
Map data (c) OpenStreetMap contributors, ODbL."""
import json, os, sys, time, requests

ENDPOINTS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]

def ensure(name, lat, lon, radius_m=2500):
    """download the footprints once; returns the file path"""
    path = f"data/osm_{name}.json"
    if os.path.exists(path): return path
    os.makedirs("data", exist_ok=True)
    q = f"""[out:json][timeout:300][maxsize:1073741824];
(way["building"](around:{radius_m},{lat},{lon});
 relation["building"](around:{radius_m},{lat},{lon}););
out geom tags;"""
    err = None
    for attempt in range(3):
        for url in ENDPOINTS:
            try:
                r = requests.post(url, data={"data": q}, timeout=600, headers={"User-Agent": "urban-fire-risk-model"})
                r.raise_for_status()
                d = r.json()
                json.dump(d, open(path, "w"))
                print(f"{name}: {len(d['elements'])} OSM building elements within {radius_m} m", file=sys.stderr)
                return path
            except Exception as e:      # busy public servers are common; try the next one
                err = e; print(f"{url}: {e}", file=sys.stderr); time.sleep(10)
    raise SystemExit(f"could not fetch OSM footprints for {name}: {err}")
