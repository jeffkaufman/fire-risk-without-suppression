"""Download LA County 2023 building footprints joined to CAL FIRE damage inspections (DINS) for the 2025
Eaton and Palisades fires (about 45k buildings, 33 MB) into data/la/buildings_dins.geojson."""
import json, os, requests
B = "https://services.arcgis.com/RmCCgQtiZLDCtblq/ArcGIS/rest/services/Buildings_2023_with_DINS/FeatureServer/4/query"
os.makedirs("data/la", exist_ok=True)
feats = []; off = 0
while True:
    r = requests.get(B, params=dict(where="1=1", outFields="OBJECTID,HEIGHT,AREA,DAMAGE,STRUCTURET,CODE", outSR=4326, f="geojson",
                                    resultOffset=off, resultRecordCount=2000, orderByFields="OBJECTID"),
                     timeout=120, headers={"User-Agent": "urban-fire-risk-model"})
    fs = r.json()["features"]; feats += fs; off += len(fs)
    if len(fs) == 0: break
json.dump({"type": "FeatureCollection", "features": feats}, open("data/la/buildings_dins.geojson", "w"))
print(len(feats), "buildings")
