"""Weather day types from hourly airport observations (Iowa Environmental Mesonet ASOS archive).

Per day:
  W4   = 4th-highest hourly sustained wind, km/h (so at least 4 hours at or above it)
  RH4  = mean relative humidity in those 4 hours
  dry  = under 0.05 in of precipitation that day and the day before
Day types:
  red-flag    : dry, RH4 <= 50, W4 >= 40      (the 1973 Chelsea MA conflagration day at Boston Logan: W4 = 44, RH 25%)
  windy & dry : dry, RH4 <= 50, 32 <= W4 < 40
  calm/damp   : rain that day or the day before
  ordinary    : everything else
Snow cover and longer-term drought are not accounted for.

    python weather.py BOS          # fetch if needed, print a summary, update data/weather_daytypes.json

Station IDs are the 3- or 4-character ASOS identifiers listed at https://mesonet.agron.iastate.edu/request/download.phtml
"""
import csv, json, os, sys
from collections import defaultdict
import numpy as np
import requests

KT = 1.852
TYPES = ["calm/damp", "ordinary", "windy & dry", "red-flag"]
YEARS = (2005, 2024)
CACHE = "data/weather_daytypes.json"


def fetch(station, years=YEARS):
    path = f"data/asos_{station}.csv"
    if os.path.exists(path): return path
    os.makedirs("data", exist_ok=True)
    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=%s&data=sknt&data=drct&data=gust&data=relh&data=p01i&data=tmpf"
           "&year1=%d&month1=1&day1=1&year2=%d&month2=1&day2=1&tz=America%%2FNew_York&format=onlycomma&latlon=no&missing=M&trace=T&report_type=3"
           % (station, years[0], years[1] + 1))
    r = requests.get(url, timeout=900, headers={"User-Agent": "urban-fire-risk-model"})
    r.raise_for_status()
    if r.text.count("\n") < 1000:
        raise SystemExit(f"almost no observations returned for station {station!r}; check the station ID")
    open(path, "w").write(r.text)
    return path


def load(station):
    """-> dict of per-day arrays: date, w4, rh4, wind_from, dry, mean_wind, prec, prec_prev, temp_c"""
    days = defaultdict(list)
    for r in csv.DictReader(open(fetch(station))):
        try: s = float(r["sknt"]); rh = float(r["relh"])
        except ValueError: continue
        try: d = float(r["drct"])
        except ValueError: d = np.nan
        try: t = (float(r["tmpf"]) - 32) / 1.8
        except ValueError: t = np.nan
        p = 0.0 if r["p01i"] in ("M", "", "T") else float(r["p01i"])
        days[r["valid"][:10]].append((s * KT, d, rh, p, t))
    keys = sorted(days)
    prec = {k: sum(h[3] for h in days[k]) for k in keys}
    out = defaultdict(list)
    for i, k in enumerate(keys):
        h = np.array(days[k])
        if len(h) < 18: continue                       # too few observations to characterise the day
        o = np.argsort(-h[:, 0])[:4]
        dr = h[o, 1]; dr = dr[~np.isnan(dr)]
        wd = (np.degrees(np.arctan2(np.sin(np.radians(dr)).mean(), np.cos(np.radians(dr)).mean())) + 360) % 360 if len(dr) else np.nan
        prev = prec[keys[i - 1]] if i else 0.0
        out["date"].append(k); out["w4"].append(h[o[-1], 0]); out["rh4"].append(h[o, 2].mean()); out["wind_from"].append(wd)
        out["dry"].append(prec[k] < 0.05 and prev < 0.05); out["mean_wind"].append(h[:, 0].mean())
        out["prec"].append(prec[k]); out["prec_prev"].append(prev); out["temp_c"].append(np.nanmean(h[o, 4]))
    return {k: np.array(v) for k, v in out.items()}


def classify(D):
    """-> array of indices into TYPES"""
    t = np.full(len(D["w4"]), 1)
    t[~D["dry"]] = 0
    t[D["dry"] & (D["rh4"] <= 50) & (D["w4"] >= 32)] = 2
    t[D["dry"] & (D["rh4"] <= 50) & (D["w4"] >= 40)] = 3
    return t


def _rose(wd):
    return np.histogram((wd + 22.5) % 360, bins=np.arange(0, 361, 45))[0]


def summarize(station):
    D = load(station); t = classify(D); wd = D["wind_from"]; ok = ~np.isnan(wd)
    mon = np.array([int(d[5:7]) for d in D["date"]])
    alld = _rose(wd[ok]) + 1.0
    o = {"n_days": len(t), "types": {}, "by_month": {}}
    for i, nm in enumerate(TYPES):
        m = (t == i)
        h = _rose(wd[m & ok]) + (alld / alld.sum())                 # lightly smoothed
        if m.sum() < 20: h = h + 20 * alld / alld.sum()              # few days: shrink toward the all-day rose
        o["types"][nm] = {"p": float(m.mean()), "n": int(m.sum()), "dirs": [float(x) for x in h / h.sum()]}
    for mo in range(1, 13):
        o["by_month"][str(mo)] = [float(np.mean(t[mon == mo] == i)) for i in range(4)]
    return o


def day_types(station):
    """-> cached summary for a station: {"types": {name: {"p", "n", "dirs"}}, "by_month": {month: [p per type]}}"""
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if station not in cache:
        cache[station] = summarize(station)
        os.makedirs("data", exist_ok=True)
        json.dump(cache, open(CACHE, "w"), indent=1)
    return cache[station]


def describe(station):
    D = load(station); t = classify(D); n = len(t)
    w4, rh, dry = D["w4"], D["rh4"], D["dry"]
    print(f"== {station}: {n} days ({D['date'][0]} .. {D['date'][-1]}); mean hourly wind {D['mean_wind'].mean():.1f} km/h; dry days {dry.mean():.2f}")
    print("   W4 km/h percentiles 10/50/90/99:", np.round(np.percentile(w4, [10, 50, 90, 99]), 0))
    for i, nm in enumerate(TYPES):
        m = t == i
        rose = _rose(D["wind_from"][m & ~np.isnan(D["wind_from"])])
        print(f"   {nm:12s} {m.mean():7.4f} of days ({m.sum() / n * 365.25:5.1f}/yr)  wind from N/NE/E/SE/S/SW/W/NW: "
              f"{np.round(rose / max(rose.sum(), 1), 2)}")
    g = w4 >= 40
    print(f"   gale-wind days (W4 >= 40 km/h) by dryness rule, share of all days:")
    for lab, m in [("any", g), ("no rain that day", g & (D["prec"] < 0.05)), ("no rain that day or the day before", g & dry),
                   ("... and RH4 <= 50 (red-flag class)", g & dry & (rh <= 50)), ("... and RH4 <= 35", g & dry & (rh <= 35))]:
        print(f"      {lab:38s} {m.mean():.4f} ({m.sum()})")
    rf = t == 3
    if rf.any():
        print(f"   red-flag days: median temperature {np.nanmedian(D['temp_c'][rf]):.0f} C, {np.mean(D['temp_c'][rf] < 0):.0%} below freezing")
    bm = summarize(station)["by_month"]
    print("   red-flag share by month (%):", [round(100 * bm[str(mo)][3], 1) for mo in range(1, 13)])


if __name__ == "__main__":
    for st in sys.argv[1:] or ["BOS"]:
        describe(st); day_types(st)
