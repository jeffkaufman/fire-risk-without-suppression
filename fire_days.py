"""Boston Logan weather on the days of three Massachusetts fires that beat a full fire-department response.
Used to place them in the day-type scheme of weather.py (W4 and RH4 as defined there).

  Chelsea, 1973-10-14: 18 blocks burned.
  Lynn, 1981-11-28: downtown mill district.
  Lawrence, 2008-01-21: 15-16 buildings.
"""
import time, numpy as np, requests
for lab, y, m, d in [("Chelsea 1973-10-14", 1973, 10, 14), ("Lynn 1981-11-28", 1981, 11, 28), ("Lawrence 2008-01-21", 2008, 1, 21)]:
    u = (f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=BOS&data=sknt&data=drct&data=gust&data=relh&year1={y}&month1={m}&day1={d}"
         f"&year2={y}&month2={m}&day2={d + 1}&tz=America%2FNew_York&format=onlycomma&latlon=no&missing=M&trace=T&report_type=3")
    rows = []
    for attempt in range(4):                       # the archive occasionally returns an empty body
        t = requests.get(u, timeout=120, headers={"User-Agent": "urban-fire-risk-model"}).text.strip().splitlines()[1:]
        rows = [r.split(",") for r in t]
        rows = [r for r in rows if len(r) > 5 and r[1].startswith(f"{y}-{m:02d}-{d:02d}") and r[2] != "M" and r[5] != "M"]
        if rows: break
        time.sleep(5)
    if not rows: print(lab, "no data returned"); continue
    s = np.array([float(r[2]) for r in rows]) * 1.852; rh = np.array([float(r[5]) for r in rows])
    g = [float(r[4]) * 1.852 for r in rows if r[4] != "M"]
    o = np.argsort(-s)[:4]
    print(f"{lab}: W4 {s[o[-1]]:.0f} km/h, RH4 {rh[o].mean():.0f}%, max gust {max(g) if g else float('nan'):.0f} km/h, "
          f"wind from {sorted(set(rows[i][3] for i in o))}")
