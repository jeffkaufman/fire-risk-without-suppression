"""Burn probabilities from the samples written by run_site.py.

    python report.py NAME [NAME ...] [--q0 6e-6] [--days 30]

P(burn in T days) = 1 - ( E_day[exp(-q0 * W)] * (1 - q0 * a_t * own) )^T
  q0  = escaped ignitions per 100 m2 of building footprint per day
  W   = footprint/100 m2 of the buildings whose fire would reach the house that day (sampled per day type)
  a_t = the house's own footprint/100 m2, own = its own-ignition multiplier at that defense level
"""
import argparse, json, pickle
import numpy as np
import model, weather

TYPES = [d[0] for d in model.DAY_TYPES]
DEF = list(model.DEFENSE)


def pburn(res, key, q0, T, probs):
    """probs: {day type: probability}"""
    own = model.DEFENSE[key[1]][3]; a_t = res["area"] / 100; s = 0
    for dt, p in probs.items():
        if p == 0: continue
        W = np.array([w for (d, w, h, n) in res["runs"][key] if d == dt])
        s += p * np.mean(np.exp(-q0 * W))
    return 1 - (s * (1 - q0 * a_t * own)) ** T


def pc(x):
    return f"{100 * x:.2f}%" if x >= 0.001 else f"{100 * x:.3f}%"


def report(name, q0, T):
    res = pickle.load(open(f"data/results_{name}.pkl", "rb"))
    wx = weather.day_types(res["desc"]["station"])
    annual = {k: v["p"] for k, v in wx["types"].items()}
    print(f"\n== {name}  ({res['desc']['station']} weather; {res['desc']['n_buildings']} buildings in window)")
    print("  day-type frequency:", {k: round(v, 4) for k, v in annual.items()})
    parts = {}
    for dt in TYPES:
        recs = [r for r in res["runs"][(1.0, "do nothing")] if r[0] == dt]
        W = np.array([r[1] for r in recs]); n = np.array([r[3] for r in recs])
        parts[dt] = (annual[dt] * np.mean(1 - np.exp(-q0 * W)), np.median(n), np.percentile(n, 90), np.mean([r[2] for r in recs]))
    own = q0 * res["area"] / 100; tot = sum(p[0] for p in parts.values()) + own
    print("  daily hazard share (do nothing): own ignition %.0f%%; " % (100 * own / tot) +
          "; ".join(f"{dt} {100 * p[0] / tot:.0f}% (reach-set median {p[1]:.0f}, p90 {p[2]:.0f}, hits window edge {p[3]:.2f})" for dt, p in parts.items()))
    print(f"  {T}-day period, annual-average weather, q0={q0:.0e}:   " + " / ".join(DEF))
    print("    house alone:                         " + " / ".join(pc(pburn(res, (1.0, d), q0, T, annual)) for d in DEF))
    print(f"    whole block ({res['desc']['n_block']} buildings) at that level: " +
          " / ".join(pc(pburn(res, (1.0, d) if d == "do nothing" else (1.0, d, "block"), q0, T, annual)) for d in DEF))
    print("  sensitivity, do nothing:")
    print("    q0 x1/3 and x3.3:                    " + " / ".join(pc(pburn(res, (1.0, "do nothing"), q, T, annual)) for q in (q0 / 3, q0 * 10 / 3)))
    print("    spread distances x0.8 and x1.25:     " + " / ".join(pc(pburn(res, (s, "do nothing"), q0, T, annual)) for s in (0.8, 1.25)))
    print("    calm and ordinary days at m = 1:     " + pc(pburn(res, (1.0, "do nothing", "m1"), q0, T, annual)))
    for lab, months in [("March", [3]), ("Dec-Apr", [12, 1, 2, 3, 4]), ("Jun-Sep", [6, 7, 8, 9])]:
        pr = {t: float(np.mean([wx["by_month"][str(m)][i] for m in months])) for i, t in enumerate(TYPES)}
        print(f"    weather of {lab:8s}:                " + " / ".join(pc(pburn(res, (1.0, d), q0, T, pr)) for d in DEF))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--q0", type=float, default=6e-6, help="escaped ignitions per 100 m2 footprint per day")
    ap.add_argument("--days", type=int, default=30)
    a = ap.parse_args()
    for name in a.names:
        report(name, a.q0, a.days)
