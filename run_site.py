"""Run the model for one or more sites listed in sites.json (see sites.example.json).

    python run_site.py NAME [NAME ...]

For each site this fetches OSM footprints and airport weather if they are not already in data/, samples
reach sets for the building at the site's coordinates, and writes data/results_<name>.pkl.  report.py turns
those into burn probabilities.

res["runs"] keys:
  (1.0, defense level)                 the house alone at that defense level
  (1.0, defense level, "block")        every building within block_radius_m at that level too
  (scale, "do nothing")                50% spread distance scaled: 0.77 = no adjustment for taller houses
                                       (HOUSE_FACTOR 1.0), 1.23 = the top of the physics range (1.6)
  (1.0, "do nothing", "wind25"/"wind75") windy-and-dry days with wind weight 0.25 / 0.75 instead of 0.5
  (1.0, "do nothing", "nolimit")       no travel-time limit on wind-driven days
  (1.0, "do nothing", "lost")          the LA curve for buildings actually lost (firefighters saved some)
Each value is a list of (day type, W, hit_window_edge, n_buildings, X) samples; X is the wildland
exposure (wildland.py), zero unless the site sets "wildland_burn_fraction".
"""
import json, pickle, sys, zlib
import numpy as np
import model, osm, weather, wildland

NS = 400     # samples per day type
LOST_CURVE = dict(D50_LA=3.53, K_LA=0.92, AREA_COEF=0.18, HOUSE_FACTOR=1.0)   # calib_fit3.py, outcome "lost"


def run(name, cfg):
    radius = cfg.get("radius_m", 2500)
    edge = radius - 300                 # reach sets touching this ring are flagged as truncated
    osm.ensure(name, cfg["lat"], cfg["lon"], radius)
    wx = weather.day_types(cfg["station"])["types"]
    polys, info = model.build(model.load(name), cfg["lat"], cfg["lon"])
    G = model.graph(polys, info, cache=f"data/graph_{name}.npz")
    t = model.find_target(polys, info, cfg.get("osm_way"))
    gaps = G["gap"][G["indptr"][t]:G["indptr"][t + 1]]
    block = np.hypot(*(G["cents"] - G["cents"][t]).T) <= cfg.get("block_radius_m", 100)
    block[t] = False
    desc = dict(n_buildings=len(polys), target_dist_m=round(float(polys[t].distance(model.Point(0, 0))), 1), area=round(info[t]["area"]),
                nearest=[round(float(g), 1) for g in np.sort(gaps)[:5]], n10=int((gaps < 10).sum()), n20=int((gaps < 20).sum()),
                n_block=int(block.sum()), n_block_houses=int((block & (G["area"] >= 60)).sum()), window_edge=edge, station=cfg["station"])
    wild = None; wild_h = None
    if "wildland_burn_fraction" in cfg:
        fxy, fpatch, fshare = wildland.patches(name, cfg["lat"], cfg["lon"], radius)
        patch, pw, fdist = wildland.exposure(G["cents"], fxy, fpatch)
        wild = (patch, pw)
        wild_h = wildland.daily_prob(cfg["wildland_burn_fraction"], {k: v["p"] for k, v in wx.items()})
        desc.update(forest_dist_m=round(float(fdist[t])) if np.isfinite(fdist[t]) else None, n_forest_edge=int((patch >= 0).sum()), forest_share=round(fshare, 2))
    print(name, desc, flush=True)
    res = {"desc": desc, "area": info[t]["area"], "runs": {}, "wild_h": wild_h}
    variants = [((1.0, d), {}, d, None) for d in model.DEFENSE]
    variants += [((1.0, d, "block"), {}, d, block) for d in model.DEFENSE if d != "do nothing"]
    variants += [((s, "do nothing"), dict(SPREAD_SCALE=s), "do nothing", None) for s in (0.77, 1.23)]
    variants += [((1.0, "do nothing", "wind25"), dict(TAIL_W={"windy & dry": 0.25}), "do nothing", None),
                 ((1.0, "do nothing", "wind75"), dict(TAIL_W={"windy & dry": 0.75}), "do nothing", None),
                 ((1.0, "do nothing", "nolimit"), dict(EVENT={}), "do nothing", None),
                 ((1.0, "do nothing", "lost"), LOST_CURVE, "do nothing", None)]
    for key, overrides, dname, blk in variants:
        saved = {k: getattr(model, k) for k in overrides}
        for k, v in overrides.items(): setattr(model, k, v)
        rng = np.random.default_rng(zlib.crc32(repr((name,) + key).encode()))
        recs = []
        for day in model.DAY_TYPES:
            rose = np.array(wx[day[0]]["dirs"])
            for _ in range(NS):
                wf = rng.choice(model.WIND_DIRS, p=rose) + rng.uniform(-22.5, 22.5)
                W, hit, n, X = model.sample_W(t, G, rng, day, wf, model.DEFENSE[dname], edge, blk, wild)
                recs.append((day[0], W, hit, n, X))
        for k, v in saved.items(): setattr(model, k, v)
        res["runs"][key] = recs
        print(" ", name, key, "done", flush=True)
    pickle.dump(res, open(f"data/results_{name}.pkl", "wb"))


if __name__ == "__main__":
    try:
        sites = json.load(open("sites.json"))
    except FileNotFoundError:
        raise SystemExit("no sites.json here; copy sites.example.json to sites.json and edit it")
    names = sys.argv[1:]
    if not names:
        raise SystemExit(f"usage: python run_site.py NAME [NAME ...]   (sites.json has: {', '.join(sites)})")
    for name in names:
        if name not in sites: raise SystemExit(f"{name!r} is not in sites.json")
        run(name, sites[name])
