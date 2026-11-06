"""Run the model for one or more sites listed in sites.json (see sites.example.json).

    python run_site.py NAME [NAME ...]

For each site this fetches OSM footprints and airport weather if they are not already in data/, samples
reach sets for the building at the site's coordinates, and writes data/results_<name>.pkl.  report.py turns
those into burn probabilities.

res["runs"] keys:
  (spread scale, defense level)        the house alone at that defense level
  (1.0, defense level, "block")        every building within block_radius_m at that level too
  (1.0, "do nothing", "m1")            calm and ordinary days use the LA curve unscaled (m = 1)
Each value is a list of (day type, W, hit_window_edge, n_buildings) samples.
"""
import json, pickle, sys, zlib
import numpy as np
import model, osm, weather

NS = {"calm/damp": 1000, "ordinary": 1000, "windy & dry": 1000, "red-flag": 400}   # samples per day type


def run(name, cfg):
    radius = cfg.get("radius_m", 2500)
    edge = radius - 300                 # reach sets touching this ring are flagged as truncated
    osm.ensure(name, cfg["lat"], cfg["lon"], radius)
    wx = weather.day_types(cfg["station"])["types"]
    polys, info = model.build(model.load(name), cfg["lat"], cfg["lon"])
    nb, cents = model.neighbor_lists(polys)
    t = model.find_target(polys, info, cfg.get("osm_way"))
    gaps = nb[t][1]
    block = set(int(i) for i in np.where(np.hypot(*(cents - cents[t]).T) <= cfg.get("block_radius_m", 100))[0]) - {t}
    desc = dict(n_buildings=len(polys), target_dist_m=round(float(polys[t].distance(model.Point(0, 0))), 1), area=round(info[t]["area"]), nearest=[round(float(g), 1) for g in np.sort(gaps)[:5]],
                n10=int((gaps < 10).sum()), n20=int((gaps < 20).sum()), n_block=len(block), window_edge=edge, station=cfg["station"])
    print(name, desc, flush=True)
    res = {"desc": desc, "area": info[t]["area"], "runs": {}}
    variants = [((1.0, d), 1.0, d, None, None) for d in model.DEFENSE]
    variants += [((1.0, d, "block"), 1.0, d, block, None) for d in model.DEFENSE if d != "do nothing"]
    variants += [((s, "do nothing"), s, "do nothing", None, None) for s in (0.8, 1.25)]
    variants += [((1.0, "do nothing", "m1"), 1.0, "do nothing", None, 1.0)]
    for key, scale, dname, blk, mfloor in variants:
        model.SPREAD_SCALE = scale; model.M_FLOOR = mfloor
        rng = np.random.default_rng(zlib.crc32(repr((name,) + key).encode()))
        recs = []
        for day in model.DAY_TYPES:
            rose = np.array(wx[day[0]]["dirs"])
            for _ in range(NS[day[0]]):
                wf = rng.choice(model.WIND_DIRS, p=rose) + rng.uniform(-22.5, 22.5)
                W, hit, n = model.sample_W(t, nb, info, cents, rng, day, wf, model.DEFENSE[dname], edge, blk)
                recs.append((day[0], W, hit, n))
        res["runs"][key] = recs
        print(" ", name, key, "done", flush=True)
    model.SPREAD_SCALE = 1.0; model.M_FLOOR = None
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
